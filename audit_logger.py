import json
import logging
import socket
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from operations import OperationResult, OperationStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditRecord:
    """One line of the audit log: what happened to one file, on which client, and when."""

    timestamp: str  # UTC, ISO 8601, e.g. "2026-10-03T14:05:21.123+00:00"
    client_id: str  # the machine that performed the operation
    operation: str  # "move" or "copy"
    status: str  # "completed", "dry_run", "skipped" or "failed"
    source: str  # full path the file was at
    file_name: str  # the file's name at the source (stored on its own, see from_result)
    destination: str | None = None  # full path it went to
    destination_name: str | None = None  # its final name; differs from file_name after a clash
    detail: str = ""
    reason: str | None = None  # the failure reason when status is "failed"
    # Newest fields go last so existing positional uses of AuditRecord keep working.
    extension: str = ""  # the file type in lower case without the dot, e.g. "pdf"; "" if none
    size_bytes: int | None = None  # None when the caller did not supply a size

    @classmethod
    def from_result(
        cls,
        result: OperationResult,
        timestamp: str,
        client_id: str,
        size_bytes: int | None = None,
    ) -> "AuditRecord":
        """Build a record from a FileOperations result.

        `size_bytes` is not part of the result, so the caller supplies it (the
        scanner already measured it). The file type is worked out from the name.

        The file names are stored separately from the paths on purpose. The client
        works them out with its own operating system's rules, so a Central Node
        running on another system can search by name without having to parse
        paths written in a different style (C:\\Users\\... versus /home/...).
        """
        return cls(
            timestamp=timestamp,
            client_id=client_id,
            operation=result.operation.value,
            status=result.status.value,
            source=str(result.source),
            file_name=result.source.name,
            extension=result.source.suffix.lower().lstrip("."),
            size_bytes=size_bytes,
            destination=None if result.destination is None else str(result.destination),
            destination_name=None if result.destination is None else result.destination.name,
            detail=result.detail,
            reason=None if result.reason is None else result.reason.value,
        )

    @property
    def location(self) -> str:
        """Where the file is as far as this record knows.

        The destination if the operation completed, otherwise the source (a failed,
        skipped or dry-run operation left the file where it was). For a copy both
        files exist; this reports the new one.
        """
        if self.status == OperationStatus.COMPLETED and self.destination is not None:
            return self.destination
        return self.source

    def matches(
        self,
        name: str | None = None,
        client_id: str | None = None,
        status: str | OperationStatus | None = None,
    ) -> bool:
        """True if this record satisfies every filter that was given.

        Args:
            name: Part of a file name. Ignores letter case, and is checked against
                both the original name and the final name, so "report" finds
                "report.pdf" and also its renamed copy "report (1).pdf".
                Blank means no name filter.
            client_id: A client's identifier. Ignores letter case.
            status: Only records with this status, e.g. OperationStatus.COMPLETED.

        This works on any collection of records, so the Central Node can reuse it
        on records gathered from many clients.
        """
        if client_id is not None and self.client_id.casefold() != client_id.strip().casefold():
            return False
        if status is not None and self.status != status:
            return False
        needle = (name or "").strip().casefold()
        if needle:
            names = [self.file_name, self.destination_name or ""]
            if not any(needle in candidate.casefold() for candidate in names):
                return False
        return True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AuditRecord":
        """Rebuild a record from a dictionary. Unknown keys are ignored.

        Raises:
            ValueError: if `data` is not a dictionary.
            TypeError: if a required field is missing.
        """
        if not isinstance(data, dict):
            raise ValueError("An audit line must be a JSON object.")
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


class AuditLogger:
    def __init__(
        self,
        log_path: str | Path,
        client_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the audit logger.

        Every record is appended as one line of JSON ("JSON Lines"), so the log is
        readable in any text editor and a crash can only damage the last line.
        The file and its folders are created on first use.

        Not safe for several threads or programs writing to the same log at once.

        Args:
            log_path: Where the log file lives.
            client_id: Identifies this machine in every record it writes, so records
                from many clients can be told apart (and searched) at the Central
                Node. Defaults to this computer's name. The Central Node is expected
                to assign real identifiers once the network phase exists.
            clock: A function returning the current time as a datetime. Tests pass
                a fake clock. Defaults to the current UTC time.

        Raises:
            ValueError: if client_id is given but blank.
        """
        if client_id is not None and not client_id.strip():
            raise ValueError("client_id must not be blank.")
        self.client_id: str = client_id.strip() if client_id is not None else socket.gethostname()
        self.log_path: Path = Path(log_path)
        self._clock: Callable[[], datetime] = clock if clock is not None else lambda: datetime.now(timezone.utc)

    def record(self, result: OperationResult, size_bytes: int | None = None) -> AuditRecord:
        """Append one operation result to the log and return the record written.

        `size_bytes` is the file's size, if the caller knows it (for example from
        the scanner's snapshot). It is optional and left empty when not given.

        Whatever result is passed is recorded, dry runs included; the "status"
        field tells them apart. Timestamps are always stored in UTC.

        Raises:
            ValueError: if size_bytes is negative.
            OSError: if the log cannot be written. This is deliberately not hidden:
                the caller decides whether to carry on without an audit trail.
        """
        if size_bytes is not None and size_bytes < 0:
            raise ValueError("size_bytes must not be negative.")
        entry = AuditRecord.from_result(result, self._timestamp(), self.client_id, size_bytes)
        line = json.dumps(entry.to_dict(), ensure_ascii=False)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" keeps the file identical on Windows; backslashreplace means an
        # odd file name can never make the write fail.
        with open(self.log_path, "a", encoding="utf-8", errors="backslashreplace", newline="\n") as file:
            file.write(line + "\n")
        return entry

    def record_all(self, results: Iterable[OperationResult]) -> list[AuditRecord]:
        """Record several results, in order. Returns the records written."""
        return [self.record(result) for result in results]

    def read_all(self) -> list[AuditRecord]:
        """Return every readable record in the log, oldest first.

        A missing log gives an empty list. Blank lines are ignored, and a line that
        is damaged (for example the last line after a crash) is skipped and
        reported in the log instead of making the whole file unreadable.
        """
        if not self.log_path.exists():
            return []

        records: list[AuditRecord] = []
        with open(self.log_path, "r", encoding="utf-8", errors="replace") as file:
            for number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(AuditRecord.from_dict(json.loads(line)))
                except (ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
                    logger.warning("Skipping unreadable audit line %d in %s: %s", number, self.log_path, exc)
        return records

    def search(
        self,
        name: str | None = None,
        client_id: str | None = None,
        status: str | OperationStatus | None = None,
    ) -> list[AuditRecord]:
        """Find records by file name, client, status, or any combination.

        See AuditRecord.matches for how each filter works. With no filters every
        record is returned. Results are newest first, so the first result says
        where a file ended up most recently (see AuditRecord.location).
        """
        found = [record for record in self.read_all() if record.matches(name, client_id, status)]
        found.sort(key=lambda record: record.timestamp)  # stable: ties keep file order
        found.reverse()
        return found

    def _timestamp(self) -> str:
        """The clock's current time as a UTC ISO 8601 string."""
        return self._clock().astimezone(timezone.utc).isoformat(timespec="milliseconds")
