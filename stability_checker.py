import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence

from scanner import ScannedFile


class StabilityStatus(str, Enum):
    """How safe a file is to process right now."""

    CHANGING = "changing"  # new, or its size/modified time changed recently
    STABLE = "stable"  # unchanged for the whole stability period: eligible for processing
    TEMPORARY = "temporary"  # has a temporary-download extension: never eligible


@dataclass(frozen=True)
class FileAssessment:
    """The checker's verdict on one file in one scan."""

    file: ScannedFile
    status: StabilityStatus
    unchanged_for: float  # seconds the size and modified time have stayed the same

    @property
    def path(self) -> Path:
        return self.file.path


@dataclass(frozen=True)
class _Record:
    """What the checker remembers about one file between scans."""

    size: int
    modified_time: float
    unchanged_since: float  # clock reading when these exact values were first seen

    def matches(self, file: ScannedFile) -> bool:
        return self.size == file.size and self.modified_time == file.modified_time


class StabilityPolicy:
    # Extensions that mean "still downloading or temporary". Lower-case, with the dot.
    DEFAULT_TEMPORARY_SUFFIXES: frozenset[str] = frozenset(
        {".crdownload", ".part", ".tmp", ".download"}
    )

    def __init__(
        self,
        stability_period: float = 5.0,
        temporary_suffixes: Sequence[str] | str | None = None,
    ) -> None:
        """Settings for the checker. Plain data, so the Central Node can send it later.

        Args:
            stability_period: Seconds a file's size and modified time must stay the
                same before it counts as STABLE. 0 means STABLE on first sight.
            temporary_suffixes: Extensions that mark a file as TEMPORARY, e.g.
                [".crdownload", ".part"]. The leading dot is optional and case is
                ignored. A single string is accepted. Defaults to
                DEFAULT_TEMPORARY_SUFFIXES; an empty list means no file is temporary.

        Raises:
            ValueError: if stability_period is negative.
        """
        if stability_period < 0:
            raise ValueError("stability_period must not be negative.")
        if isinstance(temporary_suffixes, str):
            temporary_suffixes = [temporary_suffixes]
        suffixes = self.DEFAULT_TEMPORARY_SUFFIXES if temporary_suffixes is None else temporary_suffixes

        self.stability_period: float = stability_period
        cleaned = (s.strip().lower().lstrip(".") for s in suffixes)
        self.temporary_suffixes: frozenset[str] = frozenset(f".{c}" for c in cleaned if c)


class FileStabilityChecker:
    def __init__(
        self,
        policy: StabilityPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the checker.

        The checker is stateful: it remembers each file's last size and modified
        time between calls to check().

        Args:
            policy: Settings to apply. Defaults to StabilityPolicy().
            clock: A function returning the current time in seconds. Tests pass a
                fake clock so they never have to wait. Must never run backwards.
        """
        self.policy: StabilityPolicy = policy if policy is not None else StabilityPolicy()
        self._clock: Callable[[], float] = clock
        self._records: dict[Path, _Record] = {}

    def check(self, files: Iterable[ScannedFile]) -> list[FileAssessment]:
        """Assess every file from one scan and return one verdict per file, in input order.

        Pass everything one scan found. Files that were tracked before but are
        missing from this call are forgotten, so a file that disappears and comes
        back starts again as CHANGING.

        STABLE means "unchanged for the stability period and eligible for
        processing". It does not prove a download is complete: a stalled download
        also stops changing.
        """
        now = self._clock()  # read once, so every file in this scan shares one moment
        records: dict[Path, _Record] = {}
        assessments: list[FileAssessment] = []

        for file in files:
            if file.path.suffix.lower() in self.policy.temporary_suffixes:
                assessments.append(FileAssessment(file, StabilityStatus.TEMPORARY, 0.0))
                continue

            previous = self._records.get(file.path)
            if previous is not None and previous.matches(file):
                unchanged_since = previous.unchanged_since
            else:
                unchanged_since = now  # new file, or it just changed: restart the timer

            records[file.path] = _Record(file.size, file.modified_time, unchanged_since)

            unchanged_for = now - unchanged_since
            if unchanged_for >= self.policy.stability_period:
                status = StabilityStatus.STABLE
            else:
                status = StabilityStatus.CHANGING
            assessments.append(FileAssessment(file, status, unchanged_for))

        self._records = records
        return assessments
