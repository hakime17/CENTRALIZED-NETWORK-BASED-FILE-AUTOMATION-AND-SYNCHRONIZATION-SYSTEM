import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OperationType(str, Enum):
    MOVE = "move"
    COPY = "copy"


class OperationStatus(str, Enum):
    COMPLETED = "completed"  # the file was moved or copied
    DRY_RUN = "dry_run"  # nothing was touched; this is what would have happened
    SKIPPED = "skipped"  # nothing needed doing (see detail)
    FAILED = "failed"  # the operation could not be done (see reason and detail)


class FailureReason(str, Enum):
    """Why an operation FAILED, in a form the engine and the audit log can act on."""

    SOURCE_MISSING = "source_missing"  # not an existing file, or it vanished mid-operation
    UNSUPPORTED_SOURCE = "unsupported_source"  # e.g. a symbolic link
    DESTINATION_INVALID = "destination_invalid"  # not a folder, bad path, or the folder cannot be created
    NO_FREE_NAME = "no_free_name"  # every "name (n).ext" in the folder was already taken
    PERMISSION_DENIED = "permission_denied"  # includes a file locked by another program on Windows
    IO_ERROR = "io_error"  # any other operating-system error, such as a full disk


@dataclass(frozen=True)
class OperationResult:
    """What happened (or would have happened) to one file."""

    operation: OperationType
    source: Path
    destination: Path | None  # final file path; None if it was never worked out
    status: OperationStatus
    detail: str = ""  # the reason for SKIPPED, or the error text for FAILED
    reason: FailureReason | None = None  # set only when status is FAILED


class FileOperations:
    # How many "name (1).ext", "name (2).ext", ... to try before giving up.
    MAX_NAME_ATTEMPTS: int = 1000

    def __init__(self, dry_run: bool = False) -> None:
        """Initialize the file operations.

        This is the only class that changes files on disk. It never overwrites an
        existing file (a name clash becomes "name (1).ext"), and it never raises
        for a problem with one file: the problem is returned in the result, so one
        bad file cannot stop the rest of the automation.

        Known limitation: the free name is chosen just before the transfer, so a
        file that another program creates at that exact name in between could be
        overwritten. On a single-user folder such as Downloads the window is tiny.

        Args:
            dry_run: If True, nothing is created, moved or copied. Each call still
                works out and returns exactly what it would have done.
        """
        self.dry_run: bool = dry_run

    def move_file(self, source: str | Path, destination_folder: str | Path) -> OperationResult:
        """Move `source` into `destination_folder`, creating the folder if needed.

        A file that is already in the destination folder is SKIPPED, not renamed.
        """
        return self._transfer(OperationType.MOVE, Path(source), Path(destination_folder))

    def copy_file(self, source: str | Path, destination_folder: str | Path) -> OperationResult:
        """Copy `source` into `destination_folder`, creating the folder if needed.

        Copying into the folder the file is already in makes "name (1).ext".
        """
        return self._transfer(OperationType.COPY, Path(source), Path(destination_folder))

    def _transfer(self, operation: OperationType, source: Path, folder: Path) -> OperationResult:
        """Shared logic for move and copy."""
        destination: Path | None = None
        started = False
        try:
            # Symbolic links are refused: move would relocate the link itself while
            # copy would duplicate its target, and a relative link breaks when moved.
            if source.is_symlink():
                return self._failed(
                    operation, source, None, FailureReason.UNSUPPORTED_SOURCE, "Symbolic links are not supported."
                )
            if not source.is_file():
                return self._failed(
                    operation, source, None, FailureReason.SOURCE_MISSING, "Source is not an existing file."
                )

            if operation is OperationType.MOVE and self._same_folder(source.parent, folder):
                return OperationResult(
                    operation, source, source, OperationStatus.SKIPPED, "File is already in the destination folder."
                )

            if folder.exists() and not folder.is_dir():
                return self._failed(
                    operation, source, None, FailureReason.DESTINATION_INVALID, "Destination is not a folder."
                )

            destination = self._unique_destination(folder / source.name)
            if destination is None:
                return self._failed(
                    operation, source, None, FailureReason.NO_FREE_NAME,
                    f"No free name found for {source.name} in {folder}.",
                )

            if self.dry_run:
                return OperationResult(operation, source, destination, OperationStatus.DRY_RUN)

            folder.mkdir(parents=True, exist_ok=True)
            started = True
            if operation is OperationType.MOVE:
                shutil.move(str(source), str(destination))
            else:
                shutil.copy2(source, destination)
            return OperationResult(operation, source, destination, OperationStatus.COMPLETED)

        except (OSError, ValueError) as exc:  # ValueError: e.g. a null byte in a path
            if started and destination is not None:
                self._remove_leftover(operation, source, destination)
            return self._failed(operation, source, destination, self._reason_for(exc, source), str(exc))

    @staticmethod
    def _failed(
        operation: OperationType,
        source: Path,
        destination: Path | None,
        reason: FailureReason,
        detail: str,
    ) -> OperationResult:
        return OperationResult(operation, source, destination, OperationStatus.FAILED, detail, reason)

    @staticmethod
    def _reason_for(exc: Exception, source: Path) -> FailureReason:
        """Translate an exception into a FailureReason."""
        if isinstance(exc, PermissionError):
            return FailureReason.PERMISSION_DENIED
        if isinstance(exc, FileNotFoundError):
            # Either the source vanished, or part of the destination path does not exist.
            return FailureReason.DESTINATION_INVALID if source.exists() else FailureReason.SOURCE_MISSING
        if isinstance(exc, (NotADirectoryError, ValueError)):
            return FailureReason.DESTINATION_INVALID
        return FailureReason.IO_ERROR

    def _unique_destination(self, path: Path) -> Path | None:
        """Return `path`, or "name (1).ext", "name (2).ext", ... if it is taken.

        Returns None if no free name was found within MAX_NAME_ATTEMPTS.
        """
        if not os.path.lexists(path):  # lexists: a broken link still occupies the name
            return path
        for counter in range(1, self.MAX_NAME_ATTEMPTS + 1):
            candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
            if not os.path.lexists(candidate):
                return candidate
        return None

    @staticmethod
    def _same_folder(first: Path, second: Path) -> bool:
        """True if both paths are the same existing folder.

        Uses samefile, which compares the real folders on disk, so differences in
        spelling (".", "..", letter case, links, Windows short names) do not matter.
        A destination that does not exist yet cannot be the source's own folder.
        """
        if not second.exists():
            return False
        return first.samefile(second)

    @staticmethod
    def _remove_leftover(operation: OperationType, source: Path, destination: Path) -> None:
        """Clean up after a failed transfer so it does not leave a stray duplicate.

        The destination name was free before we started, so anything there came
        from this attempt. A failed move can leave a copy behind (for example when
        the original is locked by another program and cannot be deleted). It is
        only removed while the original still exists, so the data is never lost.
        """
        try:
            if destination.exists() and (operation is OperationType.COPY or source.exists()):
                destination.unlink()
        except OSError:
            pass  # nothing more can be done; the FAILED result already reports the problem
