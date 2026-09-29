import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from stat import FILE_ATTRIBUTE_HIDDEN
from typing import Sequence

logger = logging.getLogger(__name__)


class SkipReason(str, Enum):
    """Why a file that should have been reported could not be scanned."""

    PERMISSION_DENIED = "permission_denied"
    IO_ERROR = "io_error"


@dataclass(frozen=True)
class ScannedFile:
    """A snapshot of one file at the moment it was scanned."""

    path: Path
    size: int  # bytes
    modified_time: float  # seconds since the epoch (st_mtime)


@dataclass(frozen=True)
class SkippedFile:
    """A file the scanner found but could not read, with the reason why."""

    path: Path
    reason: SkipReason
    detail: str = ""


@dataclass(frozen=True)
class ScanResult:
    """Everything one scan produced: readable files plus the ones that failed."""

    files: tuple[ScannedFile, ...]
    skipped: tuple[SkippedFile, ...]


class FileScanner:
    # Files nobody wants organized. Exact names only, compared case-insensitively.
    DEFAULT_IGNORED_NAMES: frozenset[str] = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})

    def __init__(
        self,
        include_hidden: bool = False,
        ignore_names: Sequence[str] | str | None = None,
        follow_symlinks: bool = False,
    ) -> None:
        """Initialize the scanner.

        The scanner is stateless: it remembers nothing between scans.

        Args:
            include_hidden: If True, files whose names start with "." are included.
            ignore_names: Exact file names to never report, e.g. ["Thumbs.db"].
                Case-insensitive. Defaults to DEFAULT_IGNORED_NAMES. Exact names
                only, on purpose: extension-based rules such as ".crdownload"
                belong to FileStabilityChecker, which must see those files to
                report them as not ready.
            follow_symlinks: If False (default), symbolic links are skipped, so the
                scanner never reports a file that lives outside the scanned folder.
                If True, links to files are reported using the target's size and time.
        """
        if isinstance(ignore_names, str):
            ignore_names = [ignore_names]
        names = self.DEFAULT_IGNORED_NAMES if ignore_names is None else ignore_names

        self.include_hidden: bool = include_hidden
        self.ignore_names: frozenset[str] = frozenset(n.strip().lower() for n in names if n.strip())
        self.follow_symlinks: bool = follow_symlinks

    def scan(self, directory: str | Path) -> ScanResult:
        """Scan the files directly inside `directory`.

        Returns a ScanResult with two lists, both sorted by path:
          files:   snapshots of every readable file.
          skipped: files that should have been reported but could not be read
                   (permissions, I/O errors), each with a reason. Files skipped
                   on purpose (hidden, ignored names, symlinks, folders) and
                   files that vanished mid-scan are not listed.

        Raises:
            FileNotFoundError: if the directory does not exist.
            NotADirectoryError: if the path is not a directory.
            PermissionError: if the directory itself cannot be read.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files: list[ScannedFile] = []
        skipped: list[SkippedFile] = []
        try:
            for item in directory.iterdir():
                outcome = self._inspect(item)
                if isinstance(outcome, ScannedFile):
                    files.append(outcome)
                elif isinstance(outcome, SkippedFile):
                    skipped.append(outcome)
        except PermissionError as exc:
            raise PermissionError(f"Permission denied reading directory: {directory}") from exc

        return ScanResult(
            files=tuple(sorted(files, key=lambda f: f.path)),
            skipped=tuple(sorted(skipped, key=lambda s: s.path)),
        )

    def _inspect(self, path: Path) -> ScannedFile | SkippedFile | None:
        """Snapshot one path. None means it was deliberately left out."""
        name = path.name.lower()
        if not self.include_hidden and name.startswith("."):
            return None
        if name in self.ignore_names:
            return None

        try:
            if path.is_symlink() and not self.follow_symlinks:
                logger.debug("Skipping symbolic link: %s", path)
                return None
            if not path.is_file():  # folders, broken links, special files
                return None
            stat = path.stat()
            # Windows marks hidden files with an attribute, not a leading dot.
            # The attribute does not exist on other systems, so this is a no-op there.
            if not self.include_hidden and getattr(stat, "st_file_attributes", 0) & FILE_ATTRIBUTE_HIDDEN:
                return None
        except PermissionError as exc:
            logger.warning("Permission denied, skipping file: %s", path)
            return SkippedFile(path, SkipReason.PERMISSION_DENIED, str(exc))
        except FileNotFoundError:
            # Deleted or moved between listing and reading: normal, not an error.
            logger.debug("File vanished during scan: %s", path)
            return None
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return SkippedFile(path, SkipReason.IO_ERROR, str(exc))

        return ScannedFile(path=path, size=stat.st_size, modified_time=stat.st_mtime)
