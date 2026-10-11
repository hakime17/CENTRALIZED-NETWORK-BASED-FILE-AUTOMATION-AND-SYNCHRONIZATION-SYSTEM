import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from audit_logger import AuditLogger
from file_rules import FileRuleManager
from operations import FileOperations, OperationResult, OperationStatus
from scanner import FileScanner, ScannedFile, SkippedFile
from stability_checker import FileStabilityChecker, StabilityStatus


@dataclass(frozen=True)
class FolderProblem:
    """A watched folder that could not be scanned (missing, not a folder, no access)."""

    folder: Path
    detail: str


@dataclass(frozen=True)
class PassReport:
    """The outcome of one pass over all the watched folders."""

    scanned_folders: int  # how many folders were scanned successfully
    moved: tuple[OperationResult, ...] = ()  # files moved
    failed: tuple[OperationResult, ...] = ()  # moves that failed (see each result's reason)
    planned: tuple[OperationResult, ...] = ()  # what a dry run would have moved
    in_place: int = 0  # files already in the folder they belong in
    waiting: int = 0  # files still changing; checked again on the next pass
    temporary: int = 0  # unfinished downloads and other temporary files
    unmatched: int = 0  # files with no rule, or no destination folder for their category
    unreadable: tuple[SkippedFile, ...] = ()  # files the scanner found but could not read
    folder_problems: tuple[FolderProblem, ...] = ()
    unrecorded: tuple[OperationResult, ...] = ()  # results that could not be written to the audit log
    audit_error: str | None = None  # why the pass stopped early, if the audit log failed


class AutomationEngine:
    # The folders scanned by default, under the user's home folder.
    PRIMARY_FOLDER_NAMES: tuple[str, ...] = ("Downloads", "Desktop", "Documents", "Pictures", "Music", "Videos")

    def __init__(
        self,
        watched_folders: Iterable[str | Path],
        destinations: Mapping[str, str | Path],
        scanner: FileScanner,
        stability_checker: FileStabilityChecker,
        rules: FileRuleManager,
        operations: FileOperations,
        audit_logger: AuditLogger,
    ) -> None:
        """Initialize the engine.

        The engine coordinates the other classes and does none of their work itself.
        Each call to run_once() makes one pass: it scans EVERY watched folder, asks
        the stability checker which files have stopped changing, looks up each
        stable file's category, moves it into that category's folder, and records
        the outcome in the audit log.

        Args:
            watched_folders: Every folder to scan, such as Downloads, Desktop,
                Pictures, Music and Videos. Only the top level of each is scanned.
                A leading "~" is expanded and repeated folders are scanned once.
            destinations: Which folder each category goes to, e.g. {"Pictures":
                "~/Pictures", "Archives": "~/Documents"}. A file whose category has
                no entry here is left where it is.
            scanner, stability_checker, rules, operations, audit_logger: The other
                classes. Dry-run mode is set on `operations`.

        Raises:
            ValueError: if watched_folders or destinations is empty.
        """
        folders = self._unique(Path(folder).expanduser() for folder in watched_folders)
        if not folders:
            raise ValueError("watched_folders must not be empty.")
        mapping = {category: Path(folder).expanduser() for category, folder in destinations.items()}
        if not mapping:
            raise ValueError("destinations must not be empty.")

        self.watched_folders: list[Path] = folders
        self.destinations: dict[str, Path] = mapping
        self.scanner: FileScanner = scanner
        self.stability_checker: FileStabilityChecker = stability_checker
        self.rules: FileRuleManager = rules
        self.operations: FileOperations = operations
        self.audit_logger: AuditLogger = audit_logger

    @staticmethod
    def primary_folders(home: str | Path | None = None) -> list[Path]:
        """The standard folders to scan, under `home` (the user's home folder by default).

        Note: on some Windows computers Desktop, Documents and Pictures are redirected
        (for example to OneDrive), so these paths may need replacing in the settings.
        """
        base = Path(home).expanduser() if home is not None else Path.home()
        return [base / name for name in AutomationEngine.PRIMARY_FOLDER_NAMES]

    @staticmethod
    def default_destinations(home: str | Path | None = None) -> dict[str, Path]:
        """Where each category goes by default: into the matching primary folder.

        Archives have no primary folder of their own, so they go into Documents.
        """
        base = Path(home).expanduser() if home is not None else Path.home()
        return {
            "Pictures": base / "Pictures",
            "Music": base / "Music",
            "Videos": base / "Videos",
            "Documents": base / "Documents",
            "Archives": base / "Documents",
        }

    def run_once(self) -> PassReport:
        """Make one pass over all the watched folders and report what happened.

        The engine does not loop or wait. Whatever calls it decides how often.
        A folder that cannot be scanned is reported and the pass carries on. If the
        audit log cannot be written the pass stops, because moving files without an
        audit trail defeats the point; the report says which results are unrecorded.
        """
        snapshots: list[ScannedFile] = []
        unreadable: list[SkippedFile] = []
        folder_problems: list[FolderProblem] = []
        scanned = 0

        for folder in self.watched_folders:
            try:
                result = self.scanner.scan(folder)
            except OSError as exc:  # missing folder, not a folder, no access
                folder_problems.append(FolderProblem(folder, str(exc)))
                continue
            scanned += 1
            snapshots.extend(result.files)
            unreadable.extend(result.skipped)

        # One call for all folders together: the checker forgets any file missing from a call.
        assessments = self.stability_checker.check(snapshots)

        moved: list[OperationResult] = []
        failed: list[OperationResult] = []
        planned: list[OperationResult] = []
        unrecorded: list[OperationResult] = []
        in_place = waiting = temporary = unmatched = 0
        audit_error: str | None = None

        for assessment in assessments:
            if assessment.status is StabilityStatus.TEMPORARY:
                temporary += 1
                continue
            if assessment.status is StabilityStatus.CHANGING:
                waiting += 1
                continue

            category = self.rules.get_category(assessment.path)
            destination_folder = None if category is None else self.destinations.get(category)
            if destination_folder is None:
                unmatched += 1
                continue

            result = self.operations.move_file(assessment.path, destination_folder)

            if result.status is OperationStatus.SKIPPED:  # already in the right folder
                in_place += 1
                continue
            if result.status is OperationStatus.DRY_RUN:  # nothing happened, so nothing to audit
                planned.append(result)
                continue

            (moved if result.status is OperationStatus.COMPLETED else failed).append(result)
            try:
                self.audit_logger.record(result, size_bytes=assessment.file.size)
            except OSError as exc:
                unrecorded.append(result)
                audit_error = str(exc)
                break  # the remaining files are picked up on the next pass

        return PassReport(
            scanned_folders=scanned,
            moved=tuple(moved),
            failed=tuple(failed),
            planned=tuple(planned),
            in_place=in_place,
            waiting=waiting,
            temporary=temporary,
            unmatched=unmatched,
            unreadable=tuple(unreadable),
            folder_problems=tuple(folder_problems),
            unrecorded=tuple(unrecorded),
            audit_error=audit_error,
        )

    @staticmethod
    def _unique(folders: Iterable[Path]) -> list[Path]:
        """Drop repeated folders (however they are spelled), keeping the first of each."""
        seen: set[str] = set()
        unique: list[Path] = []
        for folder in folders:
            key = os.path.normcase(os.path.abspath(folder))
            if key not in seen:
                seen.add(key)
                unique.append(folder)
        return unique
