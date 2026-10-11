"""Tests for automation_engine.py. Run with:  python -m pytest test_automation_engine.py -v"""
import dataclasses
from pathlib import Path

import pytest

from audit_logger import AuditLogger
from automation_engine import AutomationEngine, FolderProblem, PassReport
from file_rules import FileRuleManager
from operations import FailureReason, FileOperations, OperationStatus
from scanner import FileScanner
from stability_checker import FileStabilityChecker, StabilityPolicy

PERIOD = 5.0
CLIENT = "LAB-PC-01"


# ---------------------------------------------------------------- helpers

class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    for name in AutomationEngine.PRIMARY_FOLDER_NAMES:
        (home / name).mkdir(parents=True)
    return home


def put(folder: Path, name: str, content: str = "x") -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def audit_path(home: Path) -> Path:
    return home / "audit" / "log.jsonl"


def make_engine(home: Path, *, dry_run=False, folders=None, destinations=None, log_path=None,
                clock=None, checker=None, scanner=None):
    clock = clock or FakeClock()
    engine = AutomationEngine(
        folders if folders is not None else AutomationEngine.primary_folders(home),
        destinations if destinations is not None else AutomationEngine.default_destinations(home),
        scanner or FileScanner(),
        checker or FileStabilityChecker(StabilityPolicy(PERIOD), clock=clock),
        FileRuleManager({"zip": "Archives"}),  # explicit, so the tests do not depend on the default list
        FileOperations(dry_run=dry_run),
        AuditLogger(log_path or audit_path(home), client_id=CLIENT),
    )
    return engine, clock


def two_passes(engine, clock) -> PassReport:
    """First pass: every file is new, so nothing moves. Then time passes and the second pass acts."""
    engine.run_once()
    clock.advance(PERIOD)
    return engine.run_once()


def records(home: Path):
    return AuditLogger(audit_path(home), client_id="reader").read_all()


# ================================================================ construction

def test_primary_folders_are_the_six_standard_ones_in_order(tmp_path):
    folders = AutomationEngine.primary_folders(tmp_path)
    assert folders == [tmp_path / n for n in ["Downloads", "Desktop", "Documents", "Pictures", "Music", "Videos"]]


def test_primary_folders_default_to_the_home_folder():
    assert AutomationEngine.primary_folders()[0] == Path.home() / "Downloads"
    assert len(AutomationEngine.primary_folders()) == 6


def test_default_destinations_send_each_category_to_its_primary_folder(tmp_path):
    assert AutomationEngine.default_destinations(tmp_path) == {
        "Pictures": tmp_path / "Pictures",
        "Music": tmp_path / "Music",
        "Videos": tmp_path / "Videos",
        "Documents": tmp_path / "Documents",
        "Archives": tmp_path / "Documents",
    }


def test_the_engine_watches_every_primary_folder_by_default(tmp_path):
    engine, _ = make_engine(make_home(tmp_path))
    assert [p.name for p in engine.watched_folders] == ["Downloads", "Desktop", "Documents", "Pictures", "Music", "Videos"]


@pytest.mark.parametrize("kwargs", [{"folders": []}, {"destinations": {}}])
def test_empty_folders_or_destinations_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError, match="must not be empty"):
        make_engine(make_home(tmp_path), **kwargs)


def test_folders_and_destinations_may_be_given_as_text_and_become_paths(tmp_path):
    engine, _ = make_engine(tmp_path, folders=[str(tmp_path / "in")], destinations={"Pictures": str(tmp_path / "out")})
    assert engine.watched_folders == [tmp_path / "in"]
    assert engine.destinations == {"Pictures": tmp_path / "out"}
    assert all(isinstance(p, Path) for p in engine.watched_folders)


def test_a_generator_of_folders_is_accepted(tmp_path):
    engine, _ = make_engine(tmp_path, folders=(tmp_path / n for n in ["a", "b"]))
    assert [p.name for p in engine.watched_folders] == ["a", "b"]


def test_a_leading_tilde_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    engine, _ = make_engine(tmp_path, folders=["~/Downloads"], destinations={"Pictures": "~/Pictures"})
    assert engine.watched_folders == [tmp_path / "Downloads"]
    assert engine.destinations["Pictures"] == tmp_path / "Pictures"


def test_a_repeated_folder_is_scanned_once_however_it_is_spelled(tmp_path):
    home = make_home(tmp_path)
    downloads = home / "Downloads"
    engine, _ = make_engine(home, folders=[downloads, downloads, downloads / ".", str(downloads) + "/"])
    assert engine.watched_folders == [downloads]


def test_the_first_of_each_repeated_folder_keeps_its_position(tmp_path):
    engine, _ = make_engine(tmp_path, folders=[tmp_path / "b", tmp_path / "a", tmp_path / "b"])
    assert [p.name for p in engine.watched_folders] == ["b", "a"]


# ================================================================ the first pass

def test_the_first_pass_moves_nothing_because_every_file_is_new(tmp_path):
    home = make_home(tmp_path)
    put(home / "Downloads", "report.pdf")
    put(home / "Downloads", "photo.jpg")
    engine, _ = make_engine(home)

    report = engine.run_once()
    assert report.moved == () and report.failed == () and report.planned == ()
    assert report.waiting == 2
    assert report.scanned_folders == 6
    assert (home / "Downloads" / "report.pdf").exists()
    assert not audit_path(home).exists()  # nothing happened, so nothing is audited


# ================================================================ sorting into the primary folders

@pytest.mark.parametrize(
    "start_folder, name, destination_folder",
    [
        ("Downloads", "report.pdf", "Documents"),
        ("Downloads", "photo.jpg", "Pictures"),
        ("Downloads", "song.mp3", "Music"),
        ("Downloads", "clip.mp4", "Videos"),
        ("Downloads", "backup.zip", "Documents"),  # archives go into Documents
        ("Downloads", "notes.txt", "Documents"),
        ("Downloads", "essay.docx", "Documents"),
        ("Desktop", "pic.png", "Pictures"),
        ("Pictures", "invoice.pdf", "Documents"),
        ("Documents", "song.mp3", "Music"),
        ("Videos", "notes.txt", "Documents"),
        ("Music", "photo.jpg", "Pictures"),
        ("Downloads", "PHOTO.JPG", "Pictures"),
        ("Downloads", "Ñandi 文件.pdf", "Documents"),
    ],
)
def test_a_file_goes_to_the_primary_folder_for_its_type(tmp_path, start_folder, name, destination_folder):
    home = make_home(tmp_path)
    source = put(home / start_folder, name, "12345")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    expected = home / destination_folder / name
    assert expected.read_text() == "12345"
    assert not source.exists()
    (moved,) = report.moved
    assert moved.status is OperationStatus.COMPLETED
    assert moved.source == source
    assert moved.destination == expected
    assert report.failed == () and report.unmatched == 0


def test_every_primary_folder_is_scanned(tmp_path):
    home = make_home(tmp_path)
    for folder in ["Downloads", "Desktop", "Pictures", "Music", "Videos"]:
        put(home / folder, f"from_{folder}.pdf")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert report.scanned_folders == 6
    assert len(report.moved) == 5
    assert sorted(p.name for p in (home / "Documents").iterdir()) == sorted(f"from_{f}.pdf" for f in ["Downloads", "Desktop", "Pictures", "Music", "Videos"])


@pytest.mark.parametrize(
    "folder, name",
    [("Pictures", "holiday.jpg"), ("Music", "song.mp3"), ("Videos", "clip.mp4"), ("Documents", "doc.docx"), ("Documents", "backup.zip")],
)
def test_a_file_already_in_the_right_folder_is_left_alone_and_not_audited(tmp_path, folder, name):
    home = make_home(tmp_path)
    source = put(home / folder, name, "stay")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert source.read_text() == "stay"
    assert report.in_place == 1
    assert report.moved == () and report.failed == ()
    assert not audit_path(home).exists()
    assert [p.name for p in (home / folder).iterdir()] == [name]  # no "name (1)" copy appeared


def test_a_name_clash_is_numbered_and_the_existing_file_survives(tmp_path):
    home = make_home(tmp_path)
    put(home / "Documents", "report.pdf", "ORIGINAL")
    put(home / "Downloads", "report.pdf", "NEWER")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert (home / "Documents" / "report.pdf").read_text() == "ORIGINAL"
    assert (home / "Documents" / "report (1).pdf").read_text() == "NEWER"
    assert report.moved[0].destination.name == "report (1).pdf"


# ================================================================ files that are left alone

@pytest.mark.parametrize("name", ["setup.exe", "shortcut.lnk", "README", "data.unknownext"])
def test_files_with_no_rule_are_left_where_they_are(tmp_path, name):
    home = make_home(tmp_path)
    put(home / "Downloads", name)
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert (home / "Downloads" / name).exists()
    assert report.unmatched == 1
    assert report.moved == () and not audit_path(home).exists()


def test_a_category_with_no_destination_folder_is_left_alone(tmp_path):
    home = make_home(tmp_path)
    destinations = AutomationEngine.default_destinations(home)
    del destinations["Archives"]
    put(home / "Downloads", "backup.zip")
    put(home / "Downloads", "report.pdf")
    engine, clock = make_engine(home, destinations=destinations)

    report = two_passes(engine, clock)

    assert (home / "Downloads" / "backup.zip").exists()
    assert (home / "Documents" / "report.pdf").exists()
    assert report.unmatched == 1
    assert len(report.moved) == 1


def test_unfinished_downloads_are_never_moved(tmp_path):
    home = make_home(tmp_path)
    download = put(home / "Downloads", "movie.mp4.crdownload")
    engine, clock = make_engine(home)

    engine.run_once()
    clock.advance(1000)
    report = engine.run_once()

    assert download.exists()
    assert report.temporary == 1
    assert report.moved == ()


def test_subfolders_and_their_contents_are_not_scanned(tmp_path):
    home = make_home(tmp_path)
    nested = put(home / "Pictures" / "sub", "nested.pdf")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert nested.exists()
    assert report.moved == () and report.in_place == 0 and report.unmatched == 0


def test_hidden_files_and_system_junk_are_ignored(tmp_path):
    home = make_home(tmp_path)
    hidden = put(home / "Downloads", ".hidden.jpg")
    junk = put(home / "Desktop", "desktop.ini")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert hidden.exists() and junk.exists()
    assert (report.moved, report.in_place, report.waiting, report.temporary, report.unmatched) == ((), 0, 0, 0, 0)


def test_a_file_that_changes_between_passes_has_to_settle_again(tmp_path):
    home = make_home(tmp_path)
    source = put(home / "Downloads", "growing.mp4", "part one")
    engine, clock = make_engine(home)

    engine.run_once()
    clock.advance(PERIOD)
    source.write_text("part one and a lot more data")  # the download grew
    report = engine.run_once()

    assert source.exists()
    assert report.waiting == 1 and report.moved == ()

    clock.advance(PERIOD)
    assert len(engine.run_once().moved) == 1


# ================================================================ repeated passes

def test_later_passes_do_nothing_more_and_audit_nothing_more(tmp_path):
    home = make_home(tmp_path)
    put(home / "Downloads", "report.pdf")
    put(home / "Downloads", "photo.jpg")
    put(home / "Downloads", "setup.exe")
    engine, clock = make_engine(home)
    two_passes(engine, clock)
    logged = len(records(home))

    for _ in range(3):  # the moved files are new to the checker, then settle in place
        clock.advance(PERIOD)
        report = engine.run_once()
        assert report.moved == () and report.failed == ()

    assert report.in_place == 2 and report.unmatched == 1 and report.waiting == 0
    assert len(records(home)) == logged == 2


# ================================================================ missing and unusable folders

def test_a_missing_folder_is_reported_and_the_pass_carries_on(tmp_path):
    home = make_home(tmp_path)
    (home / "Music").rmdir()
    (home / "Videos").rmdir()
    put(home / "Downloads", "photo.jpg")
    engine, clock = make_engine(home)

    report = two_passes(engine, clock)

    assert {p.folder.name for p in report.folder_problems} == {"Music", "Videos"}
    assert all(isinstance(p, FolderProblem) and p.detail for p in report.folder_problems)
    assert report.scanned_folders == 4
    assert (home / "Pictures" / "photo.jpg").exists()


def test_files_can_still_be_moved_into_a_destination_folder_that_does_not_exist_yet(tmp_path):
    home = make_home(tmp_path)
    (home / "Music").rmdir()
    put(home / "Downloads", "song.mp3")
    engine, clock = make_engine(home)

    two_passes(engine, clock)

    assert (home / "Music" / "song.mp3").exists()


def test_a_watched_path_that_is_a_file_is_reported_as_a_problem(tmp_path):
    home = make_home(tmp_path)
    not_a_folder = put(home, "iamafile")
    put(home / "Downloads", "report.pdf")
    engine, clock = make_engine(home, folders=[home / "Downloads", not_a_folder])

    report = two_passes(engine, clock)

    assert [p.folder for p in report.folder_problems] == [not_a_folder]
    assert report.scanned_folders == 1
    assert len(report.moved) == 1


def test_a_pass_with_every_folder_missing_still_returns_a_report(tmp_path):
    engine, _ = make_engine(tmp_path, folders=[tmp_path / "nope1", tmp_path / "nope2"])
    report = engine.run_once()
    assert report.scanned_folders == 0
    assert len(report.folder_problems) == 2
    assert report.moved == ()


def test_unreadable_files_are_reported(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    put(home / "Downloads", "notes.txt")
    put(home / "Downloads", "report.pdf")
    engine, clock = make_engine(home)

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "notes.txt":
            raise PermissionError("denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    report = two_passes(engine, clock)

    assert [s.path.name for s in report.unreadable] == ["notes.txt"]
    assert report.unreadable[0].reason.value == "permission_denied"
    assert [m.source.name for m in report.moved] == ["report.pdf"]


# ================================================================ how the stability checker is used

def test_the_checker_gets_one_combined_list_for_all_folders_each_pass(tmp_path):
    home = make_home(tmp_path)
    for folder in ["Downloads", "Desktop", "Pictures"]:
        put(home / folder, f"{folder}.pdf")
    clock = FakeClock()
    checker = FileStabilityChecker(StabilityPolicy(PERIOD), clock=clock)
    calls = []
    real_check = checker.check

    def spy(files):
        files = list(files)
        calls.append(files)
        return real_check(files)

    checker.check = spy
    engine, _ = make_engine(home, clock=clock, checker=checker)
    engine.run_once()

    assert len(calls) == 1
    assert {f.path.parent.name for f in calls[0]} == {"Downloads", "Desktop", "Pictures"}


# ================================================================ dry run

def test_a_dry_run_reports_the_plan_and_changes_nothing(tmp_path):
    home = make_home(tmp_path)
    source = put(home / "Downloads", "report.pdf")
    put(home / "Pictures", "holiday.jpg")
    engine, clock = make_engine(home, dry_run=True)

    report = two_passes(engine, clock)

    assert source.exists()
    assert report.moved == () and report.failed == ()
    (planned,) = report.planned
    assert planned.status is OperationStatus.DRY_RUN
    assert planned.destination == home / "Documents" / "report.pdf"
    assert report.in_place == 1
    assert not audit_path(home).exists()  # dry runs are not audited


# ================================================================ the audit trail

def test_every_move_is_audited_with_client_size_and_type(tmp_path):
    home = make_home(tmp_path)
    source = put(home / "Downloads", "Report.PDF", "12345")
    engine, clock = make_engine(home)

    two_passes(engine, clock)

    (record,) = records(home)
    assert record.client_id == CLIENT
    assert record.operation == "move"
    assert record.status == "completed"
    assert record.source == str(source)
    assert record.destination == str(home / "Documents" / "Report.PDF")
    assert record.file_name == "Report.PDF"
    assert record.extension == "pdf"
    assert record.size_bytes == 5


def test_audit_records_follow_the_order_the_files_were_processed(tmp_path):
    home = make_home(tmp_path)
    put(home / "Downloads", "b.pdf")
    put(home / "Downloads", "a.jpg")
    put(home / "Desktop", "c.mp3")
    engine, clock = make_engine(home)

    two_passes(engine, clock)

    assert [r.file_name for r in records(home)] == ["a.jpg", "b.pdf", "c.mp3"]  # folder order, then name


def test_the_audit_log_can_be_searched_for_where_a_file_went(tmp_path):
    home = make_home(tmp_path)
    put(home / "Downloads", "thesis.pdf")
    engine, clock = make_engine(home)
    two_passes(engine, clock)

    (found,) = AuditLogger(audit_path(home), client_id="x").search(name="thesis", client_id=CLIENT)
    assert found.location == str(home / "Documents" / "thesis.pdf")


# ================================================================ failures

def test_a_failed_move_is_reported_and_audited_and_the_pass_continues(tmp_path):
    home = make_home(tmp_path)
    (home / "Pictures").rmdir()
    put(home, "Pictures")  # "Pictures" is now a file, so nothing can be moved into it
    folders = [f for f in AutomationEngine.primary_folders(home) if f.name != "Pictures"]
    put(home / "Downloads", "photo.jpg")
    put(home / "Downloads", "report.pdf")
    engine, clock = make_engine(home, folders=folders)

    report = two_passes(engine, clock)

    (failed,) = report.failed
    assert failed.source.name == "photo.jpg"
    assert failed.reason is FailureReason.DESTINATION_INVALID
    assert (home / "Downloads" / "photo.jpg").exists()
    assert [m.source.name for m in report.moved] == ["report.pdf"]  # the pass went on
    assert sorted((r.file_name, r.status, r.reason) for r in records(home)) == [
        ("photo.jpg", "failed", "destination_invalid"),
        ("report.pdf", "completed", None),
    ]


def test_a_file_that_vanishes_after_the_scan_is_a_failed_move(tmp_path):
    home = make_home(tmp_path)
    put(home / "Downloads", "vanish.pdf")
    put(home / "Downloads", "stay.pdf")

    class VanishingScanner(FileScanner):
        def scan(self, directory):
            result = super().scan(directory)
            vanish = Path(directory) / "vanish.pdf"
            if vanish.exists() and self.armed:
                vanish.unlink()
            return result

    scanner = VanishingScanner()
    scanner.armed = False
    engine, clock = make_engine(home, scanner=scanner)
    engine.run_once()
    clock.advance(PERIOD)
    scanner.armed = True

    report = engine.run_once()

    (failed,) = report.failed
    assert failed.reason is FailureReason.SOURCE_MISSING
    assert [m.source.name for m in report.moved] == ["stay.pdf"]
    assert {r.file_name: r.status for r in records(home)} == {"vanish.pdf": "failed", "stay.pdf": "completed"}


def test_an_unwritable_audit_log_stops_the_pass_and_reports_the_unrecorded_move(tmp_path):
    home = make_home(tmp_path)
    (home / "adir").mkdir()  # a folder where the log file should be
    first = put(home / "Downloads", "a.pdf")
    second = put(home / "Downloads", "b.pdf")
    engine, clock = make_engine(home, log_path=home / "adir")

    report = two_passes(engine, clock)

    assert report.audit_error
    assert [r.source.name for r in report.unrecorded] == ["a.pdf"]
    assert [m.source.name for m in report.moved] == ["a.pdf"]  # it did happen, it just is not recorded
    assert not first.exists() and (home / "Documents" / "a.pdf").exists()
    assert second.exists()  # the pass stopped before touching the next file


def test_the_pass_picks_up_where_it_stopped_once_the_log_works_again(tmp_path):
    home = make_home(tmp_path)
    (home / "adir").mkdir()
    put(home / "Downloads", "a.pdf")
    put(home / "Downloads", "b.pdf")
    engine, clock = make_engine(home, log_path=home / "adir")
    two_passes(engine, clock)

    engine.audit_logger = AuditLogger(audit_path(home), client_id=CLIENT)  # the problem is fixed
    clock.advance(PERIOD)
    report = engine.run_once()

    assert report.audit_error is None and report.unrecorded == ()
    assert [m.source.name for m in report.moved] == ["b.pdf"]
    assert [r.file_name for r in records(home)] == ["b.pdf"]  # a.pdf's move stays unrecorded


def test_a_healthy_pass_reports_no_audit_problem(tmp_path):
    home = make_home(tmp_path)
    put(home / "Downloads", "a.pdf")
    engine, clock = make_engine(home)
    report = two_passes(engine, clock)
    assert report.audit_error is None
    assert report.unrecorded == ()


# ================================================================ the report object

def test_the_report_has_sensible_defaults():
    report = PassReport(scanned_folders=0)
    assert report.moved == report.failed == report.planned == report.unreadable == ()
    assert report.folder_problems == report.unrecorded == ()
    assert (report.in_place, report.waiting, report.temporary, report.unmatched) == (0, 0, 0, 0)
    assert report.audit_error is None


def test_the_report_and_its_parts_are_immutable(tmp_path):
    home = make_home(tmp_path)
    (home / "Music").rmdir()
    engine, _ = make_engine(home)
    report = engine.run_once()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(report, "waiting", 5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(report.folder_problems[0], "detail", "x")
    assert isinstance(report.moved, tuple) and isinstance(report.folder_problems, tuple)


# ================================================================ custom settings

def test_the_engine_works_with_any_folders_and_destinations_not_just_the_standard_ones(tmp_path):
    inbox = tmp_path / "Inbox"
    sorted_pdfs = tmp_path / "Sorted" / "PDFs"
    put(inbox, "report.pdf")
    put(inbox, "photo.jpg")
    engine, clock = make_engine(tmp_path, folders=[inbox], destinations={"Documents": sorted_pdfs})

    report = two_passes(engine, clock)

    assert (sorted_pdfs / "report.pdf").exists()
    assert (inbox / "photo.jpg").exists()  # no destination for Pictures in these settings
    assert report.unmatched == 1
