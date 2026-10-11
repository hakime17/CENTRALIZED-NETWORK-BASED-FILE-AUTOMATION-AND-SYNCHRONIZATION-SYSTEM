"""Tests for audit_logger.py. Run with:  python -m pytest test_audit_logger.py -v"""
import dataclasses
import json
import logging
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from audit_logger import AuditLogger, AuditRecord
from operations import (
    FailureReason,
    FileOperations,
    OperationResult,
    OperationStatus,
    OperationType,
)
from scanner import FileScanner

COMPLETED = OperationStatus.COMPLETED
DRY_RUN = OperationStatus.DRY_RUN
SKIPPED = OperationStatus.SKIPPED
FAILED = OperationStatus.FAILED

T0 = datetime(2026, 10, 3, 9, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- helpers

def clock_at(minutes: float = 0, microseconds: int = 0):
    moment = T0 + timedelta(minutes=minutes, microseconds=microseconds)
    return lambda: moment


def moved(name: str, folder: str = "Documents", final_name: str | None = None) -> OperationResult:
    return OperationResult(
        OperationType.MOVE, Path(name), Path(folder) / (final_name or name), COMPLETED
    )


def make_logger(tmp_path: Path, client_id: str = "LAB-PC-01", minutes: float = 0, name: str = "audit.jsonl") -> AuditLogger:
    return AuditLogger(tmp_path / name, client_id=client_id, clock=clock_at(minutes))


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def record_dicts(path: Path) -> list[dict]:
    return [json.loads(line) for line in lines_of(path)]


def names(records) -> list[tuple[str, str]]:
    return [(r.client_id, r.file_name) for r in records]


def make_record(**overrides) -> AuditRecord:
    values = dict(
        timestamp="2026-10-03T09:00:00.000+00:00",
        client_id="LAB-PC-01",
        operation="move",
        status="completed",
        source="in/a.txt",
        file_name="a.txt",
    )
    values.update(overrides)
    return AuditRecord(**values)


# ================================================================ AuditRecord

def test_from_result_for_a_completed_move():
    result = OperationResult(OperationType.MOVE, Path("in/Report.pdf"), Path("Docs/Report (1).pdf"), COMPLETED)
    record = AuditRecord.from_result(result, "2026-10-03T09:00:00.000+00:00", "LAB-PC-01", size_bytes=2048)
    assert record == AuditRecord(
        timestamp="2026-10-03T09:00:00.000+00:00",
        client_id="LAB-PC-01",
        operation="move",
        status="completed",
        source=str(Path("in/Report.pdf")),
        file_name="Report.pdf",
        destination=str(Path("Docs/Report (1).pdf")),
        destination_name="Report (1).pdf",
        detail="",
        reason=None,
        extension="pdf",
        size_bytes=2048,
    )


def test_from_result_for_a_failed_operation_has_no_destination():
    result = OperationResult(
        OperationType.COPY, Path("gone.txt"), None, FAILED, "Source is not an existing file.", FailureReason.SOURCE_MISSING
    )
    record = AuditRecord.from_result(result, "t", "c")
    assert record.operation == "copy"
    assert record.status == "failed"
    assert record.destination is None
    assert record.destination_name is None
    assert record.reason == "source_missing"
    assert record.detail == "Source is not an existing file."


@pytest.mark.parametrize("status", [SKIPPED, DRY_RUN])
def test_from_result_keeps_skipped_and_dry_run_statuses(status):
    result = OperationResult(OperationType.MOVE, Path("a.txt"), Path("a.txt"), status, "note")
    record = AuditRecord.from_result(result, "t", "c")
    assert record.status == status.value
    assert record.reason is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("report.PDF", "pdf"),
        ("photo.JPG", "jpg"),
        ("essay.docx", "docx"),
        ("song.mp3", "mp3"),
        ("archive.tar.gz", "gz"),
        ("Mixed.Case.XlSx", "xlsx"),
        ("README", ""),
        (".env", ""),
        ("weird.", ""),
    ],
)
def test_extension_is_lower_case_without_the_dot(name, expected):
    assert AuditRecord.from_result(moved(name), "t", "c").extension == expected


def test_size_is_empty_unless_supplied():
    assert AuditRecord.from_result(moved("a.txt"), "t", "c").size_bytes is None
    assert AuditRecord.from_result(moved("a.txt"), "t", "c", size_bytes=0).size_bytes == 0


def test_file_names_are_stored_separately_from_the_paths():
    record = AuditRecord.from_result(moved("deep/er/Thesis.docx", "X/Y"), "t", "c")
    assert record.file_name == "Thesis.docx"
    assert record.destination_name == "Thesis.docx"


# ---------------------------------------------------------------- location

def test_location_of_a_completed_operation_is_the_destination():
    record = make_record(status="completed", destination="Docs/a.txt", destination_name="a.txt")
    assert record.location == "Docs/a.txt"


@pytest.mark.parametrize("status", ["failed", "skipped", "dry_run"])
def test_location_otherwise_is_the_source(status):
    record = make_record(status=status, destination="Docs/a.txt")
    assert record.location == "in/a.txt"


def test_location_falls_back_to_the_source_if_there_is_no_destination():
    assert make_record(status="completed", destination=None).location == "in/a.txt"


# ---------------------------------------------------------------- dictionaries

def test_to_dict_is_json_ready_and_keeps_field_order():
    data = make_record().to_dict()
    assert list(data) == [
        "timestamp", "client_id", "operation", "status", "source", "file_name",
        "destination", "destination_name", "detail", "reason", "extension", "size_bytes",
    ]
    assert json.loads(json.dumps(data)) == data


def test_dictionary_round_trip():
    record = make_record(destination="d/a.txt", destination_name="a.txt", detail="x", reason="io_error", extension="txt", size_bytes=5)
    assert AuditRecord.from_dict(record.to_dict()) == record


def test_from_dict_ignores_unknown_keys():
    data = {**make_record().to_dict(), "future_field": 123}
    assert AuditRecord.from_dict(data) == make_record()


def test_lines_from_before_extension_and_size_existed_still_load():
    data = {"timestamp": "t", "client_id": "c", "operation": "move", "status": "completed", "source": "s", "file_name": "a.txt"}
    record = AuditRecord.from_dict(data)
    assert record.extension == ""
    assert record.size_bytes is None


def test_from_dict_requires_the_core_fields():
    with pytest.raises(TypeError):
        AuditRecord.from_dict({"operation": "move"})


@pytest.mark.parametrize("not_a_dict", [[1, 2], "text", 5, None])
def test_from_dict_rejects_things_that_are_not_objects(not_a_dict):
    with pytest.raises(ValueError):
        AuditRecord.from_dict(not_a_dict)


def test_records_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(make_record(), "client_id", "other")


# ---------------------------------------------------------------- matches

def test_a_record_matches_when_no_filters_are_given():
    assert make_record().matches()


@pytest.mark.parametrize("needle", ["report", "REPORT", "  Report  ", "port.p", "Report.pdf"])
def test_name_filter_is_a_case_insensitive_partial_match(needle):
    assert make_record(file_name="Report.pdf").matches(name=needle)


def test_name_filter_rejects_names_that_do_not_contain_it():
    assert not make_record(file_name="Report.pdf").matches(name="invoice")


def test_name_filter_also_checks_the_final_name():
    record = make_record(file_name="Report.pdf", destination_name="Report (1).pdf")
    assert record.matches(name="(1)")
    assert not make_record(file_name="Report.pdf", destination_name=None).matches(name="(1)")


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_name_means_no_name_filter(blank):
    assert make_record().matches(name=blank)


def test_name_filter_handles_non_english_names():
    record = make_record(file_name="Ñandi 文件.txt")
    assert record.matches(name="ñANDI")
    assert record.matches(name="文件")


def test_client_filter_is_exact_and_ignores_case():
    record = make_record(client_id="LAB-PC-01")
    assert record.matches(client_id="lab-pc-01")
    assert record.matches(client_id="  LAB-PC-01 ")
    assert not record.matches(client_id="LAB-PC-0")
    assert not record.matches(client_id="LAB")
    assert not record.matches(client_id="LAB-PC-02")


@pytest.mark.parametrize("status", ["completed", COMPLETED])
def test_status_filter_accepts_text_or_enum(status):
    assert make_record(status="completed").matches(status=status)
    assert not make_record(status="failed").matches(status=status)


def test_all_filters_must_match_together():
    record = make_record(file_name="Report.pdf", client_id="LAB-PC-01", status="completed")
    assert record.matches(name="report", client_id="lab-pc-01", status=COMPLETED)
    assert not record.matches(name="report", client_id="lab-pc-02", status=COMPLETED)
    assert not record.matches(name="invoice", client_id="lab-pc-01", status=COMPLETED)
    assert not record.matches(name="report", client_id="lab-pc-01", status=FAILED)


def test_a_record_from_a_windows_client_is_searchable_on_any_system():
    record = make_record(
        client_id="LAB-PC-09",
        source=r"C:\Users\Ada\Downloads\Thesis Final.docx",
        file_name="Thesis Final.docx",
        destination=r"C:\Users\Ada\Documents\Thesis Final.docx",
        destination_name="Thesis Final.docx",
    )
    assert record.matches(name="thesis", client_id="lab-pc-09")
    assert record.location == r"C:\Users\Ada\Documents\Thesis Final.docx"


# ================================================================ AuditLogger: client id

def test_client_id_defaults_to_the_computer_name(tmp_path):
    assert AuditLogger(tmp_path / "a.jsonl").client_id == socket.gethostname()


def test_client_id_is_trimmed(tmp_path):
    assert AuditLogger(tmp_path / "a.jsonl", client_id="  Lab 3  ").client_id == "Lab 3"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_client_id_is_rejected(tmp_path, blank):
    with pytest.raises(ValueError, match="client_id"):
        AuditLogger(tmp_path / "a.jsonl", client_id=blank)


def test_every_record_carries_the_clients_id(tmp_path):
    logger = make_logger(tmp_path, client_id="LAB-PC-07")
    logger.record(moved("a.txt"))
    logger.record(moved("b.txt"))
    assert [d["client_id"] for d in record_dicts(logger.log_path)] == ["LAB-PC-07", "LAB-PC-07"]


# ================================================================ AuditLogger: writing

def test_record_writes_one_json_line(tmp_path):
    logger = make_logger(tmp_path)
    returned = logger.record(moved("Report.pdf", "Docs"), size_bytes=10)

    (data,) = record_dicts(logger.log_path)
    assert data == {
        "timestamp": "2026-10-03T09:00:00.000+00:00",
        "client_id": "LAB-PC-01",
        "operation": "move",
        "status": "completed",
        "source": str(Path("Report.pdf")),
        "file_name": "Report.pdf",
        "destination": str(Path("Docs/Report.pdf")),
        "destination_name": "Report.pdf",
        "detail": "",
        "reason": None,
        "extension": "pdf",
        "size_bytes": 10,
    }
    assert returned == AuditRecord.from_dict(data)


def test_the_log_and_its_folders_are_created_on_first_use(tmp_path):
    logger = AuditLogger(tmp_path / "logs" / "deep" / "audit.jsonl", client_id="c", clock=clock_at())
    assert not logger.log_path.parent.exists()
    logger.record(moved("a.txt"))
    assert logger.log_path.exists()


def test_records_are_appended_not_overwritten(tmp_path):
    make_logger(tmp_path).record(moved("a.txt"))
    make_logger(tmp_path).record(moved("b.txt"))  # a second logger on the same file
    assert [d["file_name"] for d in record_dicts(tmp_path / "audit.jsonl")] == ["a.txt", "b.txt"]


def test_record_all_writes_in_order_and_returns_the_records(tmp_path):
    logger = make_logger(tmp_path)
    written = logger.record_all([moved("a.txt"), moved("b.txt"), moved("c.txt")])
    assert [r.file_name for r in written] == ["a.txt", "b.txt", "c.txt"]
    assert logger.read_all() == written


def test_record_all_with_nothing_writes_nothing(tmp_path):
    logger = make_logger(tmp_path)
    assert logger.record_all([]) == []
    assert not logger.log_path.exists()


def test_every_status_is_recorded_including_dry_runs(tmp_path):
    logger = make_logger(tmp_path)
    results = [
        moved("a.txt"),
        OperationResult(OperationType.MOVE, Path("b.txt"), Path("b.txt"), SKIPPED, "already there"),
        OperationResult(OperationType.COPY, Path("c.txt"), Path("Out/c.txt"), DRY_RUN),
        OperationResult(OperationType.MOVE, Path("d.txt"), None, FAILED, "nope", FailureReason.PERMISSION_DENIED),
    ]
    logger.record_all(results)
    assert [r.status for r in logger.read_all()] == ["completed", "skipped", "dry_run", "failed"]


# ---------------------------------------------------------------- timestamps

def test_timestamps_have_millisecond_precision_in_utc(tmp_path):
    logger = AuditLogger(tmp_path / "a.jsonl", client_id="c", clock=clock_at(microseconds=123456))
    assert logger.record(moved("a.txt")).timestamp == "2026-10-03T09:00:00.123+00:00"


def test_other_time_zones_are_converted_to_utc(tmp_path):
    lagos = timezone(timedelta(hours=1))
    logger = AuditLogger(tmp_path / "a.jsonl", client_id="c", clock=lambda: datetime(2026, 10, 3, 15, 0, 0, tzinfo=lagos))
    assert logger.record(moved("a.txt")).timestamp == "2026-10-03T14:00:00.000+00:00"


def test_the_default_clock_gives_the_current_utc_time(tmp_path):
    logger = AuditLogger(tmp_path / "a.jsonl", client_id="c")
    stamped = datetime.fromisoformat(logger.record(moved("a.txt")).timestamp)
    assert stamped.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - stamped) < timedelta(minutes=1)


# ---------------------------------------------------------------- size

def test_size_is_stored_when_supplied(tmp_path):
    logger = make_logger(tmp_path)
    logger.record(moved("a.txt"), size_bytes=5)
    logger.record(moved("empty.txt"), size_bytes=0)  # an empty file is a real size
    logger.record(moved("unknown.txt"))
    assert [r.size_bytes for r in logger.read_all()] == [5, 0, None]


@pytest.mark.parametrize("bad", [-1, -500])
def test_negative_size_is_rejected_and_nothing_is_written(tmp_path, bad):
    logger = make_logger(tmp_path)
    logger.record(moved("a.txt"))
    with pytest.raises(ValueError, match="negative"):
        logger.record(moved("b.txt"), size_bytes=bad)
    assert len(lines_of(logger.log_path)) == 1


# ---------------------------------------------------------------- awkward content

def test_non_english_file_names_stay_readable_in_the_file(tmp_path):
    logger = make_logger(tmp_path)
    logger.record(moved("Ñandi é 文件 😀.txt"))
    assert "Ñandi é 文件 😀" in logger.log_path.read_bytes().decode("utf-8")
    assert logger.read_all()[0].file_name == "Ñandi é 文件 😀.txt"


def test_a_message_with_line_breaks_still_takes_one_line(tmp_path):
    logger = make_logger(tmp_path)
    message = "line one\nline two\r\nline three"
    logger.record(OperationResult(OperationType.COPY, Path("a.txt"), None, FAILED, message, FailureReason.IO_ERROR))
    assert len(lines_of(logger.log_path)) == 1
    assert logger.read_all()[0].detail == message


def test_an_invalid_character_in_a_name_cannot_make_the_write_fail(tmp_path):
    logger = make_logger(tmp_path)
    result = OperationResult(OperationType.MOVE, Path("bad\udcffname.txt"), None, FAILED, "x", FailureReason.IO_ERROR)
    logger.record(result)  # must not raise
    assert len(logger.read_all()) == 1


def test_lines_end_with_a_plain_line_feed_on_every_system(tmp_path):
    logger = make_logger(tmp_path)
    logger.record_all([moved("a.txt"), moved("b.txt")])
    raw = logger.log_path.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r\n" not in raw


def test_a_log_that_cannot_be_written_raises(tmp_path):
    folder = tmp_path / "adir"
    folder.mkdir()
    logger = AuditLogger(folder, client_id="c", clock=clock_at())  # a folder is not a log file
    with pytest.raises(OSError):
        logger.record(moved("a.txt"))


# ================================================================ AuditLogger: reading

def test_a_missing_log_reads_as_empty(tmp_path):
    assert make_logger(tmp_path).read_all() == []


def test_records_read_back_oldest_first_and_equal_what_was_written(tmp_path):
    logger = make_logger(tmp_path)
    written = logger.record_all([moved("a.txt"), moved("b.txt")])
    assert logger.read_all() == written


def test_blank_lines_and_windows_line_endings_are_accepted(tmp_path):
    logger = make_logger(tmp_path)
    good = json.dumps(make_record().to_dict())
    logger.log_path.write_bytes(f"{good}\r\n\r\n   \r\n{good}\r\n".encode("utf-8"))
    assert len(logger.read_all()) == 2


def test_damaged_lines_are_skipped_and_reported(tmp_path, caplog):
    logger = make_logger(tmp_path)
    good = json.dumps(make_record().to_dict())
    extra = json.dumps({**make_record().to_dict(), "future_field": 1})
    logger.log_path.write_text(
        "\n".join([good, "{not json", "[1, 2]", json.dumps({"operation": "move"}), extra, '{"timestamp": "t", "operation": "copy"']) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="audit_logger"):
        records = logger.read_all()

    assert len(records) == 2  # the good line and the one with an extra field
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 4
    assert "line 2" in messages[0]
    assert str(logger.log_path) in messages[0]


def test_bytes_that_are_not_valid_text_do_not_stop_reading(tmp_path, caplog):
    logger = make_logger(tmp_path)
    good = json.dumps(make_record().to_dict()).encode("utf-8")
    logger.log_path.write_bytes(good + b"\n\xff\xfe broken \xc3\n" + good + b"\n")
    with caplog.at_level(logging.WARNING, logger="audit_logger"):
        assert len(logger.read_all()) == 2


# ================================================================ AuditLogger: search

@pytest.fixture
def shared_log(tmp_path):
    """One log holding three clients' records, as the Central Node would."""
    path = tmp_path / "central.jsonl"
    lab1 = AuditLogger(path, client_id="LAB-PC-01", clock=clock_at(0))
    lab2 = AuditLogger(path, client_id="Lab-PC-02", clock=clock_at(10))
    lab1_later = AuditLogger(path, client_id="LAB-PC-01", clock=clock_at(20))

    lab1.record(moved("Report.pdf", "Documents"))
    lab2.record(moved("Report.pdf", "Documents", final_name="Report (1).pdf"))
    lab2.record(OperationResult(OperationType.MOVE, Path("gone.txt"), None, FAILED, "missing", FailureReason.SOURCE_MISSING))
    lab1_later.record(OperationResult(OperationType.COPY, Path("photo.JPG"), Path("Pictures/photo.JPG"), COMPLETED))
    return AuditLogger(path, client_id="whoever-is-searching")


def test_search_by_name_ignores_case_and_matches_part_of_the_name(shared_log):
    assert names(shared_log.search(name="report")) == [("Lab-PC-02", "Report.pdf"), ("LAB-PC-01", "Report.pdf")]
    assert names(shared_log.search(name="  PHOTO ")) == [("LAB-PC-01", "photo.JPG")]


def test_search_by_name_finds_the_renamed_copy_by_its_final_name(shared_log):
    (found,) = shared_log.search(name="(1)")
    assert found.destination_name == "Report (1).pdf"
    assert found.client_id == "Lab-PC-02"


def test_search_by_client_ignores_case_but_is_not_a_partial_match(shared_log):
    assert len(shared_log.search(client_id="lab-pc-01")) == 2
    assert len(shared_log.search(client_id="LAB-PC-02")) == 2
    assert shared_log.search(client_id="LAB-PC-0") == []


def test_search_by_status_accepts_text_or_enum(shared_log):
    assert len(shared_log.search(status=FAILED)) == 1
    assert len(shared_log.search(status="completed")) == 3


def test_search_filters_combine(shared_log):
    assert names(shared_log.search(name="report", client_id="lab-pc-01")) == [("LAB-PC-01", "Report.pdf")]
    assert shared_log.search(name="gone", status=COMPLETED) == []


def test_search_with_no_filters_returns_everything_newest_first(shared_log):
    found = shared_log.search()
    assert len(found) == 4
    assert [r.timestamp for r in found] == sorted((r.timestamp for r in found), reverse=True)
    assert shared_log.search(name="   ") == found


def test_search_with_no_match_returns_an_empty_list(shared_log):
    assert shared_log.search(name="nothing like this") == []


def test_the_top_result_says_where_the_file_is_now(shared_log):
    newest_report = shared_log.search(name="Report", client_id="LAB-PC-02")[0]
    assert newest_report.location == str(Path("Documents/Report (1).pdf"))
    assert shared_log.search(status=FAILED)[0].location == str(Path("gone.txt"))
    assert shared_log.search(name="photo")[0].location == str(Path("Pictures/photo.JPG"))


def test_records_with_the_same_timestamp_come_back_last_written_first(tmp_path):
    logger = make_logger(tmp_path)
    for name in ["one.txt", "two.txt", "three.txt"]:
        logger.record(moved(name))
    assert [r.file_name for r in logger.search()] == ["three.txt", "two.txt", "one.txt"]


def test_searching_a_missing_log_returns_an_empty_list(tmp_path):
    assert make_logger(tmp_path).search(name="anything") == []


def test_search_skips_damaged_lines(tmp_path):
    logger = make_logger(tmp_path)
    logger.record(moved("a.txt"))
    with open(logger.log_path, "a", encoding="utf-8") as file:
        file.write("{damaged\n")
    logger.record(moved("b.txt"))
    assert [r.file_name for r in logger.search()] == ["b.txt", "a.txt"]


# ================================================================ with the real operations and scanner

def test_end_to_end_with_real_files(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "thesis.pdf").write_text("12345")
    (downloads / "notes.txt").write_text("abc")

    scanner, operations_, audit = FileScanner(), FileOperations(), make_logger(tmp_path)
    for snapshot in scanner.scan(downloads).files:
        folder = tmp_path / ("Documents" if snapshot.path.suffix == ".pdf" else "Text")
        audit.record(operations_.move_file(snapshot.path, folder), size_bytes=snapshot.size)
    audit.record(operations_.move_file(downloads / "missing.docx", tmp_path / "Documents"))

    (thesis,) = audit.search(name="thesis")
    assert thesis.status == "completed"
    assert thesis.extension == "pdf"
    assert thesis.size_bytes == 5
    assert thesis.location == str(tmp_path / "Documents" / "thesis.pdf")
    assert Path(thesis.location).read_text() == "12345"

    (missing,) = audit.search(status=FAILED)
    assert missing.reason == "source_missing"
    assert missing.size_bytes is None
    assert missing.extension == "docx"
