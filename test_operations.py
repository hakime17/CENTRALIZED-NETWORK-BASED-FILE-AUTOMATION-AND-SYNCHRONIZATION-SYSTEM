"""Tests for operations.py. Run with:  python -m pytest test_operations.py -v"""
import dataclasses
import json
import os
import shutil
from pathlib import Path

import pytest

import operations
from operations import (
    FailureReason,
    FileOperations,
    OperationResult,
    OperationStatus,
    OperationType,
)

COMPLETED = OperationStatus.COMPLETED
DRY_RUN = OperationStatus.DRY_RUN
SKIPPED = OperationStatus.SKIPPED
FAILED = OperationStatus.FAILED


# ---------------------------------------------------------------- helpers

def make_file(folder: Path, name: str, content: str = "x") -> Path:
    path = folder / name
    path.write_text(content)
    return path


def symlink_or_skip(target: Path, link: Path, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")


def deny(*args, **kwargs):
    raise PermissionError("in use by another program")


@pytest.fixture
def ops() -> FileOperations:
    return FileOperations()


# ---------------------------------------------------------------- move

def test_move_relocates_file_and_creates_the_folder(tmp_path, ops):
    source = make_file(tmp_path, "a.txt", "hello")
    result = ops.move_file(source, tmp_path / "Documents")

    expected = tmp_path / "Documents" / "a.txt"
    assert result == OperationResult(OperationType.MOVE, source, expected, COMPLETED)
    assert not source.exists()
    assert expected.read_text() == "hello"


def test_move_creates_nested_folders(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    result = ops.move_file(source, tmp_path / "one" / "two" / "three")
    assert result.status is COMPLETED
    assert (tmp_path / "one" / "two" / "three" / "a.txt").exists()


def test_move_keeps_the_modified_time(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    os.utime(source, (1_000_000, 1_000_000))
    result = ops.move_file(source, tmp_path / "Out")
    assert result.destination.stat().st_mtime == pytest.approx(1_000_000, abs=1)


def test_move_accepts_string_paths(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    result = ops.move_file(str(source), str(tmp_path / "Out"))
    assert result.status is COMPLETED
    assert isinstance(result.source, Path) and isinstance(result.destination, Path)


def test_move_into_an_existing_folder(tmp_path, ops):
    (tmp_path / "Out").mkdir()
    make_file(tmp_path / "Out", "other.txt")
    source = make_file(tmp_path, "a.txt")
    assert ops.move_file(source, tmp_path / "Out").status is COMPLETED


# ---------------------------------------------------------------- copy

def test_copy_keeps_the_original(tmp_path, ops):
    source = make_file(tmp_path, "b.txt", "copyme")
    result = ops.copy_file(source, tmp_path / "Copies")

    expected = tmp_path / "Copies" / "b.txt"
    assert result == OperationResult(OperationType.COPY, source, expected, COMPLETED)
    assert source.read_text() == "copyme"
    assert expected.read_text() == "copyme"


def test_copy_keeps_the_modified_time(tmp_path, ops):
    source = make_file(tmp_path, "b.txt")
    os.utime(source, (2_000_000, 2_000_000))
    result = ops.copy_file(source, tmp_path / "Copies")
    assert result.destination.stat().st_mtime == pytest.approx(2_000_000, abs=1)


def test_copy_into_its_own_folder_makes_a_numbered_copy(tmp_path, ops):
    source = make_file(tmp_path, "same.txt", "s")
    result = ops.copy_file(source, tmp_path)
    assert result.status is COMPLETED
    assert result.destination == tmp_path / "same (1).txt"
    assert source.read_text() == "s"


# ---------------------------------------------------------------- name clashes

def test_name_clash_never_overwrites_and_counts_up(tmp_path, ops):
    folder = tmp_path / "Documents"
    folder.mkdir()
    existing = make_file(folder, "a.txt", "original")

    for expected_name in ["a (1).txt", "a (2).txt", "a (3).txt"]:
        source = make_file(tmp_path, "a.txt", f"new {expected_name}")
        result = ops.move_file(source, folder)
        assert result.status is COMPLETED
        assert result.destination.name == expected_name
        assert result.destination.read_text() == f"new {expected_name}"

    assert existing.read_text() == "original"


def test_copy_name_clash_never_overwrites(tmp_path, ops):
    folder = tmp_path / "Copies"
    folder.mkdir()
    existing = make_file(folder, "a.txt", "original")
    source = make_file(tmp_path, "a.txt", "newer")
    result = ops.copy_file(source, folder)
    assert result.destination.name == "a (1).txt"
    assert existing.read_text() == "original"


@pytest.mark.parametrize("name, expected", [("README", "README (1)"), (".env", ".env (1)"), ("a.tar.gz", "a.tar (1).gz")])
def test_numbering_for_unusual_names(tmp_path, ops, name, expected):
    folder = tmp_path / "Odd"
    folder.mkdir()
    make_file(folder, name)
    source = make_file(tmp_path, name, "new")
    assert ops.move_file(source, folder).destination.name == expected


def test_a_folder_with_the_same_name_counts_as_taken(tmp_path, ops):
    folder = tmp_path / "Out"
    (folder / "a.txt").mkdir(parents=True)
    source = make_file(tmp_path, "a.txt")
    result = ops.move_file(source, folder)
    assert result.status is COMPLETED
    assert result.destination.name == "a (1).txt"


def test_a_broken_symlink_counts_as_taken(tmp_path, ops):
    folder = tmp_path / "Links"
    folder.mkdir()
    symlink_or_skip(tmp_path / "nowhere", folder / "q.txt")
    source = make_file(tmp_path, "q.txt")
    assert ops.move_file(source, folder).destination.name == "q (1).txt"


def test_the_last_allowed_attempt_is_still_used(tmp_path):
    ops = FileOperations()
    ops.MAX_NAME_ATTEMPTS = 2
    folder = tmp_path / "Almost"
    folder.mkdir()
    for name in ["z.txt", "z (1).txt"]:
        make_file(folder, name)
    source = make_file(tmp_path, "z.txt")

    result = ops.move_file(source, folder)
    assert result.status is COMPLETED
    assert result.destination.name == "z (2).txt"


def test_a_broken_symlink_blocks_a_numbered_name_too(tmp_path, ops):
    folder = tmp_path / "Links"
    folder.mkdir()
    make_file(folder, "q.txt")
    symlink_or_skip(tmp_path / "nowhere", folder / "q (1).txt")
    source = make_file(tmp_path, "q.txt")
    assert ops.move_file(source, folder).destination.name == "q (2).txt"


def test_gives_up_when_no_free_name_is_left(tmp_path):
    ops = FileOperations()
    ops.MAX_NAME_ATTEMPTS = 2
    folder = tmp_path / "Full"
    folder.mkdir()
    for name in ["z.txt", "z (1).txt", "z (2).txt"]:
        make_file(folder, name)
    source = make_file(tmp_path, "z.txt", "zz")

    result = ops.move_file(source, folder)
    assert result.status is FAILED
    assert result.reason is FailureReason.NO_FREE_NAME
    assert result.destination is None
    assert source.exists()


# ---------------------------------------------------------------- dry run

def test_dry_run_changes_nothing_and_reports_the_destination(tmp_path):
    dry = FileOperations(dry_run=True)
    source = make_file(tmp_path, "a.txt", "keep")

    result = dry.move_file(source, tmp_path / "Documents")
    assert result.status is DRY_RUN
    assert result.destination == tmp_path / "Documents" / "a.txt"
    assert source.read_text() == "keep"
    assert not (tmp_path / "Documents").exists()  # not even the folder


def test_dry_run_copy_creates_nothing(tmp_path):
    source = make_file(tmp_path, "a.txt")
    result = FileOperations(dry_run=True).copy_file(source, tmp_path / "Copies")
    assert result.status is DRY_RUN
    assert not (tmp_path / "Copies").exists()


def test_dry_run_reports_the_numbered_name_it_would_use(tmp_path):
    folder = tmp_path / "Documents"
    folder.mkdir()
    make_file(folder, "a.txt")
    source = make_file(tmp_path, "a.txt")
    result = FileOperations(dry_run=True).move_file(source, folder)
    assert result.status is DRY_RUN
    assert result.destination == folder / "a (1).txt"
    assert [p.name for p in folder.iterdir()] == ["a.txt"]


def test_dry_run_still_reports_real_problems(tmp_path):
    result = FileOperations(dry_run=True).move_file(tmp_path / "missing.txt", tmp_path / "Out")
    assert result.status is FAILED
    assert result.reason is FailureReason.SOURCE_MISSING


# ---------------------------------------------------------------- moving into the same folder

def test_move_into_its_own_folder_is_skipped(tmp_path, ops):
    source = make_file(tmp_path, "a.txt", "stay")
    result = ops.move_file(source, tmp_path)
    assert result.status is SKIPPED
    assert result.destination == source
    assert "already" in result.detail
    assert source.read_text() == "stay"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]  # no "a (1).txt" created


def test_same_folder_is_recognised_whatever_the_spelling(tmp_path, ops):
    (tmp_path / "sub").mkdir()
    source = make_file(tmp_path, "a.txt")
    spellings = [tmp_path / ".", tmp_path / "sub" / "..", str(tmp_path) + os.sep]
    for spelling in spellings:
        result = ops.move_file(source, spelling)
        assert result.status is SKIPPED, spelling
    assert source.exists()


def test_same_folder_relative_to_the_working_directory(tmp_path, ops, monkeypatch):
    source = make_file(tmp_path, "a.txt")
    monkeypatch.chdir(tmp_path)
    assert ops.move_file(source, ".").status is SKIPPED


def test_same_folder_through_a_folder_symlink(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    symlink_or_skip(tmp_path, tmp_path / "dirlink", directory=True)
    assert ops.move_file(source, tmp_path / "dirlink").status is SKIPPED


def test_a_folder_that_does_not_exist_yet_is_never_the_same_folder(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    assert ops.move_file(source, tmp_path / "NotYetCreated").status is COMPLETED


# ---------------------------------------------------------------- problems come back as results

def test_missing_source_fails_without_raising(tmp_path, ops):
    result = ops.move_file(tmp_path / "missing.txt", tmp_path / "Out")
    assert result.status is FAILED
    assert result.reason is FailureReason.SOURCE_MISSING
    assert result.destination is None
    assert not (tmp_path / "Out").exists()


def test_a_folder_is_not_a_valid_source(tmp_path, ops):
    folder = tmp_path / "afolder"
    folder.mkdir()
    for method in (ops.move_file, ops.copy_file):
        result = method(folder, tmp_path / "Out")
        assert result.status is FAILED
        assert result.reason is FailureReason.SOURCE_MISSING


def test_symlink_sources_are_refused_by_move_and_copy(tmp_path, ops):
    real = make_file(tmp_path, "real.txt", "R")
    symlink_or_skip(real, tmp_path / "link.txt")
    for method in (ops.move_file, ops.copy_file):
        result = method(tmp_path / "link.txt", tmp_path / "Out")
        assert result.status is FAILED
        assert result.reason is FailureReason.UNSUPPORTED_SOURCE
    assert (tmp_path / "link.txt").is_symlink()  # untouched
    assert not (tmp_path / "Out").exists()


def test_destination_that_is_a_file_fails_and_leaves_the_source(tmp_path, ops):
    blocker = make_file(tmp_path, "iamafile")
    source = make_file(tmp_path, "c.txt")
    result = ops.move_file(source, blocker)
    assert result.status is FAILED
    assert result.reason is FailureReason.DESTINATION_INVALID
    assert "not a folder" in result.detail
    assert source.exists()


def test_destination_path_running_through_a_file_fails(tmp_path, ops):
    blocker = make_file(tmp_path, "iamafile")
    source = make_file(tmp_path, "c.txt")
    result = ops.move_file(source, blocker / "sub")
    assert result.status is FAILED
    assert result.reason in (FailureReason.DESTINATION_INVALID, FailureReason.IO_ERROR)
    assert source.exists()


def test_invalid_path_characters_do_not_raise(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    result = ops.move_file(source, str(tmp_path / "bad\0folder"))
    assert result.status is FAILED
    assert result.reason is FailureReason.DESTINATION_INVALID
    assert source.exists()


@pytest.mark.parametrize(
    "error, expected_reason",
    [
        (PermissionError("denied"), FailureReason.PERMISSION_DENIED),
        (NotADirectoryError("not a dir"), FailureReason.DESTINATION_INVALID),
        (ValueError("bad path"), FailureReason.DESTINATION_INVALID),
        (OSError("disk full"), FailureReason.IO_ERROR),
        (shutil.Error("shutil problem"), FailureReason.IO_ERROR),
    ],
)
def test_errors_during_the_copy_are_mapped_to_reasons(tmp_path, ops, monkeypatch, error, expected_reason):
    source = make_file(tmp_path, "a.txt")

    def fail(src, dst, **kwargs):
        raise error

    monkeypatch.setattr(operations.shutil, "copy2", fail)
    result = ops.copy_file(source, tmp_path / "Out")
    assert result.status is FAILED
    assert result.reason is expected_reason
    assert str(error) in result.detail
    assert source.exists()


def test_source_vanishing_during_the_transfer_is_source_missing(tmp_path, ops, monkeypatch):
    source = make_file(tmp_path, "a.txt")

    def vanish(src, dst, **kwargs):
        Path(src).unlink()
        raise FileNotFoundError("gone")

    monkeypatch.setattr(operations.shutil, "copy2", vanish)
    result = ops.copy_file(source, tmp_path / "Out")
    assert result.status is FAILED
    assert result.reason is FailureReason.SOURCE_MISSING


def test_unreachable_destination_with_source_present_is_destination_invalid(tmp_path, ops, monkeypatch):
    source = make_file(tmp_path, "a.txt")
    monkeypatch.setattr(operations.shutil, "copy2", lambda src, dst, **k: (_ for _ in ()).throw(FileNotFoundError("no drive")))
    result = ops.copy_file(source, tmp_path / "Out")
    assert result.reason is FailureReason.DESTINATION_INVALID
    assert source.exists()


# ---------------------------------------------------------------- cleanup after a failure

def test_copy_that_fails_halfway_leaves_no_partial_file(tmp_path, ops, monkeypatch):
    source = make_file(tmp_path, "big.bin", "B" * 50)

    def half_copy(src, dst, **kwargs):
        Path(dst).write_bytes(b"B")
        raise OSError("disk full")

    monkeypatch.setattr(operations.shutil, "copy2", half_copy)
    result = ops.copy_file(source, tmp_path / "Half")
    assert result.status is FAILED
    assert not (tmp_path / "Half" / "big.bin").exists()
    assert source.read_text() == "B" * 50


def test_locked_original_leaves_no_stray_duplicate(tmp_path, ops, monkeypatch):
    # A move that falls back to copy-then-delete, where the delete is refused
    # because another program holds the original open.
    source = make_file(tmp_path, "locked.txt", "L")
    real_unlink = os.unlink

    def guarded_unlink(path, *args, **kwargs):
        if Path(path) == source:
            raise PermissionError("in use by another program")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", deny)
    monkeypatch.setattr(os, "unlink", guarded_unlink)

    result = ops.move_file(source, tmp_path / "Locked")

    assert result.status is FAILED
    assert result.reason is FailureReason.PERMISSION_DENIED
    assert source.read_text() == "L"  # the original is safe
    assert not (tmp_path / "Locked" / "locked.txt").exists()  # and no duplicate left behind


def test_cleanup_never_deletes_the_only_copy_of_a_moved_file(tmp_path, monkeypatch):
    # If the original is already gone when a move reports an error, the file at the
    # destination is the only copy left and must not be deleted.
    source = make_file(tmp_path, "a.txt", "only copy")
    destination = tmp_path / "Out" / "a.txt"
    destination.parent.mkdir()
    destination.write_text("only copy")
    source.unlink()

    FileOperations._remove_leftover(OperationType.MOVE, source, destination)
    assert destination.read_text() == "only copy"


def test_cleanup_removes_a_leftover_while_the_original_still_exists(tmp_path):
    source = make_file(tmp_path, "a.txt", "orig")
    leftover = make_file(tmp_path, "a (1).txt", "orig")
    FileOperations._remove_leftover(OperationType.MOVE, source, leftover)
    assert not leftover.exists()
    assert source.exists()


# ---------------------------------------------------------------- result objects

def test_successful_results_carry_no_failure_reason(tmp_path, ops):
    source = make_file(tmp_path, "a.txt")
    assert ops.move_file(source, tmp_path / "Out").reason is None


def test_results_are_immutable(tmp_path, ops):
    result = ops.move_file(make_file(tmp_path, "a.txt"), tmp_path / "Out")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "status", FAILED)


def test_enum_values_are_plain_strings_that_serialize_to_json():
    assert OperationStatus.DRY_RUN == "dry_run"
    assert OperationType.MOVE == "move"
    assert FailureReason.NO_FREE_NAME == "no_free_name"
    payload = {"op": OperationType.COPY, "status": OperationStatus.FAILED, "reason": FailureReason.IO_ERROR}
    assert json.dumps(payload) == '{"op": "copy", "status": "failed", "reason": "io_error"}'


def test_a_failure_in_one_file_does_not_stop_the_next(tmp_path, ops):
    good = make_file(tmp_path, "good.txt")
    results = [ops.move_file(path, tmp_path / "Out") for path in (tmp_path / "missing.txt", good)]
    assert [r.status for r in results] == [FAILED, COMPLETED]
