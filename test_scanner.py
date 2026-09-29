"""Tests for scanner.py. Run with:  python -m pytest test_scanner.py -v"""
import dataclasses
import json
import logging
import os
import types
from pathlib import Path

import pytest

from scanner import FileScanner, ScanResult, ScannedFile, SkippedFile, SkipReason


# ---------------------------------------------------------------- helpers

def make_file(folder: Path, name: str, content: str = "x") -> Path:
    path = folder / name
    path.write_text(content)
    return path


def names(items) -> list[str]:
    return [item.path.name for item in items]


def patch_stat(monkeypatch, behaviour):
    """Replace Path.stat. `behaviour(path, real_stat_result)` may raise or return a stat-like object."""
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        return behaviour(self, real_stat(self, *args, **kwargs))

    monkeypatch.setattr(Path, "stat", fake_stat)


# ---------------------------------------------------------------- basic discovery

def test_returns_scan_result_with_tuples(tmp_path):
    make_file(tmp_path, "a.txt")
    result = FileScanner().scan(tmp_path)
    assert isinstance(result, ScanResult)
    assert isinstance(result.files, tuple)
    assert isinstance(result.skipped, tuple)


def test_empty_folder_gives_empty_result(tmp_path):
    result = FileScanner().scan(tmp_path)
    assert result.files == ()
    assert result.skipped == ()


def test_snapshot_has_path_size_and_modified_time(tmp_path):
    path = make_file(tmp_path, "report.pdf", "hello")  # 5 bytes
    (snapshot,) = FileScanner().scan(tmp_path).files
    assert isinstance(snapshot, ScannedFile)
    assert snapshot.path == path
    assert snapshot.size == 5
    assert snapshot.modified_time == path.stat().st_mtime


def test_results_are_sorted_by_path(tmp_path):
    for name in ["c.txt", "a.txt", "b.txt"]:
        make_file(tmp_path, name)
    assert names(FileScanner().scan(tmp_path).files) == ["a.txt", "b.txt", "c.txt"]


def test_accepts_string_path(tmp_path):
    make_file(tmp_path, "a.txt")
    assert names(FileScanner().scan(str(tmp_path)).files) == ["a.txt"]


def test_files_without_extension_are_reported(tmp_path):
    make_file(tmp_path, "README")
    assert names(FileScanner().scan(tmp_path).files) == ["README"]


def test_subfolders_and_their_contents_are_ignored(tmp_path):
    make_file(tmp_path, "top.txt")
    sub = tmp_path / "sub"
    sub.mkdir()
    make_file(sub, "inner.txt")
    assert names(FileScanner().scan(tmp_path).files) == ["top.txt"]


# ---------------------------------------------------------------- statelessness

def test_scanner_is_stateless(tmp_path):
    make_file(tmp_path, "a.txt")
    scanner = FileScanner()
    first = scanner.scan(tmp_path)
    assert scanner.scan(tmp_path) == first
    make_file(tmp_path, "b.txt")
    assert names(scanner.scan(tmp_path).files) == ["a.txt", "b.txt"]


def test_snapshots_and_results_are_immutable(tmp_path):
    make_file(tmp_path, "a.txt")
    result = FileScanner().scan(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.files[0].size = 999
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.files = ()


# ---------------------------------------------------------------- hidden files

def test_dot_files_are_hidden_by_default(tmp_path):
    make_file(tmp_path, ".hidden")
    make_file(tmp_path, "visible.txt")
    assert names(FileScanner().scan(tmp_path).files) == ["visible.txt"]


def test_include_hidden_reports_dot_files(tmp_path):
    make_file(tmp_path, ".hidden")
    make_file(tmp_path, "visible.txt")
    assert names(FileScanner(include_hidden=True).scan(tmp_path).files) == [".hidden", "visible.txt"]


def _fake_windows_attributes(monkeypatch, hidden_name: str):
    """Make Path.stat report Windows attribute flags: HIDDEN for one name, ARCHIVE for the rest."""
    import stat as stat_module

    def behaviour(path, real):
        attributes = stat_module.FILE_ATTRIBUTE_HIDDEN if path.name == hidden_name else 0x20
        return types.SimpleNamespace(
            st_size=real.st_size, st_mtime=real.st_mtime, st_mode=real.st_mode,
            st_file_attributes=attributes,
        )

    patch_stat(monkeypatch, behaviour)


def test_windows_hidden_attribute_is_skipped_by_default(tmp_path, monkeypatch):
    make_file(tmp_path, "secret.dat")
    make_file(tmp_path, "normal.dat")
    _fake_windows_attributes(monkeypatch, "secret.dat")
    assert names(FileScanner().scan(tmp_path).files) == ["normal.dat"]


def test_windows_hidden_attribute_is_included_when_requested(tmp_path, monkeypatch):
    make_file(tmp_path, "secret.dat")
    make_file(tmp_path, "normal.dat")
    _fake_windows_attributes(monkeypatch, "secret.dat")
    assert names(FileScanner(include_hidden=True).scan(tmp_path).files) == ["normal.dat", "secret.dat"]


def test_missing_windows_attribute_changes_nothing(tmp_path):
    # On Linux/macOS the real stat result has no st_file_attributes at all.
    make_file(tmp_path, "a.txt")
    assert names(FileScanner().scan(tmp_path).files) == ["a.txt"]


# ---------------------------------------------------------------- ignore_names

@pytest.mark.parametrize("junk", ["Thumbs.db", "THUMBS.DB", "desktop.ini", "Desktop.INI"])
def test_default_junk_names_are_ignored_case_insensitively(tmp_path, junk):
    make_file(tmp_path, junk)
    make_file(tmp_path, "keep.txt")
    assert names(FileScanner().scan(tmp_path).files) == ["keep.txt"]


def test_custom_ignore_names_replace_the_defaults(tmp_path):
    make_file(tmp_path, "skip-me.txt")
    make_file(tmp_path, "Thumbs.db")
    result = FileScanner(ignore_names=["Skip-Me.TXT"]).scan(tmp_path)
    assert names(result.files) == ["Thumbs.db"]


def test_empty_ignore_names_ignores_nothing(tmp_path):
    make_file(tmp_path, "Thumbs.db")
    assert names(FileScanner(ignore_names=[]).scan(tmp_path).files) == ["Thumbs.db"]


def test_single_string_ignore_name_is_not_split_into_characters(tmp_path):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "notes.txt")
    make_file(tmp_path, "t")
    result = FileScanner(ignore_names="a.txt").scan(tmp_path)
    assert names(result.files) == ["notes.txt", "t"]


def test_blank_ignore_names_are_dropped():
    scanner = FileScanner(ignore_names=["  ", "", " Foo.TXT "])
    assert scanner.ignore_names == frozenset({"foo.txt"})


@pytest.mark.parametrize("temp", ["movie.mp4.crdownload", "big.iso.part", "data.tmp", "X.CRDOWNLOAD"])
def test_temporary_downloads_are_still_reported(tmp_path, temp):
    # FileStabilityChecker owns the temp-download decision, so the scanner must not hide these.
    make_file(tmp_path, temp)
    assert names(FileScanner().scan(tmp_path).files) == [temp]


# ---------------------------------------------------------------- symlinks

def _symlink(target: Path, link: Path):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")


def test_symlinks_are_skipped_by_default(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    real = make_file(outside, "real.txt")
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    _symlink(real, scan_dir / "link.txt")
    result = FileScanner().scan(scan_dir)
    assert result.files == ()
    assert result.skipped == ()


def test_symlinks_are_followed_when_requested(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    real = make_file(outside, "real.txt", "12345")
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    _symlink(real, scan_dir / "link.txt")
    (snapshot,) = FileScanner(follow_symlinks=True).scan(scan_dir).files
    assert snapshot.path.name == "link.txt"
    assert snapshot.size == 5


@pytest.mark.parametrize("follow", [False, True])
def test_broken_symlinks_are_left_out_without_error(tmp_path, follow):
    _symlink(tmp_path / "missing-target", tmp_path / "broken.txt")
    result = FileScanner(follow_symlinks=follow).scan(tmp_path)
    assert result.files == ()
    assert result.skipped == ()


# ---------------------------------------------------------------- errors on the folder itself

def test_missing_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        FileScanner().scan(tmp_path / "nope")


def test_path_that_is_a_file_raises_not_a_directory(tmp_path):
    file_path = make_file(tmp_path, "a.txt")
    with pytest.raises(NotADirectoryError, match="Not a directory"):
        FileScanner().scan(file_path)


def test_unreadable_directory_raises_permission_error_with_cause(tmp_path, monkeypatch):
    def deny(self):
        raise PermissionError("denied by test")

    monkeypatch.setattr(Path, "iterdir", deny)
    with pytest.raises(PermissionError) as excinfo:
        FileScanner().scan(tmp_path)
    assert str(tmp_path) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, PermissionError)


# ---------------------------------------------------------------- per-file problems

def test_permission_denied_file_is_reported_as_skipped(tmp_path, monkeypatch, caplog):
    make_file(tmp_path, "secret.txt")
    make_file(tmp_path, "ok.txt")

    def behaviour(path, real):
        if path.name == "secret.txt":
            raise PermissionError("no access")
        return real

    patch_stat(monkeypatch, behaviour)
    with caplog.at_level(logging.WARNING, logger="scanner"):
        result = FileScanner().scan(tmp_path)

    assert names(result.files) == ["ok.txt"]
    (skipped,) = result.skipped
    assert isinstance(skipped, SkippedFile)
    assert skipped.path.name == "secret.txt"
    assert skipped.reason is SkipReason.PERMISSION_DENIED
    assert "no access" in skipped.detail
    assert any("secret.txt" in record.getMessage() for record in caplog.records)


def test_other_os_errors_are_reported_as_io_error(tmp_path, monkeypatch, caplog):
    make_file(tmp_path, "bad.txt")
    make_file(tmp_path, "ok.txt")

    def behaviour(path, real):
        if path.name == "bad.txt":
            raise OSError("disk error")
        return real

    patch_stat(monkeypatch, behaviour)
    with caplog.at_level(logging.WARNING, logger="scanner"):
        result = FileScanner().scan(tmp_path)

    assert names(result.files) == ["ok.txt"]
    (skipped,) = result.skipped
    assert skipped.reason is SkipReason.IO_ERROR
    assert "disk error" in skipped.detail
    assert any("bad.txt" in record.getMessage() for record in caplog.records)


def test_file_that_vanishes_mid_scan_is_in_neither_list(tmp_path, monkeypatch, caplog):
    make_file(tmp_path, "gone.txt")
    make_file(tmp_path, "ok.txt")

    def behaviour(path, real):
        if path.name == "gone.txt":
            raise FileNotFoundError("vanished")
        return real

    patch_stat(monkeypatch, behaviour)
    with caplog.at_level(logging.WARNING, logger="scanner"):
        result = FileScanner().scan(tmp_path)

    assert names(result.files) == ["ok.txt"]
    assert result.skipped == ()
    assert caplog.records == []  # normal on a busy folder: no warning


def test_skipped_files_are_sorted_by_path(tmp_path, monkeypatch):
    for name in ["c.txt", "a.txt", "b.txt"]:
        make_file(tmp_path, name)

    def behaviour(path, real):
        if path.suffix == ".txt":  # leave the folder itself readable
            raise PermissionError("no")
        return real

    patch_stat(monkeypatch, behaviour)
    result = FileScanner().scan(tmp_path)
    assert names(result.skipped) == ["a.txt", "b.txt", "c.txt"]


# ---------------------------------------------------------------- SkipReason

def test_skip_reason_values_are_plain_strings_that_serialize_to_json():
    assert SkipReason.PERMISSION_DENIED == "permission_denied"
    assert SkipReason.IO_ERROR == "io_error"
    assert json.dumps({"reason": SkipReason.IO_ERROR}) == '{"reason": "io_error"}'
    assert not hasattr(SkipReason, "UNREADABLE")
