"""Tests for stability_checker.py. Run with:  python -m pytest test_stability_checker.py -v"""
import dataclasses
import json
from pathlib import Path

import pytest

from scanner import FileScanner, ScannedFile
from stability_checker import (
    FileAssessment,
    FileStabilityChecker,
    StabilityPolicy,
    StabilityStatus,
)

CHANGING = StabilityStatus.CHANGING
STABLE = StabilityStatus.STABLE
TEMPORARY = StabilityStatus.TEMPORARY


# ---------------------------------------------------------------- helpers

class FakeClock:
    """A clock the test controls. Calling it returns the current fake time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def snap(name: str, size: int = 100, mtime: float = 1.0) -> ScannedFile:
    return ScannedFile(path=Path(name), size=size, modified_time=mtime)


def make_checker(period: float = 5.0, **policy_kwargs):
    clock = FakeClock()
    checker = FileStabilityChecker(StabilityPolicy(period, **policy_kwargs), clock=clock)
    return checker, clock


def statuses(assessments) -> list[StabilityStatus]:
    return [a.status for a in assessments]


# ---------------------------------------------------------------- the core timing rule

def test_first_sighting_is_changing_with_zero_unchanged_time():
    checker, _ = make_checker()
    (result,) = checker.check([snap("a.txt")])
    assert isinstance(result, FileAssessment)
    assert result.status is CHANGING
    assert result.unchanged_for == 0


def test_becomes_stable_exactly_when_the_period_is_reached():
    checker, clock = make_checker(period=5)
    file = snap("a.txt")
    checker.check([file])

    clock.advance(4.9)
    (result,) = checker.check([file])
    assert result.status is CHANGING
    assert result.unchanged_for == pytest.approx(4.9)

    clock.advance(0.1)
    (result,) = checker.check([file])
    assert result.status is STABLE
    assert result.unchanged_for == pytest.approx(5.0)


def test_size_change_restarts_the_timer():
    checker, clock = make_checker(period=5)
    checker.check([snap("a.txt", size=100)])
    clock.advance(4)
    (result,) = checker.check([snap("a.txt", size=150)])
    assert result.status is CHANGING
    assert result.unchanged_for == 0
    clock.advance(4)
    (result,) = checker.check([snap("a.txt", size=150)])
    assert result.status is CHANGING  # only 4 seconds since the change, not 8


def test_modified_time_change_alone_restarts_the_timer():
    checker, clock = make_checker(period=5)
    checker.check([snap("a.txt", mtime=1.0)])
    clock.advance(10)
    (result,) = checker.check([snap("a.txt", mtime=2.0)])
    assert result.status is CHANGING
    assert result.unchanged_for == 0


def test_stable_file_stays_stable_until_it_changes_again():
    checker, clock = make_checker(period=5)
    file = snap("a.txt")
    checker.check([file])
    clock.advance(5)
    assert statuses(checker.check([file])) == [STABLE]
    clock.advance(60)
    assert statuses(checker.check([file])) == [STABLE]

    (result,) = checker.check([snap("a.txt", size=999)])
    assert result.status is CHANGING


def test_download_example_from_the_architecture_notes():
    # Scans every 5 seconds, stability period 10 seconds.
    checker, clock = make_checker(period=10)
    sizes_and_expected = [
        (100, CHANGING),  # t=0   first sighting
        (150, CHANGING),  # t=5   grew
        (200, CHANGING),  # t=10  grew
        (200, CHANGING),  # t=15  unchanged for 5s
        (200, STABLE),    # t=20  unchanged for 10s
    ]
    for index, (size, expected) in enumerate(sizes_and_expected):
        if index:
            clock.advance(5)
        (result,) = checker.check([snap("movie.mp4", size=size)])
        assert result.status is expected, f"scan {index}"


def test_zero_period_means_stable_on_first_sight():
    checker, _ = make_checker(period=0)
    assert statuses(checker.check([snap("a.txt")])) == [STABLE]


# ---------------------------------------------------------------- several files at once

def test_files_are_judged_independently_and_in_input_order():
    checker, clock = make_checker(period=5)
    checker.check([snap("movie.mp4", size=100), snap("report.pdf", size=10)])
    clock.advance(5)
    results = checker.check([snap("movie.mp4", size=200), snap("report.pdf", size=10)])
    assert [a.path.name for a in results] == ["movie.mp4", "report.pdf"]
    assert statuses(results) == [CHANGING, STABLE]


def test_clock_is_read_once_per_check():
    # This clock moves forward by 1 every time it is read. Two checks read it twice in
    # total, so every file must show exactly 1 second unchanged. If check() read the
    # clock once per file, the readings would be spread out and the numbers would differ.
    ticks = iter(range(1000, 2000))
    checker = FileStabilityChecker(StabilityPolicy(5), clock=lambda: float(next(ticks)))
    files = [snap("a.txt"), snap("b.txt"), snap("c.txt")]
    checker.check(files)
    results = checker.check(files)
    assert [a.unchanged_for for a in results] == [1, 1, 1]


def test_empty_input_returns_empty_list():
    checker, _ = make_checker()
    assert checker.check([]) == []


def test_accepts_any_iterable_such_as_a_generator():
    checker, _ = make_checker()
    results = checker.check(snap(name) for name in ["a.txt", "b.txt"])
    assert [a.path.name for a in results] == ["a.txt", "b.txt"]


# ---------------------------------------------------------------- forgetting files

def test_file_missing_from_a_scan_is_forgotten():
    checker, clock = make_checker(period=5)
    file = snap("a.txt")
    checker.check([file])
    clock.advance(10)
    checker.check([])  # the file disappeared (moved, deleted)
    clock.advance(10)
    (result,) = checker.check([file])  # same values come back
    assert result.status is CHANGING
    assert result.unchanged_for == 0


# ---------------------------------------------------------------- temporary downloads

@pytest.mark.parametrize("name", ["movie.mp4.crdownload", "big.iso.part", "x.tmp", "y.download", "Z.CRDOWNLOAD"])
def test_temporary_extensions_are_always_temporary(name):
    checker, clock = make_checker(period=5)
    file = snap(name)
    checker.check([file])
    clock.advance(1000)  # unchanged for ages, still not eligible
    (result,) = checker.check([file])
    assert result.status is TEMPORARY
    assert result.unchanged_for == 0


def test_renamed_download_starts_fresh_as_changing():
    checker, clock = make_checker(period=5)
    checker.check([snap("movie.mp4.crdownload", size=500)])
    clock.advance(100)
    (result,) = checker.check([snap("movie.mp4", size=500)])
    assert result.status is CHANGING  # a new path: the browser just finished it


def test_only_the_last_extension_counts():
    checker, _ = make_checker()
    assert statuses(checker.check([snap("notes.tmp.txt")])) == [CHANGING]


# ---------------------------------------------------------------- StabilityPolicy

def test_policy_defaults():
    policy = StabilityPolicy()
    assert policy.stability_period == 5.0
    assert policy.temporary_suffixes == {".crdownload", ".part", ".tmp", ".download"}


def test_policy_normalizes_suffixes():
    policy = StabilityPolicy(temporary_suffixes=["PART", ".Tmp", " crdownload ", "", "  ", "."])
    assert policy.temporary_suffixes == {".part", ".tmp", ".crdownload"}


def test_policy_accepts_a_single_string_without_splitting_it():
    assert StabilityPolicy(temporary_suffixes=".iso").temporary_suffixes == {".iso"}


def test_custom_suffixes_replace_the_defaults():
    checker, _ = make_checker(temporary_suffixes=[".iso"])
    results = checker.check([snap("disk.iso"), snap("movie.mp4.crdownload")])
    assert statuses(results) == [TEMPORARY, CHANGING]


def test_empty_suffix_list_means_nothing_is_temporary():
    checker, _ = make_checker(temporary_suffixes=[])
    assert statuses(checker.check([snap("movie.mp4.crdownload")])) == [CHANGING]


def test_negative_period_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        StabilityPolicy(stability_period=-1)


# ---------------------------------------------------------------- misc

def test_default_policy_and_clock_work_out_of_the_box():
    checker = FileStabilityChecker()
    assert checker.policy.stability_period == 5.0
    assert statuses(checker.check([snap("a.txt")])) == [CHANGING]


def test_assessments_are_immutable():
    checker, _ = make_checker()
    (result,) = checker.check([snap("a.txt")])
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "status", STABLE)


def test_status_values_are_plain_strings_that_serialize_to_json():
    assert StabilityStatus.STABLE == "stable"
    assert json.dumps({"status": StabilityStatus.TEMPORARY}) == '{"status": "temporary"}'


# ---------------------------------------------------------------- with the real scanner

def test_end_to_end_with_scanner_and_real_files(tmp_path):
    scanner = FileScanner()
    checker, clock = make_checker(period=5)
    download = tmp_path / "movie.mp4"
    download.write_text("part one")
    (tmp_path / "report.pdf").write_text("done")
    (tmp_path / "other.mp4.crdownload").write_text("x")

    first = checker.check(scanner.scan(tmp_path).files)
    assert {a.path.name: a.status for a in first} == {
        "movie.mp4": CHANGING, "other.mp4.crdownload": TEMPORARY, "report.pdf": CHANGING,
    }

    clock.advance(5)
    second = checker.check(scanner.scan(tmp_path).files)
    assert {a.path.name: a.status for a in second} == {
        "movie.mp4": STABLE, "other.mp4.crdownload": TEMPORARY, "report.pdf": STABLE,
    }

    download.write_text("part one and a lot more data")  # the download grows
    clock.advance(1)
    third = checker.check(scanner.scan(tmp_path).files)
    assert {a.path.name: a.status for a in third}["movie.mp4"] is CHANGING
    assert {a.path.name: a.status for a in third}["report.pdf"] is STABLE
