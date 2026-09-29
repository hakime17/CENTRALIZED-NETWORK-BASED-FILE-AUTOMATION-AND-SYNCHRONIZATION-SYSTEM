import pytest
from file_rules import FileRuleManager


@pytest.fixture
def manager():
    return FileRuleManager()


def test_default_categories(manager):
    assert manager.get_category("photo.jpg") == "Pictures"
    assert manager.get_category("report.pdf") == "Documents"
    assert manager.get_category("song.mp3") == "Music"
    assert manager.get_category("movie.mp4") == "Videos"
    assert manager.get_category("backup.zip") == "Archives"


def test_case_insensitivity(manager):
    assert manager.get_category("IMAGE.PNG") == "Pictures"


def test_unsupported_extension(manager):
    assert manager.get_category("script.py") is None


def test_add_rule(manager):
    manager.add_rule("py", "Code")
    assert manager.get_category("script.py") == "Code"


def test_add_rule_validation(manager):
    with pytest.raises(ValueError):
        manager.add_rule("", "Documents")
