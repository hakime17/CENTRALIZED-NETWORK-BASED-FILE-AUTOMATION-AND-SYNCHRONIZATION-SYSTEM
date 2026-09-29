from pathlib import Path


class FileRuleManager:
    # Class-level default rules (avoids recreating the dictionary on every instantiation)
    DEFAULT_RULES: dict[str, str] = {
        # Pictures
        ".jpg": "Pictures",
        ".jpeg": "Pictures",
        ".png": "Pictures",
        ".gif": "Pictures",
        # Music
        ".mp3": "Music",
        ".wav": "Music",
        ".flac": "Music",
        # Documents
        ".pdf": "Documents",
        ".docx": "Documents",
        ".txt": "Documents",
        ".xlsx": "Documents",
        ".pptx": "Documents",
        # Videos
        ".mp4": "Videos",
        ".mkv": "Videos",
        ".mov": "Videos",
        ".avi": "Videos",
         # Archives
        ".zip": "Archives",
        ".rar": "Archives",
        ".7z": "Archives",
        ".tar": "Archives",
        ".gz": "Archives",
        ".bz2": "Archives",
        ".xz": "Archives",
        ".iso": "Archives",
    }

    def __init__(self, custom_rules: dict[str, str] | None = None) -> None:
        """Initialize rule manager with default rules and optional custom overrides."""
        self.rules: dict[str, str] = self.DEFAULT_RULES.copy()

        if custom_rules:
            for ext, category in custom_rules.items():
                self.add_rule(ext, category)

    def get_category(self, file_path: str | Path) -> str | None:
        """Determine the category a file belongs to.

        Only the last extension is checked, so "backup.tar.gz" is treated
        as ".gz". Multi-part extensions are not supported in V1.

        Returns:
            The destination category if a rule exists.
            None if the extension is unsupported.
        """
        extension = Path(file_path).suffix.lower()
        return self.rules.get(extension)

    def add_rule(self, extension: str, category: str) -> None:
        """Add or update a mapping, ensuring extension format consistency.

        Raises:
            ValueError: if the extension or category is empty.
        """
        extension = extension.strip().lower().lstrip(".")
        category = category.strip()

        if not extension:
            raise ValueError("Extension must not be empty.")
        if not category:
            raise ValueError("Category must not be empty.")

        self.rules[f".{extension}"] = category

    def get_rules(self) -> dict[str, str]:
        """Return a copy of the current rules so callers cannot modify them directly."""
        return self.rules.copy()
