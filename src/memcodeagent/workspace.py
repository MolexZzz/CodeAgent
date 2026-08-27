from __future__ import annotations

from pathlib import Path


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def ensure_exists(self) -> None:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Workspace does not exist: {self.root}")

    def resolve_inside(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def should_ignore(self, path: Path) -> bool:
        parts = set(path.relative_to(self.root).parts)
        return bool(parts & {".git", ".venv", "__pycache__", ".pytest_cache", ".memcode"})

    def ensure_safe_command(self, command: str) -> None:
        lowered = command.lower()
        blocked = [
            "rm -rf",
            "del /s",
            "rmdir /s",
            "format ",
            "git reset --hard",
            "git clean",
            "shutdown",
        ]
        if any(token in lowered for token in blocked):
            raise ValueError(f"Blocked potentially destructive command: {command}")
