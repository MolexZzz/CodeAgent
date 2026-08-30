from __future__ import annotations

from pathlib import Path
import re


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

    def ensure_writable_path(self, path: str, *, allow_existing: bool = True) -> Path:
        """Resolve a write target and reject links/sensitive runtime files."""
        lexical = self.root / path
        if lexical.is_symlink():
            raise ValueError(f"Refusing to modify symbolic link: {path}")
        resolved = self.resolve_inside(path)
        parts = set(resolved.relative_to(self.root).parts)
        if parts & {".git", ".memcode", ".venv", "__pycache__"}:
            raise ValueError(f"Protected runtime path cannot be modified: {path}")
        if resolved.name.lower() in {".env", ".env.local", ".env.production"}:
            raise ValueError(f"Sensitive file cannot be modified by the agent: {path}")
        if not allow_existing and resolved.exists():
            raise ValueError(f"File already exists: {path}")
        return resolved

    def should_ignore(self, path: Path) -> bool:
        parts = set(path.relative_to(self.root).parts)
        return bool(parts & {".git", ".venv", "__pycache__", ".pytest_cache", ".memcode"})

    def ensure_safe_command(self, command: str) -> None:
        lowered = " ".join(command.lower().split())
        blocked = [
            r"(^|[;&|])\s*(rm|del|erase|rmdir|remove-item|rd|format)\b",
            r"\bgit\s+(reset\s+--hard|clean\b)",
            r"\b(shutdown|restart-computer|stop-computer)\b",
        ]
        if any(re.search(pattern, lowered) for pattern in blocked):
            raise ValueError(f"Blocked potentially destructive command: {command}")
