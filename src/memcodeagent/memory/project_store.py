"""Project-level durable guidance."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ProjectMemoryStore:
    """Store explicit repository rules separately from task history."""

    def __init__(self, workspace_root: Path) -> None:
        self.path = workspace_root / ".memcode" / "project_memory.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        payload.setdefault("schema_version", 1)
        fd, temp_name = tempfile.mkstemp(
            prefix="project.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def to_context_text(self) -> str:
        rules = self.load().get("rules", [])
        if not isinstance(rules, list) or not rules:
            return ""
        return "Project memory:\n" + "\n".join(f"- {rule}" for rule in rules)
