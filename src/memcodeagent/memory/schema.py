from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MemoryItem:
    """A single piece of retrieved context shown to the model before a task starts."""

    kind: str
    text: str
    path: Path | None = None
    score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class TaskRecord:
    """A persisted summary of one completed agent run, used for future keyword retrieval."""

    task: str
    summary: str
    changed_files: list[str] = field(default_factory=list)
    failed_commands: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "summary": self.summary,
            "changed_files": self.changed_files,
            "failed_commands": self.failed_commands,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TaskRecord":
        return TaskRecord(
            task=data.get("task", ""),
            summary=data.get("summary", ""),
            changed_files=list(data.get("changed_files", [])),
            failed_commands=list(data.get("failed_commands", [])),
            timestamp=data.get("timestamp", ""),
        )
