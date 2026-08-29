"""Persistent task memory separate from code retrieval."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from memcodeagent.memory.schema import MemoryItem, TaskRecord

_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "add", "please", "task", "with", "that", "this", "it", "be", "as",
}
_MAX_RECORDS = 200
_MAX_RETRIEVED_TASK = 3


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_./\\-]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in _STOPWORDS]


class TaskMemoryStore:
    """Store task outcomes and retrieve relevant historical tasks."""

    def __init__(self, workspace_root: Path) -> None:
        self.path = workspace_root / ".memcode" / "memory.json"
        self._changed_files: list[str] = []
        self._failed_commands: list[str] = []

    def retrieve(self, query: str) -> list[MemoryItem]:
        records = self._load_records()
        query_tokens = set(_tokenize(query))
        if not records or not query_tokens:
            return []

        scored: list[tuple[float, TaskRecord]] = []
        for record in records:
            record_tokens = set(_tokenize(record.task) + _tokenize(record.summary))
            for path in record.changed_files:
                record_tokens.update(_tokenize(path))
            for command in record.failed_commands:
                record_tokens.update(_tokenize(command))
            for lesson in record.lessons_learned:
                record_tokens.update(_tokenize(lesson))
            overlap = query_tokens & record_tokens
            if overlap:
                scored.append((len(overlap) / len(query_tokens), record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        items: list[MemoryItem] = []
        for score, record in scored[:_MAX_RETRIEVED_TASK]:
            text = f"Previous task: {record.task}\nOutcome: {record.summary}"
            if record.changed_files:
                text += f"\nChanged files: {', '.join(record.changed_files)}"
            if record.failed_commands:
                text += f"\nFailed commands: {', '.join(record.failed_commands)}"
            if record.lessons_learned:
                text += f"\nLessons learned: {', '.join(record.lessons_learned)}"
            items.append(
                MemoryItem(
                    kind="task_history",
                    text=text,
                    score=score,
                    reason="keyword overlap",
                )
            )
        return items

    def record_tool_result(
        self,
        tool_name: str,
        ok: bool,
        args: dict,
        content: str,
    ) -> None:
        if ok and tool_name in {"write_file", "apply_patch"}:
            path = args.get("path")
            if path and path not in self._changed_files:
                self._changed_files.append(path)
        if tool_name == "run_command":
            command = args.get("command", "")
            if command and (not ok or "exit_code=0" not in content):
                self._failed_commands.append(command)

    def remember_task(self, task: str, summary: str) -> None:
        records = self._load_records()
        status = "paused" if "stopped" in summary.lower() or "暂停" in summary else "completed"
        records.append(
            TaskRecord(
                task=task,
                summary=summary,
                task_id=f"task-{uuid.uuid4().hex[:12]}",
                status=status,
                key_changes=list(self._changed_files),
                changed_files=list(self._changed_files),
                failed_commands=list(self._failed_commands),
                unresolved_issues=(
                    list(self._failed_commands) if status == "paused" else []
                ),
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        self._save_records(records)
        self._changed_files.clear()
        self._failed_commands.clear()

    def _load_records(self) -> list[TaskRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [TaskRecord.from_dict(item) for item in raw.get("records", [])]

    def _save_records(self, records: list[TaskRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [record.to_dict() for record in records[-_MAX_RECORDS:]]}
        fd, temp_name = tempfile.mkstemp(
            prefix="memory.", suffix=".tmp", dir=self.path.parent
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
