from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from memcodeagent.memory.schema import MemoryItem, TaskRecord
from memcodeagent.workspace import Workspace

_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "add", "please", "task", "with", "that", "this", "it", "be", "as",
}
_MAX_RECORDS = 200
_MAX_RETRIEVED = 5


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_./\\-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


@dataclass(slots=True)
class RetrievalContext:
    items: list[MemoryItem] = field(default_factory=list)

    def to_prompt(self) -> str:
        if not self.items:
            return "(no retrieved context yet)"
        return "\n\n".join(f"[{item.kind}] {item.text}" for item in self.items)

    def to_display(self) -> str:
        if not self.items:
            return "(no retrieved context yet)"
        lines = []
        for item in self.items:
            location = f" {item.path}" if item.path else ""
            lines.append(f"- {item.kind}{location} score={item.score:.2f} reason={item.reason}\n  {item.text}")
        return "\n".join(lines)


class SimpleRetriever:
    """Lightweight keyword-based memory: persists task history to a local JSON file
    inside the workspace (.memcode/memory.json) and retrieves by simple token
    overlap / filename matching. No embeddings, no vector store, no external service.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._store_path = self.workspace.root / ".memcode" / "memory.json"
        self._changed_files: list[str] = []
        self._failed_commands: list[str] = []

    # -- run-time bookkeeping -------------------------------------------------

    def record_tool_result(self, tool_name: str, ok: bool, args: dict, content: str) -> None:
        """Called by the agent loop after each tool execution to track what changed."""
        if ok and tool_name in {"write_file", "apply_patch"}:
            path = args.get("path")
            if path and path not in self._changed_files:
                self._changed_files.append(path)
        if tool_name == "run_command" and not ok:
            command = args.get("command", "")
            if command:
                self._failed_commands.append(command)
        if tool_name == "run_command" and ok and "exit_code=0" not in content:
            command = args.get("command", "")
            if command:
                self._failed_commands.append(command)

    # -- persistence -----------------------------------------------------------

    def _load_records(self) -> list[TaskRecord]:
        if not self._store_path.exists():
            return []
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [TaskRecord.from_dict(item) for item in raw.get("records", [])]

    def _save_records(self, records: list[TaskRecord]) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [record.to_dict() for record in records[-_MAX_RECORDS:]]}
        self._store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def remember_task(self, task: str, summary: str) -> None:
        records = self._load_records()
        records.append(
            TaskRecord(
                task=task,
                summary=summary,
                changed_files=list(self._changed_files),
                failed_commands=list(self._failed_commands),
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        self._save_records(records)
        self._changed_files = []
        self._failed_commands = []

    # -- retrieval ---------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalContext:
        records = self._load_records()
        if not records:
            return RetrievalContext()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return RetrievalContext()

        scored: list[tuple[float, TaskRecord]] = []
        for record in records:
            record_tokens = _tokenize(record.task) | _tokenize(record.summary)
            record_tokens |= {token for path in record.changed_files for token in _tokenize(path)}
            record_tokens |= {token for cmd in record.failed_commands for token in _tokenize(cmd)}
            overlap = query_tokens & record_tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            scored.append((score, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        items: list[MemoryItem] = []
        for score, record in scored[:_MAX_RETRIEVED]:
            text = f"Previous task: {record.task}\nOutcome: {record.summary}"
            if record.changed_files:
                text += f"\nChanged files: {', '.join(record.changed_files)}"
            if record.failed_commands:
                text += f"\nFailed commands: {', '.join(record.failed_commands)}"
            items.append(
                MemoryItem(
                    kind="task_history",
                    text=text,
                    score=score,
                    reason="keyword overlap with current task",
                )
            )
        return RetrievalContext(items=items)

    def index_workspace(self) -> str:
        file_count = sum(
            1 for path in self.workspace.root.glob("**/*") if path.is_file() and not self.workspace.should_ignore(path)
        )
        return f"Indexed workspace: {file_count} files discovered (keyword index is built lazily from task history)."
