"""Unified memory facade for the coding agent.

The first migration step intentionally delegates to the existing components.
This keeps retrieval behavior stable while giving CodingAgent one memory boundary.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from memcodeagent.context_manager import ContextManager
from memcodeagent.memory.hybrid_retriever import HybridRetriever, RetrievalContext
from memcodeagent.memory.task_store import TaskMemoryStore
from memcodeagent.memory.working_memory import WorkingMemory
from memcodeagent.memory.project_store import ProjectMemoryStore
from memcodeagent.workspace import Workspace


class TranscriptStore:
    """Persist a resumable snapshot and an append-only transcript."""

    def __init__(self, workspace: Workspace) -> None:
        self.path = workspace.root / ".memcode" / "session.json"
        self.transcript_path = workspace.root / ".memcode" / "transcript.jsonl"

    def save(self, payload: dict[str, Any]) -> None:
        messages = payload.get("messages", [])
        if isinstance(messages, list):
            previous_count = self._message_count_from_snapshot()
            if len(messages) < previous_count:
                previous_count = 0
            new_messages = messages[previous_count:]
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            if new_messages:
                with self.transcript_path.open("a", encoding="utf-8") as handle:
                    for index, message in enumerate(
                        new_messages, start=previous_count
                    ):
                        json.dump(
                            {"index": index, "message": message},
                            handle,
                            ensure_ascii=False,
                        )
                        handle.write("\n")
            payload = dict(payload)
            payload["transcript_message_count"] = len(messages)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="session.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def load(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    def _message_count_from_snapshot(self) -> int:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0
        count = payload.get("transcript_message_count", 0)
        return int(count) if isinstance(count, int) and count >= 0 else 0


class MemoryManager:
    """Coordinate working context, task memory, code retrieval, and sessions."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_context_turns: int = 20,
        max_context_tokens: int = 24000,
        llm_client: Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.task_memory = TaskMemoryStore(workspace.root)
        self.project_memory = ProjectMemoryStore(workspace.root)
        self.code_retriever = HybridRetriever(
            workspace,
            task_memory_store=self.task_memory,
        )
        self.working_memory = WorkingMemory()
        self.context_manager = ContextManager(
            max_turns=max_context_turns,
            max_tokens=max_context_tokens,
            enable_summarization=True,
            llm_client=llm_client,
            working_memory=self.working_memory,
        )
        self.transcript = TranscriptStore(workspace)

        # Compatibility aliases for existing callers during migration.
        self.retriever = self.code_retriever

    def retrieve(self, query: str) -> RetrievalContext:
        return RetrievalContext(
            items=(
                self.code_retriever.retrieve_code(query)
                + self.task_memory.retrieve(query)
            )
        )

    def record_tool_result(
        self,
        tool_name: str,
        ok: bool,
        args: dict[str, Any],
        content: str,
    ) -> None:
        self.task_memory.record_tool_result(tool_name, ok, args, content)
        if tool_name in {"write_file", "apply_patch"} and ok:
            path = args.get("path")
            if path and path not in self.working_memory.changed_files:
                self.working_memory.changed_files.append(str(path))

    def remember_task(self, task: str, summary: str) -> None:
        self.task_memory.remember_task(task, summary)

    def trim_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        project_text = self.project_memory.to_context_text()
        if project_text:
            messages = [{"role": "system", "content": project_text}, *messages]
        return self.context_manager.trim(messages)

    def summarize_recent(self, messages: list[dict[str, Any]], max_turns: int = 8) -> str:
        return self.context_manager.summarize_recent(messages, max_turns=max_turns)

    def reset_working_memory(self) -> None:
        self.working_memory.reset()

    def record_test_result(self, command: str, passed: bool, summary: str = "") -> None:
        self.working_memory.record_test(command, passed, summary)

    @property
    def last_trim_notice(self) -> str | None:
        return self.context_manager.last_trim_notice

    def save_session(self, payload: dict[str, Any]) -> None:
        """Persist a session atomically so interrupted writes do not corrupt it."""
        payload = dict(payload)
        payload["context_manager"] = self.context_manager.persist_state()
        payload["working_memory"] = self.working_memory.persist_state()
        self.transcript.save(payload)

    def load_session(self) -> dict[str, Any] | None:
        payload = self.transcript.load()
        if payload is None:
            return None
        self.context_manager.restore_state(payload.get("context_manager"))
        self.working_memory.restore_state(payload.get("working_memory"))
        return payload
