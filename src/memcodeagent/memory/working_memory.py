"""Runtime state for the currently active task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkingMemory:
    current_task: str = ""
    phase: str = "IDLE"
    plan_text: str = ""
    history_summary: str = ""
    changed_files: list[str] = field(default_factory=list)
    last_verification_command: str = ""
    verification_kind: str | None = None
    verification_done: bool = False
    verification_passed: bool = False
    last_diff_summary: str = ""
    unresolved_issue: str = ""
    recent_tests: list[str] = field(default_factory=list)

    def reset(self, task: str = "") -> None:
        self.current_task = task
        self.phase = "IDLE"
        self.plan_text = ""
        self.history_summary = ""
        self.changed_files.clear()
        self.last_verification_command = ""
        self.verification_kind = None
        self.verification_done = False
        self.verification_passed = False
        self.last_diff_summary = ""
        self.unresolved_issue = ""
        self.recent_tests.clear()

    def record_test(self, command: str, passed: bool, summary: str = "") -> None:
        result = "passed" if passed else "failed"
        detail = f": {summary}" if summary else ""
        self.last_verification_command = command
        self.verification_done = True
        self.verification_passed = passed
        self.recent_tests.append(f"{command} [{result}]{detail}")
        del self.recent_tests[:-5]
        self.unresolved_issue = "" if passed else (summary or "test failure")

    def persist_state(self) -> dict[str, Any]:
        return {
            "current_task": self.current_task,
            "phase": self.phase,
            "plan_text": self.plan_text,
            "history_summary": self.history_summary,
            "changed_files": list(self.changed_files),
            "last_verification_command": self.last_verification_command,
            "verification_kind": self.verification_kind,
            "verification_done": self.verification_done,
            "verification_passed": self.verification_passed,
            "last_diff_summary": self.last_diff_summary,
            "unresolved_issue": self.unresolved_issue,
            "recent_tests": list(self.recent_tests),
        }

    def restore_state(self, data: dict[str, Any] | None) -> None:
        if not isinstance(data, dict):
            return
        self.current_task = str(data.get("current_task", ""))
        self.phase = str(data.get("phase", "IDLE"))
        self.plan_text = str(data.get("plan_text", ""))
        self.history_summary = str(data.get("history_summary", ""))
        self.changed_files = [str(item) for item in data.get("changed_files", [])]
        self.last_verification_command = str(data.get("last_verification_command", ""))
        kind = data.get("verification_kind")
        self.verification_kind = str(kind) if kind else None
        self.verification_done = bool(data.get("verification_done", False))
        self.verification_passed = bool(data.get("verification_passed", False))
        self.last_diff_summary = str(data.get("last_diff_summary", ""))
        self.unresolved_issue = str(data.get("unresolved_issue", ""))
        self.recent_tests = [str(item) for item in data.get("recent_tests", [])][-5:]

    def to_context_text(self) -> str:
        tests = "\n".join(f"- {item}" for item in self.recent_tests[-3:]) or "- none"
        return (
            "Current working memory:\n"
            f"- Task: {self.current_task or 'none'}\n"
            f"- Phase: {self.phase}\n"
            f"- Plan: {self.plan_text or 'none'}\n"
            f"- Changed files: {', '.join(self.changed_files) or 'none'}\n"
            f"- Recent tests:\n{tests}\n"
            f"- Unresolved issue: {self.unresolved_issue or 'none'}"
        )
