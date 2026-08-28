"""Progress and loop-safety monitoring for the single-agent runtime."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Small, serializable view of useful work observed so far."""

    files_discovered: int = 0
    symbols_discovered: int = 0
    files_modified: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0


@dataclass(frozen=True, slots=True)
class ProgressAlert:
    kind: str
    message: str
    severity: str = "warning"


class ProgressMonitor:
    """Detect repeated actions and stagnant exploration without judging code."""

    def __init__(
        self,
        *,
        duplicate_limit: int = 1,
        no_progress_limit: int = 3,
        tool_window: int = 8,
        monotony_limit: int = 6,
    ) -> None:
        self.duplicate_limit = max(1, duplicate_limit)
        self.no_progress_limit = max(1, no_progress_limit)
        self.tool_window = max(2, tool_window)
        self.monotony_limit = max(2, monotony_limit)
        self._calls: Counter[tuple[str, str]] = Counter()
        self._recent_tools: deque[str] = deque(maxlen=self.tool_window)
        self._last_snapshot = ProgressSnapshot()
        self._no_progress_steps = 0
        self._snapshot = ProgressSnapshot()

    @staticmethod
    def call_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
        return name, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)

    def record_tool(self, name: str, args: dict[str, Any]) -> ProgressAlert | None:
        key = self.call_key(name, args)
        self._calls[key] += 1
        self._recent_tools.append(name)
        if self._calls[key] > self.duplicate_limit:
            return ProgressAlert(
                "duplicate_tool",
                f"检测到重复调用 {name}，参数完全相同；已暂停本次重复操作。",
            )
        if (
            len(self._recent_tools) >= self.monotony_limit
            and len(set(self._recent_tools)) == 1
        ):
            return ProgressAlert(
                "tool_monotony",
                f"最近 {len(self._recent_tools)} 次都在调用 {name}，暂时没有看到策略变化。",
            )
        return None

    def record_snapshot(self, snapshot: ProgressSnapshot) -> ProgressAlert | None:
        if snapshot == self._last_snapshot:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        self._last_snapshot = snapshot
        if self._no_progress_steps >= self.no_progress_limit:
            return ProgressAlert(
                "no_progress",
                f"连续 {self._no_progress_steps} 个步骤没有产生新的文件、修改或测试进展。",
            )
        return None

    def record_observation(
        self,
        *,
        tool_name: str,
        ok: bool,
        content: str = "",
    ) -> ProgressAlert | None:
        """Update coarse progress counters from a tool result."""
        if not ok:
            return self.record_snapshot(self._snapshot)
        current = self._snapshot
        if tool_name in {"list_files", "read_file", "search_text", "summarize_tree", "summarize_symbols"}:
            current = ProgressSnapshot(
                files_discovered=current.files_discovered + 1,
                symbols_discovered=current.symbols_discovered,
                files_modified=current.files_modified,
                tests_run=current.tests_run,
                tests_passed=current.tests_passed,
                tests_failed=current.tests_failed,
            )
        elif tool_name in {"write_file", "apply_patch"}:
            current = ProgressSnapshot(
                files_discovered=current.files_discovered,
                symbols_discovered=current.symbols_discovered,
                files_modified=current.files_modified + 1,
                tests_run=current.tests_run,
                tests_passed=current.tests_passed,
                tests_failed=current.tests_failed,
            )
        elif tool_name == "run_command":
            passed = "exit_code=0" in content
            current = ProgressSnapshot(
                files_discovered=current.files_discovered,
                symbols_discovered=current.symbols_discovered,
                files_modified=current.files_modified,
                tests_run=current.tests_run + 1,
                tests_passed=current.tests_passed + int(passed),
                tests_failed=current.tests_failed + int(not passed),
            )
        self._snapshot = current
        return self.record_snapshot(current)

    def reset(self) -> None:
        self._calls.clear()
        self._recent_tools.clear()
        self._last_snapshot = ProgressSnapshot()
        self._no_progress_steps = 0
        self._snapshot = ProgressSnapshot()

    def persist_state(self) -> dict[str, Any]:
        return {
            "calls": [
                {"name": name, "args": args, "count": count}
                for (name, args), count in self._calls.items()
            ],
            "recent_tools": list(self._recent_tools),
            "last_snapshot": asdict(self._last_snapshot),
            "snapshot": asdict(self._snapshot),
            "no_progress_steps": self._no_progress_steps,
        }
