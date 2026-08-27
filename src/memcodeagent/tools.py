from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcodeagent.workspace import Workspace


@dataclass(slots=True)
class ToolObservation:
    tool_name: str
    ok: bool
    content: str

    def to_message(self) -> dict[str, str]:
        status = "ok" if self.ok else "error"
        return {"role": "user", "content": f"Observation from {self.tool_name} ({status}):\n{self.content}"}

    def to_display(self) -> str:
        color = "green" if self.ok else "red"
        return f"[{color}]Observation from {self.tool_name}:[/{color}]\n{self.content}"


class ToolExecutor:
    def __init__(self, workspace: Workspace, dry_run: bool = False) -> None:
        self.workspace = workspace
        self.dry_run = dry_run

    def execute(self, tool_name: str | None, args: dict[str, Any]) -> ToolObservation:
        if not tool_name:
            return ToolObservation("unknown", False, "Missing tool name.")
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return ToolObservation(tool_name, False, f"Unknown tool: {tool_name}")
        try:
            return ToolObservation(tool_name, True, handler(**args))
        except Exception as exc:
            return ToolObservation(tool_name, False, f"{type(exc).__name__}: {exc}")

    def _tool_list_files(self, glob: str = "**/*", limit: int = 200) -> str:
        paths = []
        for path in self.workspace.root.glob(glob):
            if path.is_file() and not self.workspace.should_ignore(path):
                paths.append(path.relative_to(self.workspace.root).as_posix())
            if len(paths) >= limit:
                break
        return "\n".join(paths) or "(no files)"

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        file_path = self.workspace.resolve_inside(path)
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max((start_line or 1) - 1, 0)
        end = end_line if end_line is not None else len(lines)
        selected = lines[start:end]
        return "\n".join(f"{idx + start + 1}: {line}" for idx, line in enumerate(selected))

    def _tool_search_text(self, query: str, glob: str = "**/*", limit: int = 50) -> str:
        matches = []
        lowered = query.lower()
        for path in self.workspace.root.glob(glob):
            if not path.is_file() or self.workspace.should_ignore(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if lowered in line.lower():
                    rel = path.relative_to(self.workspace.root).as_posix()
                    matches.append(f"{rel}:{line_no}: {line}")
                    if len(matches) >= limit:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def _tool_apply_patch(self, path: str, old: str, new: str) -> str:
        if self.dry_run:
            return "Dry run: patch was not applied."
        file_path = self.workspace.resolve_inside(path)
        content = file_path.read_text(encoding="utf-8")
        if old not in content:
            raise ValueError("old text was not found in target file")
        file_path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"Patched {file_path.relative_to(self.workspace.root).as_posix()}"

    def _tool_run_command(self, command: str, timeout_seconds: int = 30) -> str:
        if self.dry_run:
            return f"Dry run: command was not executed: {command}"
        self.workspace.ensure_safe_command(command)
        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        output = f"exit_code={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        return output[-6000:]
