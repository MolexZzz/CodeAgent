from __future__ import annotations

import difflib
import locale
import subprocess
from dataclasses import dataclass
from typing import Any

from memcodeagent.workspace import Workspace

# How much file content to show back to the model when a patch edit fails to
# match, so it can see why and retry with corrected text.
_CONTEXT_SNIPPET_CHARS = 800
_DEFAULT_READ_MAX_LINES = 400
_DEFAULT_READ_MAX_CHARS = 24000
_DEFAULT_SEARCH_MAX_CHARS = 16000


@dataclass(slots=True)
class ToolObservation:
    tool_name: str
    ok: bool
    content: str
    tool_call_id: str | None = None

    def to_message(self) -> dict[str, str]:
        status = "ok" if self.ok else "error"
        text = f"({status}) {self.content}"
        if self.tool_call_id:
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": text}
        # Fallback for callers without a native tool_call_id (e.g. tests).
        return {"role": "user", "content": f"Observation from {self.tool_name} ({status}):\n{self.content}"}

    def to_display(self) -> str:
        color = "green" if self.ok else "red"
        return f"[{color}]Observation from {self.tool_name}:[/{color}]\n{self.content}"


# Tools whose failures are worth retrying automatically without going back to
# the model: run_command can fail transiently (timeouts, flaky installs,
# momentary lock contention). write_file/apply_patch are deterministic given
# the same args, so retrying them without a code change would just repeat the
# same failure -- those are left to the normal LLM-driven retry (it sees the
# error message and adjusts its next call).
_AUTO_RETRY_TOOLS = {"run_command"}


class ToolExecutor:
    def __init__(self, workspace: Workspace, dry_run: bool = False, max_tool_retries: int = 2) -> None:
        self.workspace = workspace
        self.dry_run = dry_run
        self.max_tool_retries = max_tool_retries

    def execute(self, tool_name: str | None, args: dict[str, Any], tool_call_id: str | None = None) -> ToolObservation:
        if not tool_name:
            return ToolObservation("unknown", False, "Missing tool name.", tool_call_id)
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return ToolObservation(tool_name, False, f"Unknown tool: {tool_name}", tool_call_id)

        max_attempts = 1 + (self.max_tool_retries if tool_name in _AUTO_RETRY_TOOLS else 0)
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return ToolObservation(tool_name, True, handler(**args), tool_call_id)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts:
                    continue
        prefix = f"(failed after {max_attempts} attempt(s)) " if max_attempts > 1 else ""
        return ToolObservation(tool_name, False, f"{prefix}{last_error}", tool_call_id)

    def _tool_list_files(self, glob: str = "**/*", limit: int = 200) -> str:
        paths = []
        for path in self.workspace.root.glob(glob):
            if path.is_file() and not self.workspace.should_ignore(path):
                paths.append(path.relative_to(self.workspace.root).as_posix())
            if len(paths) >= limit:
                break
        return "\n".join(paths) or "(no files)"

    def _tool_summarize_tree(self, max_files: int = 120) -> str:
        """Return a compact repository tree summary for planning."""
        entries: list[str] = []
        count = 0
        for path in sorted(self.workspace.root.rglob("*")):
            if count >= max_files:
                entries.append(f"... [truncated after {max_files} files]")
                break
            if not path.is_file() or self.workspace.should_ignore(path):
                continue
            rel = path.relative_to(self.workspace.root).as_posix()
            entries.append(f"{rel} ({path.stat().st_size} bytes)")
            count += 1
        return "\n".join(entries) or "(no files)"

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        file_path = self.workspace.resolve_inside(path)
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max((start_line or 1) - 1, 0)
        requested_end = end_line if end_line is not None else start + _DEFAULT_READ_MAX_LINES
        end = min(requested_end, start + _DEFAULT_READ_MAX_LINES)
        selected = lines[start:end]
        output = "\n".join(f"{idx + start + 1}: {line}" for idx, line in enumerate(selected))
        if len(output) > _DEFAULT_READ_MAX_CHARS:
            output = output[:_DEFAULT_READ_MAX_CHARS] + (
                f"\n... [truncated at {_DEFAULT_READ_MAX_CHARS} chars; "
                "use start_line/end_line to read another section]"
            )
        if end < len(lines):
            output += (
                f"\n... [showing lines {start + 1}-{end} of {len(lines)}; "
                "use start_line/end_line to continue]"
            )
        return output or "(empty file or requested range)"

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
                        output = "\n".join(matches)
                        return output[:_DEFAULT_SEARCH_MAX_CHARS] + (
                            "\n... [search output truncated; narrow the glob or query]"
                        ) if len(output) > _DEFAULT_SEARCH_MAX_CHARS else output
        output = "\n".join(matches)
        if len(output) > _DEFAULT_SEARCH_MAX_CHARS:
            output = output[:_DEFAULT_SEARCH_MAX_CHARS] + (
                "\n... [search output truncated; narrow the glob or query]"
            )
        return output or "(no matches)"

    def _tool_diff_summary(self) -> str:
        """Return a generic git diff summary when the workspace is a git repo."""
        if not (self.workspace.root / ".git").exists():
            return "No git repository found; diff summary is unavailable."
        stat = subprocess.run(
            "git diff --stat",
            cwd=self.workspace.root,
            shell=True,
            capture_output=True,
            timeout=20,
        )
        names = subprocess.run(
            "git diff --name-status",
            cwd=self.workspace.root,
            shell=True,
            capture_output=True,
            timeout=20,
        )
        code = stat.returncode or names.returncode
        stdout = self._decode_output(stat.stdout) + "\n" + self._decode_output(names.stdout)
        stderr = self._decode_output(stat.stderr) + self._decode_output(names.stderr)
        return f"exit_code={code}\nSTDOUT:\n{stdout.strip() or '(no changes)'}\nSTDERR:\n{stderr}"[-6000:]

    def _tool_write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        if self.dry_run:
            return f"Dry run: would write {path} ({len(content)} chars)."
        file_path = self.workspace.resolve_inside(path)
        if file_path.exists() and not overwrite:
            raise ValueError(
                f"{path} already exists. Pass overwrite=true to replace it, or use apply_patch to edit it."
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        action = "Overwrote" if file_path.exists() else "Created"
        return f"{action} {file_path.relative_to(self.workspace.root).as_posix()} ({len(content)} chars)"

    def _tool_apply_patch(self, path: str, edits: list[dict[str, str]]) -> str:
        if self.dry_run:
            return f"Dry run: would apply {len(edits)} edit(s) to {path}."
        if not edits:
            raise ValueError("edits must contain at least one {old, new} entry")

        file_path = self.workspace.resolve_inside(path)
        original = file_path.read_text(encoding="utf-8")
        updated = original
        for index, edit in enumerate(edits):
            old = edit.get("old", "")
            new = edit.get("new", "")
            if old == "":
                raise ValueError(f"edit #{index + 1}: 'old' must not be empty")
            occurrences = updated.count(old)
            if occurrences == 0:
                snippet = updated[:_CONTEXT_SNIPPET_CHARS]
                raise ValueError(
                    f"edit #{index + 1}: text not found in {path}.\n"
                    f"--- searched for ---\n{old}\n"
                    f"--- current file content (first {len(snippet)} chars) ---\n{snippet}"
                )
            if occurrences > 1:
                raise ValueError(
                    f"edit #{index + 1}: text matched {occurrences} times in {path}; "
                    "add more surrounding context so it matches exactly once."
                )
            updated = updated.replace(old, new, 1)

        if updated == original:
            return f"No changes applied to {path} (edits produced identical content)."

        file_path.write_text(updated, encoding="utf-8")
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
        diff_text = "".join(diff)
        rel = file_path.relative_to(self.workspace.root).as_posix()
        return f"Patched {rel} with {len(edits)} edit(s):\n{diff_text}"

    def _tool_run_command(self, command: str, timeout_seconds: int = 30) -> str:
        if self.dry_run:
            return f"Dry run: command was not executed: {command}"
        self.workspace.ensure_safe_command(command)

        # Use the platform default shell. On Windows this is cmd.exe, which
        # supports the `command1 && command2` form commonly emitted by models.
        import platform
        shell_executable = None

        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            shell=True,
            executable=shell_executable,
            capture_output=True,
            timeout=timeout_seconds,
        )
        stdout = self._decode_output(completed.stdout)
        stderr = self._decode_output(completed.stderr)
        output = f"exit_code={completed.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        return output[-6000:]

    @staticmethod
    def _decode_output(raw: bytes) -> str:
        """Decode subprocess bytes without corrupting Windows localized output."""
        if not raw:
            return ""
        import platform
        encodings = ("utf-8", locale.getpreferredencoding(False), "cp936", "cp1252")
        for encoding in encodings:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
