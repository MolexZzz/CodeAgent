from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

# JSON-Schema tool definitions passed to the model's native tool-calling API.
# The model decides which tool to call (zero, one, or several in parallel);
# execution, argument validation, and result formatting stay entirely local
# (see tools.py) -- nothing here is delegated to a hosted code/file tool.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files inside the workspace matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "Glob pattern, defaults to **/*."},
                    "limit": {"type": "integer", "description": "Maximum number of paths to return."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the workspace, optionally a line range, with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "start_line": {"type": "integer", "description": "1-based inclusive start line."},
                    "end_line": {"type": "integer", "description": "1-based inclusive end line."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search for a literal substring across files in the workspace (case-insensitive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                    "glob": {"type": "string", "description": "Glob pattern to restrict the search."},
                    "limit": {"type": "integer", "description": "Maximum number of matching lines."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file or overwrite an existing one inside the workspace. "
                "Use apply_patch instead when editing part of an existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file content to write."},
                    "overwrite": {
                        "type": "boolean",
                        "description": "Set true to replace an existing file. Defaults to false.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply one or more exact string replacements to an existing file. "
                "Each edit's `old` text must match exactly once in the file. "
                "Returns a unified diff of the change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "edits": {
                        "type": "array",
                        "description": "List of {old, new} replacements applied in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                            },
                            "required": ["old", "new"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the workspace and return exit code, stdout, and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_seconds": {"type": "integer", "description": "Timeout in seconds, defaults to 30."},
                },
                "required": ["command"],
            },
        },
    },
]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class TokenUsage:
    """Token usage statistics from an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class AgentDecision:
    """One model turn: either a final natural-language answer or one/more tool calls."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls

    def to_display(self) -> str:
        if self.is_final:
            return f"[green]Final answer:[/green] {self.content}"
        parts = [f"[yellow]Tool call:[/yellow] {tc.name} [yellow]Args:[/yellow] {tc.args}" for tc in self.tool_calls]
        return "\n".join(parts)


class LlmClient:
    """Thin wrapper around an OpenAI-compatible chat model using its native tool-calling API."""

    def __init__(self) -> None:
        # Load .env from current directory or home directory
        load_dotenv()  # Load from current directory
        load_dotenv(Path.home() / ".memcode" / ".env")  # Load from ~/.memcode/.env

        self.model = os.getenv("MEMCODE_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL") or None

    def next_action(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> AgentDecision:
        if not self.api_key:
            return AgentDecision(
                content=(
                    "LLM is not configured yet. Set OPENAI_API_KEY, then rerun the task. "
                    "The CLI framework, tool executor, and memory layer are ready."
                ),
                assistant_message={
                    "role": "assistant",
                    "content": "LLM is not configured yet. Set OPENAI_API_KEY, then rerun the task.",
                },
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        if stream:
            return self._next_action_streaming(client, messages, on_chunk)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message
        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
        return self._parse_decision(message, token_usage)

    def _next_action_streaming(
        self,
        client: OpenAI,
        messages: list[dict[str, Any]],
        on_chunk: Callable[[str], None] | None,
    ) -> AgentDecision:
        """Stream a chat completion, progressively reporting content via `on_chunk`.

        Tool calls arrive as incremental argument fragments across chunks; we
        accumulate them by index and rebuild the final message shape once the
        stream ends. Token usage is only available on the final chunk when
        `stream_options={"include_usage": True}` is requested.
        """
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            stream=True,
            stream_options={"include_usage": True},
        )

        content_parts: list[str] = []
        # tool call fragments keyed by index, since deltas can arrive split across chunks
        tool_call_fragments: dict[int, dict[str, Any]] = {}
        usage_data: Any = None

        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_data = chunk.usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and delta.content:
                content_parts.append(delta.content)
                if on_chunk:
                    on_chunk(delta.content)

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    frag = tool_call_fragments.setdefault(
                        idx, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc_delta.id:
                        frag["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            frag["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            frag["arguments"] += tc_delta.function.arguments

        token_usage = TokenUsage(
            prompt_tokens=usage_data.prompt_tokens if usage_data else 0,
            completion_tokens=usage_data.completion_tokens if usage_data else 0,
            total_tokens=usage_data.total_tokens if usage_data else 0,
        )

        full_content = "".join(content_parts) or None

        tool_calls: list[ToolCall] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_call_fragments):
            frag = tool_call_fragments[idx]
            args = self._safe_json(frag["arguments"])
            tool_calls.append(ToolCall(id=frag["id"], name=frag["name"], args=args))
            assistant_tool_calls.append(
                {
                    "id": frag["id"],
                    "type": "function",
                    "function": {"name": frag["name"], "arguments": frag["arguments"]},
                }
            )

        assistant_message: dict[str, Any] = {"role": "assistant", "content": full_content}
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls

        return AgentDecision(
            content=full_content,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            usage=token_usage,
        )

    def _parse_decision(self, message: Any, usage: TokenUsage | None = None) -> AgentDecision:
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls: list[ToolCall] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            args = self._safe_json(tc.function.arguments)
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
            assistant_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
            )

        assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content}
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls

        return AgentDecision(
            content=message.content,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            usage=usage or TokenUsage(),
        )

    @staticmethod
    def _safe_json(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
