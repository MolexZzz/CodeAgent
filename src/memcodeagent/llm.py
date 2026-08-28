from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv, set_key
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
            "name": "summarize_tree",
            "description": "Return a compact summary of files in the workspace tree with sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_files": {"type": "integer", "description": "Maximum number of files to include."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a bounded section of a text file inside the workspace, with line numbers. "
                "Defaults to at most 400 lines; use start_line/end_line for large files."
            ),
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
            "description": (
                "Run a shell command with the workspace root as the current directory. "
                "Do not add `cd /workspace` or change to another project; use relative paths."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "diff_summary",
            "description": "Return a git diff summary and changed file list for the current workspace.",
            "parameters": {"type": "object", "properties": {}},
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

    # Model registry: each model knows which provider it belongs to and what credentials to use.
    # Structure: {model_name: {"provider": "...", "api_key_env": "...", "base_url_env": "..."}}
    MODEL_REGISTRY = {
        "deepseek-chat": {
            "provider": "DeepSeek",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url_env": "DEEPSEEK_BASE_URL",
        },
        "deepseek-reasoner": {
            "provider": "DeepSeek",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url_env": "DEEPSEEK_BASE_URL",
        },
        "deepseek-v4-flash": {
            "provider": "DeepSeek",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url_env": "DEEPSEEK_BASE_URL",
        },
        "gpt-4o": {
            "provider": "OpenAI",
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
        },
        "gpt-4o-mini": {
            "provider": "OpenAI",
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
        },
        "gpt-4-turbo": {
            "provider": "OpenAI",
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
        },
        "claude-3-5-sonnet-20241022": {
            "provider": "Anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url_env": "ANTHROPIC_BASE_URL",
        },
        "claude-3-5-haiku-20241022": {
            "provider": "Anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url_env": "ANTHROPIC_BASE_URL",
        },
    }

    @property
    def AVAILABLE_MODELS(self) -> list[str]:
        """Return list of models that have credentials configured."""
        return [name for name in self.MODEL_REGISTRY if self._model_is_configured(name)]

    def __init__(self) -> None:
        # Load .env from current directory or home directory
        load_dotenv()  # Load from current directory
        load_dotenv(Path.home() / ".memcode" / ".env")  # Load from ~/.memcode/.env

        self._env_path = self._find_env_path()

        # Load the saved default model, or pick the first configured one
        saved_model = os.getenv("MEMCODE_MODEL")
        if saved_model and self._model_is_configured(saved_model):
            self.model = saved_model
        else:
            # Fallback: pick the first model that has credentials configured
            configured = self.AVAILABLE_MODELS
            self.model = configured[0] if configured else "gpt-4o-mini"

        # Credentials are resolved dynamically per request in next_action()
        self.api_key: str | None = None
        self.base_url: str | None = None

    def _find_env_path(self) -> Path:
        """Find the .env file path (current directory or ~/.memcode/.env)."""
        local_env = Path.cwd() / ".env"
        if local_env.exists():
            return local_env
        home_env = Path.home() / ".memcode" / ".env"
        home_env.parent.mkdir(parents=True, exist_ok=True)
        return home_env

    def _model_is_configured(self, model: str) -> bool:
        """Check if a model's provider credentials are configured in the environment."""
        if model not in self.MODEL_REGISTRY:
            return False
        config = self.MODEL_REGISTRY[model]
        api_key = os.getenv(config["api_key_env"])
        return api_key is not None and api_key.strip() != ""

    def _resolve_credentials(self, model: str) -> tuple[str | None, str | None]:
        """Resolve API key and base URL for the given model from environment variables."""
        if model not in self.MODEL_REGISTRY:
            return None, None
        config = self.MODEL_REGISTRY[model]
        api_key = os.getenv(config["api_key_env"])
        base_url = os.getenv(config["base_url_env"]) or None
        return api_key, base_url

    def get_model_info(self, model: str) -> dict[str, str]:
        """Get provider info and configuration status for a model."""
        if model not in self.MODEL_REGISTRY:
            return {"provider": "Unknown", "configured": "No"}
        config = self.MODEL_REGISTRY[model]
        configured = "Yes" if self._model_is_configured(model) else "No"
        return {
            "provider": config["provider"],
            "configured": configured,
            "api_key_env": config["api_key_env"],
            "base_url_env": config["base_url_env"],
        }

    def set_model(self, model: str, persist: bool = False) -> None:
        """Switch to a different model, optionally persisting to .env.

        Raises ValueError if the model's provider credentials are not configured.
        """
        if not self._model_is_configured(model):
            info = self.get_model_info(model)
            raise ValueError(
                f"Cannot switch to '{model}': missing credentials for provider "
                f"{info.get('provider', 'Unknown')}. Set {info.get('api_key_env', '<API_KEY>')} "
                "in your .env file first."
            )
        self.model = model
        if persist:
            set_key(str(self._env_path), "MEMCODE_MODEL", model)

    def next_action(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        tools_enabled: bool = True,
    ) -> AgentDecision:
        api_key, base_url = self._resolve_credentials(self.model)
        self.api_key, self.base_url = api_key, base_url

        if not api_key:
            info = self.get_model_info(self.model)
            api_key_env = info.get("api_key_env", "OPENAI_API_KEY")
            return AgentDecision(
                content=(
                    f"LLM is not configured yet. Set {api_key_env}, then rerun the task. "
                    "The CLI framework, tool executor, and memory layer are ready."
                ),
                assistant_message={
                    "role": "assistant",
                    "content": f"LLM is not configured yet. Set {api_key_env}, then rerun the task.",
                },
            )

        client = OpenAI(api_key=api_key, base_url=base_url)

        if stream:
            return self._next_action_streaming(client, messages, on_chunk)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS if tools_enabled else None,
            tool_choice="auto" if tools_enabled else "none",
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
