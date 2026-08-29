from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from memcodeagent.llm import LlmClient


def _rough_token_count(text: str) -> int:
    """Rough token estimate (~3 chars/token). Not an exact tokenizer count,
    just enough to keep the sliding window inside the model's context limit."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def _message_token_count(message: dict[str, Any]) -> int:
    content = message.get("content") or ""
    total = _rough_token_count(content if isinstance(content, str) else str(content))
    for tc in message.get("tool_calls", []) or []:
        total += _rough_token_count(str(tc))
    return total


@dataclass(slots=True)
class ContextManager:
    """Sliding-window context management with optional summarization.

    Keeps the leading system message(s) untouched, and keeps only the most
    recent `max_turns` user-initiated turns. A "turn" is one user message plus
    every assistant/tool message that follows it, up to (not including) the
    next user message. Splitting on turn boundaries (instead of a raw message
    count) guarantees we never cut a tool_calls/tool-result pair in half,
    which would otherwise make the next API request invalid.

    If the kept turns still exceed `max_tokens` (rough estimate), the oldest
    turns are dropped one at a time until the budget is met or only the most
    recent turn remains.

    When `enable_summarization` is True and an LlmClient is provided, dropped
    turns are summarized into a compact representation instead of being
    completely discarded, preserving key decisions and context.
    """

    max_turns: int = 20
    max_tokens: int = 24000
    enable_summarization: bool = True
    llm_client: "LlmClient | None" = None
    working_memory: Any | None = None
    _summary_cache: str | None = None  # Cached summary of previously dropped turns
    _summary_dropped_count: int = 0
    last_trim_notice: str | None = None

    def persist_state(self) -> dict[str, Any]:
        return {
            "summary_cache": self._summary_cache,
            "summary_dropped_count": self._summary_dropped_count,
        }

    def restore_state(self, data: dict[str, Any] | None) -> None:
        if not isinstance(data, dict):
            return
        summary = data.get("summary_cache")
        self._summary_cache = str(summary) if summary else None
        self._summary_dropped_count = int(data.get("summary_dropped_count", 0))

    def summarize_recent(self, messages: list[dict[str, Any]], max_turns: int = 8) -> str:
        """Summarize recent conversation turns for user-facing history."""
        rest = [m for m in messages if m.get("role") != "system"]
        turns = self._split_into_turns(rest)[-max(1, max_turns):]
        if not turns:
            return ""

        recent = [msg for turn in turns for msg in turn]
        prompt = (
            "Summarize this recent coding-agent conversation for the user. "
            "Cover the overall task, decisions made, files/actions, test status, "
            "and unresolved issues. Use concise Markdown with short headings and "
            "bullets. Do not repeat raw tool logs or mention this prompt.\n\n"
            + self._build_summary_prompt(recent)
        )
        if self.llm_client is None:
            return ""
        try:
            response = self.llm_client.next_action(
                [
                    {
                        "role": "system",
                        "content": (
                            "You create concise, readable status summaries for a coding-agent user."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools_enabled=False,
            )
            return response.content or ""
        except (AttributeError, TypeError, Exception):
            return ""

    def trim(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a trimmed copy of `messages` for sending to the model.

        Does not mutate the input list -- callers keep the full history for
        display, memory persistence, and /history, and only the trimmed copy
        goes over the wire.

        When summarization is enabled and turns are dropped, they are summarized
        and injected as a system-level context message.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        turns = self._split_into_turns(rest)
        original_turn_count = len(turns)

        # Apply turn limit
        if len(turns) > self.max_turns:
            turns = turns[-self.max_turns :]

        # Apply token limit
        while len(turns) > 1 and self._total_tokens(system_msgs, turns) > self.max_tokens:
            turns.pop(0)

        # Generate summary if turns were dropped
        dropped_turn_count = original_turn_count - len(turns)
        self.last_trim_notice = None
        if dropped_turn_count > 0:
            self.last_trim_notice = (
                f"Context compressed: {dropped_turn_count} older turn(s) omitted from this request; "
                "full history remains available."
            )
        summary_msg = None
        if (
            dropped_turn_count > 0
            and dropped_turn_count != self._summary_dropped_count
            and self.enable_summarization
            and self.llm_client
        ):
            summary_msg = self._get_or_create_summary(messages, len(turns))
            self._summary_dropped_count = dropped_turn_count
        elif dropped_turn_count > 0 and self._summary_cache:
            summary_msg = {"role": "system", "content": f"[Context from earlier conversation turns:]\n{self._summary_cache}"}

        trimmed_rest = [msg for turn in turns for msg in turn]

        # Inject summary between system messages and conversation turns
        working_msg = None
        if self.working_memory is not None:
            working_msg = {"role": "system", "content": self.working_memory.to_context_text()}
        if summary_msg:
            result = system_msgs + [summary_msg] + ([working_msg] if working_msg else []) + trimmed_rest
        else:
            result = system_msgs + ([working_msg] if working_msg else []) + trimmed_rest

        # A single active turn can itself exceed the budget (for example, a
        # large file read followed by test output), so turn-level trimming
        # alone is not sufficient. Keep the user request and the newest tool
        # observations, shortening older tool payloads first.
        if self._total_tokens(result, []) > self.max_tokens:
            result = self._fit_oversized_messages(result)
        return result

    def _fit_oversized_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        budget_chars = max(self.max_tokens * 3, 3000)
        result = [dict(message) for message in messages]
        total_chars = sum(len(str(message.get("content") or "")) for message in result)
        for message in result:
            if total_chars <= budget_chars:
                break
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            if len(content) <= 1200:
                continue
            keep = max(600, min(1200, len(content) // 4))
            shortened = content[:keep] + "\n...[older tool output shortened for context budget]"
            total_chars -= len(content) - len(shortened)
            message["content"] = shortened
        return result

    def stats(self, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Return counts useful for a /context-style status display."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        turns = self._split_into_turns(rest)
        # Stats must be side-effect free: do not invoke an extra summarization LLM call.
        summarize = self.enable_summarization
        self.enable_summarization = False
        try:
            trimmed = self.trim(messages)
        finally:
            self.enable_summarization = summarize
        return {
            "total_messages": len(messages),
            "total_turns": len(turns),
            "kept_messages": len(trimmed),
            "kept_turns": min(len(turns), self.max_turns),
            "estimated_tokens_full": self._total_tokens(system_msgs, turns),
            "estimated_tokens_kept": sum(_message_token_count(m) for m in trimmed),
        }

    @staticmethod
    def _split_into_turns(rest: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in rest:
            if msg.get("role") == "user" and current:
                turns.append(current)
                current = []
            current.append(msg)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _total_tokens(system_msgs: list[dict[str, Any]], turns: list[list[dict[str, Any]]]) -> int:
        total = sum(_message_token_count(m) for m in system_msgs)
        for turn in turns:
            total += sum(_message_token_count(m) for m in turn)
        return total

    def _get_or_create_summary(self, messages: list[dict[str, Any]], kept_turns: int) -> dict[str, Any]:
        """Generate or retrieve a cached summary of dropped conversation turns.

        Only summarizes the newly dropped turns (not previously summarized ones),
        and appends to the existing summary cache if one exists.
        """
        # Extract the dropped turns that need summarizing
        rest = [m for m in messages if m.get("role") != "system"]
        all_turns = self._split_into_turns(rest)
        dropped_turns = all_turns[:-kept_turns] if kept_turns > 0 else all_turns

        if not dropped_turns:
            return {"role": "system", "content": self._summary_cache or ""}

        # Flatten dropped turns into message list
        dropped_messages = [msg for turn in dropped_turns for msg in turn]

        # Build summarization prompt
        summary_prompt = self._build_summary_prompt(dropped_messages)

        try:
            # Call LLM to generate summary
            summary_response = self.llm_client.next_action([
                {
                    "role": "system",
                    "content": (
                        "You are a context compression assistant. Summarize conversation history "
                        "into a compact form that preserves: key decisions made, constraints specified, "
                        "architectural choices, user preferences, failed attempts and their reasons, "
                        "and any unresolved issues. Be concise but complete."
                    )
                },
                {"role": "user", "content": summary_prompt}
            ])

            new_summary = summary_response.content or "(no summary generated)"

            # Merge with existing cache if present
            if self._summary_cache:
                combined = f"{self._summary_cache}\n\n[Additional context from later turns:]\n{new_summary}"
                self._summary_cache = combined
            else:
                self._summary_cache = new_summary

            return {
                "role": "system",
                "content": f"[Context from earlier conversation turns:]\n{self._summary_cache}"
            }

        except Exception as e:
            # Fallback: if summarization fails, return a simple truncation notice
            return {
                "role": "system",
                "content": f"[{len(dropped_turns)} earlier conversation turns omitted due to context limits]"
            }

    @staticmethod
    def _build_summary_prompt(dropped_messages: list[dict[str, Any]]) -> str:
        """Convert dropped messages into a prompt for summarization."""
        lines = ["Summarize the following conversation turns, preserving all important context:\n"]

        for msg in dropped_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                lines.append(f"\nUser: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                    lines.append(f"\nAssistant called tools: {', '.join(tool_names)}")
                if content:
                    lines.append(f"\nAssistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                # Truncate tool output to avoid overwhelming the summary prompt
                truncated = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"\nTool ({tool_name}): {truncated}")

        return "\n".join(lines)
