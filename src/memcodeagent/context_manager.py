from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    """Sliding-window context management -- no summarization, no compression.

    Keeps the leading system message(s) untouched, and keeps only the most
    recent `max_turns` user-initiated turns. A "turn" is one user message plus
    every assistant/tool message that follows it, up to (not including) the
    next user message. Splitting on turn boundaries (instead of a raw message
    count) guarantees we never cut a tool_calls/tool-result pair in half,
    which would otherwise make the next API request invalid.

    If the kept turns still exceed `max_tokens` (rough estimate), the oldest
    turns are dropped one at a time until the budget is met or only the most
    recent turn remains.
    """

    max_turns: int = 20
    max_tokens: int = 24000

    def trim(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a trimmed copy of `messages` for sending to the model.

        Does not mutate the input list -- callers keep the full history for
        display, memory persistence, and /history, and only the trimmed copy
        goes over the wire.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        turns = self._split_into_turns(rest)
        if len(turns) > self.max_turns:
            turns = turns[-self.max_turns :]

        while len(turns) > 1 and self._total_tokens(system_msgs, turns) > self.max_tokens:
            turns.pop(0)

        trimmed_rest = [msg for turn in turns for msg in turn]
        return system_msgs + trimmed_rest

    def stats(self, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Return counts useful for a /context-style status display."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        turns = self._split_into_turns(rest)
        trimmed = self.trim(messages)
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
