"""Simple runtime event rendering for CLI output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentEventKind(str, Enum):
    PHASE = "phase"
    TOOL = "tool"
    ALERT = "alert"
    DONE = "done"
    PAUSED = "paused"


class ToolEventKind(str, Enum):
    CALL = "call"
    RESULT = "result"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: AgentEventKind
    message: str
    detail: str = ""
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """Machine-facing tool trace, kept separate from user-facing progress."""

    kind: ToolEventKind
    tool: str
    payload: dict[str, Any] | None = None


def format_agent_event(event: AgentEvent) -> str:
    if event.detail:
        return f"{event.message} — {event.detail}"
    return event.message
