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


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: AgentEventKind
    message: str
    detail: str = ""
    payload: dict[str, Any] | None = None


def format_agent_event(event: AgentEvent) -> str:
    if event.detail:
        return f"{event.message} — {event.detail}"
    return event.message

