"""Completion checks for single-agent tasks."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CompletionState:
    """Facts collected by the runtime before allowing a task to finish."""

    answer_generated: bool = False
    plan_generated: bool = False
    diff_checked: bool = False
    verification_done: bool = False
    verification_passed: bool = False
    unresolved_errors: bool = False
    files_changed: bool = False


class CompletionGuard:
    """Pure completion policy; the LLM cannot bypass these checks."""

    @staticmethod
    def can_finish(mode: object, state: CompletionState) -> bool:
        task_mode = getattr(mode, "value", str(mode)).upper()
        if task_mode == "ANSWER":
            return state.answer_generated and not state.files_changed
        if task_mode == "PLAN":
            return state.plan_generated and not state.files_changed
        if task_mode == "MODIFY":
            return (
                state.diff_checked
                and state.verification_done
                and state.verification_passed
                and not state.unresolved_errors
            )
        return False
