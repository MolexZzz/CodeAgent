"""Runtime state primitives for the single-agent controller.

The LLM may propose actions, but it cannot mutate these objects directly.
Only controller code should emit runtime events and advance the state machine.
"""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    IDLE = "IDLE"
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    CONFIRM = "CONFIRM"
    IMPLEMENT = "IMPLEMENT"
    DIFF_CHECK = "DIFF_CHECK"
    TEST = "TEST"
    VERIFY = "VERIFY"
    ANSWER = "ANSWER"
    PLAN_ONLY = "PLAN_ONLY"
    DONE = "DONE"
    PAUSED = "PAUSED"


class RuntimeEvent(str, Enum):
    TASK_STARTED = "task_started"
    ANSWER_REQUESTED = "answer_requested"
    PLAN_REQUESTED = "plan_requested"
    MODIFY_REQUESTED = "modify_requested"
    EXPLORATION_COMPLETE = "exploration_complete"
    PLAN_READY = "plan_ready"
    USER_APPROVED = "user_approved"
    USER_REJECTED = "user_rejected"
    IMPLEMENTATION_DONE = "implementation_done"
    DIFF_CHECKED = "diff_checked"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    HYPOTHESIS_INVALID = "hypothesis_invalid"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ANSWER_GENERATED = "answer_generated"
    PLAN_GENERATED = "plan_generated"


class InvalidTransition(ValueError):
    """Raised when a runtime event is not legal in the current phase."""


class TransitionGuard:
    """Pure transition policy; it does not execute tools or choose solutions."""

    _TRANSITIONS: dict[tuple[Phase, RuntimeEvent], Phase] = {
        (Phase.IDLE, RuntimeEvent.TASK_STARTED): Phase.UNDERSTAND,
        (Phase.UNDERSTAND, RuntimeEvent.ANSWER_REQUESTED): Phase.ANSWER,
        (Phase.UNDERSTAND, RuntimeEvent.PLAN_REQUESTED): Phase.PLAN_ONLY,
        (Phase.UNDERSTAND, RuntimeEvent.MODIFY_REQUESTED): Phase.UNDERSTAND,
        (Phase.UNDERSTAND, RuntimeEvent.EXPLORATION_COMPLETE): Phase.PLAN,
        (Phase.PLAN, RuntimeEvent.PLAN_READY): Phase.CONFIRM,
        (Phase.CONFIRM, RuntimeEvent.USER_APPROVED): Phase.IMPLEMENT,
        (Phase.CONFIRM, RuntimeEvent.USER_REJECTED): Phase.PAUSED,
        (Phase.IMPLEMENT, RuntimeEvent.IMPLEMENTATION_DONE): Phase.DIFF_CHECK,
        (Phase.DIFF_CHECK, RuntimeEvent.DIFF_CHECKED): Phase.TEST,
        (Phase.TEST, RuntimeEvent.TEST_PASSED): Phase.VERIFY,
        (Phase.TEST, RuntimeEvent.TEST_FAILED): Phase.IMPLEMENT,
        (Phase.IMPLEMENT, RuntimeEvent.HYPOTHESIS_INVALID): Phase.UNDERSTAND,
        (Phase.ANSWER, RuntimeEvent.ANSWER_GENERATED): Phase.DONE,
        (Phase.PLAN_ONLY, RuntimeEvent.PLAN_GENERATED): Phase.DONE,
        (Phase.VERIFY, RuntimeEvent.ANSWER_GENERATED): Phase.DONE,
        (Phase.VERIFY, RuntimeEvent.TEST_FAILED): Phase.IMPLEMENT,
    }

    @classmethod
    def next_phase(cls, current: Phase, event: RuntimeEvent) -> Phase:
        try:
            return cls._TRANSITIONS[(current, event)]
        except KeyError as exc:
            raise InvalidTransition(
                f"event {event.value!r} is not allowed in phase {current.value!r}"
            ) from exc


class StateMachine:
    """Small stateful wrapper around :class:`TransitionGuard`."""

    def __init__(self, phase: Phase = Phase.IDLE) -> None:
        self.phase = phase

    def transition(self, event: RuntimeEvent) -> Phase:
        self.phase = TransitionGuard.next_phase(self.phase, event)
        return self.phase

    def try_transition(self, event: RuntimeEvent) -> bool:
        try:
            self.transition(event)
        except InvalidTransition:
            return False
        return True

