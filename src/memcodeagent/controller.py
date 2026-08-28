"""Single-agent runtime controller.

This module owns orchestration state, while the LLM remains responsible for
technical decisions and the tool executor remains responsible for local I/O.
It is intentionally dependency-injected so the runtime can be tested without
network access or a real workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memcodeagent.llm import AgentDecision
from memcodeagent.runtime import Phase, RuntimeEvent, StateMachine


@dataclass(slots=True)
class ControllerStep:
    """Result of one controller iteration."""

    decision: AgentDecision
    observations: list[Any] = field(default_factory=list)
    phase: Phase = Phase.IDLE
    interrupted: bool = False

    @property
    def finished(self) -> bool:
        return self.decision.is_final and not self.interrupted


class AgentController:
    """Drive one model/tool turn without deciding the technical solution."""

    def __init__(
        self,
        *,
        llm: Any,
        tool_executor: Any,
        context_manager: Any | None = None,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.tool_executor = tool_executor
        self.context_manager = context_manager
        self.max_steps = max(1, max_steps)
        self.state_machine = StateMachine()
        self.task_mode: str | None = None
        self.step_count = 0
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.last_decision: AgentDecision | None = None
        self.last_result: str | None = None
        self.interrupted = False

    @property
    def phase(self) -> Phase:
        return self.state_machine.phase

    def handle_user_request(self, mode: str, messages: list[dict[str, Any]]) -> None:
        """Start a task and select its initial mode through runtime events."""
        self.task_mode = mode.upper()
        self.step_count = 0
        self.tool_calls.clear()
        self.last_decision = None
        self.last_result = None
        self.interrupted = False
        self.state_machine = StateMachine()
        self.state_machine.transition(RuntimeEvent.TASK_STARTED)
        requested = {
            "ANSWER": RuntimeEvent.ANSWER_REQUESTED,
            "PLAN": RuntimeEvent.PLAN_REQUESTED,
            "MODIFY": RuntimeEvent.MODIFY_REQUESTED,
        }.get(self.task_mode)
        if requested is not None:
            self.state_machine.transition(requested)

    def step(self, messages: list[dict[str, Any]]) -> ControllerStep:
        """Run exactly one model decision and its local tool calls.

        The method deliberately does not approve/deny tools yet; that policy is
        the next runtime layer. It does guarantee that every decision and
        observation is appended to the supplied conversation.
        """
        if self.step_count >= self.max_steps:
            raise RuntimeError("controller step budget exhausted")

        self.step_count += 1
        prompt_messages = (
            self.context_manager.trim(messages)
            if self.context_manager is not None
            else messages
        )
        try:
            decision = self.llm.next_action(prompt_messages)
        except KeyboardInterrupt:
            self.interrupted = True
            return ControllerStep(
                decision=AgentDecision(
                    content="任务已被用户中断。",
                    assistant_message={"role": "assistant", "content": "任务已被用户中断。"},
                ),
                phase=Phase.PAUSED,
                interrupted=True,
            )

        self.last_decision = decision
        messages.append(decision.assistant_message)
        if decision.is_final:
            self.last_result = decision.content or ""
            if self.task_mode == "ANSWER":
                self.state_machine.try_transition(RuntimeEvent.ANSWER_GENERATED)
            elif self.task_mode == "PLAN":
                self.state_machine.try_transition(RuntimeEvent.PLAN_GENERATED)
            return ControllerStep(decision=decision, phase=self.phase)

        observations: list[Any] = []
        for tool_call in decision.tool_calls:
            self.tool_calls.append((tool_call.name, dict(tool_call.args)))
            try:
                observation = self.tool_executor.execute(
                    tool_call.name,
                    tool_call.args,
                    tool_call.id,
                )
            except KeyboardInterrupt:
                self.interrupted = True
                return ControllerStep(
                    decision=decision,
                    observations=observations,
                    phase=Phase.PAUSED,
                    interrupted=True,
                )
            observations.append(observation)
            messages.append(observation.to_message())

        return ControllerStep(decision=decision, observations=observations, phase=self.phase)

    def persist_state(self) -> dict[str, Any]:
        """Return serializable runtime state for session persistence."""
        return {
            "task_mode": self.task_mode,
            "phase": self.phase.value,
            "step_count": self.step_count,
            "tool_calls": [
                {"name": name, "args": args} for name, args in self.tool_calls
            ],
            "last_result": self.last_result,
            "interrupted": self.interrupted,
        }

