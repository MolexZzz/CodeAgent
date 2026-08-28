"""Single-agent runtime controller.

This module owns orchestration state, while the LLM remains responsible for
technical decisions and the tool executor remains responsible for local I/O.
It is intentionally dependency-injected so the runtime can be tested without
network access or a real workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from memcodeagent.llm import AgentDecision
from memcodeagent.policy import PolicyAction, ToolPolicy
from memcodeagent.progress import ProgressAlert, ProgressMonitor
from memcodeagent.runtime import Phase, RuntimeEvent, StateMachine
from memcodeagent.tools import ToolObservation


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
        max_tool_calls: int = 64,
        tool_policy: ToolPolicy | None = None,
        confirmation_callback: Callable[[Any], bool] | None = None,
        progress_monitor: ProgressMonitor | None = None,
    ) -> None:
        self.llm = llm
        self.tool_executor = tool_executor
        self.context_manager = context_manager
        self.max_steps = max(1, max_steps)
        self.max_tool_calls = max(1, max_tool_calls)
        self.tool_policy = tool_policy
        self.confirmation_callback = confirmation_callback
        self.progress_monitor = progress_monitor or ProgressMonitor()
        self.state_machine = StateMachine()
        self.task_mode: str | None = None
        self.step_count = 0
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.last_decision: AgentDecision | None = None
        self.last_result: str | None = None
        self.interrupted = False
        self.last_transition_error: str | None = None
        self.last_progress_alert: Any | None = None

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
        self.last_transition_error = None
        self.last_progress_alert = None
        self.progress_monitor.reset()
        self.state_machine = StateMachine()
        self.state_machine.transition(RuntimeEvent.TASK_STARTED)
        requested = {
            "ANSWER": RuntimeEvent.ANSWER_REQUESTED,
            "PLAN": RuntimeEvent.PLAN_REQUESTED,
            "MODIFY": RuntimeEvent.MODIFY_REQUESTED,
        }.get(self.task_mode)
        if requested is not None:
            self.state_machine.transition(requested)

    def transition(self, event: RuntimeEvent) -> bool:
        """Advance runtime state through a guarded event."""
        try:
            self.state_machine.transition(event)
        except ValueError as exc:
            self.last_transition_error = str(exc)
            return False
        self.last_transition_error = None
        return True

    def step(
        self,
        messages: list[dict[str, Any]],
        *,
        before_tools: Callable[[AgentDecision], None] | None = None,
        tool_guard: Callable[[Any], Any | None] | None = None,
        tool_context: Callable[[Any], dict[str, Any]] | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> ControllerStep:
        """Run exactly one model decision and its local tool calls.

        The method deliberately does not approve/deny tools yet; that policy is
        the next runtime layer. It does guarantee that every decision and
        observation is appended to the supplied conversation.
        """
        """Run exactly one model decision and its local tool calls."""
        if self.step_count >= self.max_steps:
            raise RuntimeError("controller step budget exhausted")

        self.step_count += 1
        prompt_messages = (
            self.context_manager.trim(messages)
            if self.context_manager is not None
            else messages
        )
        try:
            if stream:
                decision = self.llm.next_action(
                    prompt_messages,
                    stream=True,
                    on_chunk=on_chunk,
                )
            else:
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
                self.transition(RuntimeEvent.ANSWER_GENERATED)
            elif self.task_mode == "PLAN":
                self.transition(RuntimeEvent.PLAN_GENERATED)
            return ControllerStep(decision=decision, phase=self.phase)

        if before_tools is not None:
            before_tools(decision)

        observations: list[Any] = []
        for tool_call in decision.tool_calls:
            if len(self.tool_calls) >= self.max_tool_calls:
                observation = ToolObservation(
                    tool_call.name,
                    False,
                    f"已达到工具调用预算 {self.max_tool_calls}，任务暂停。",
                    tool_call.id,
                )
                observations.append(observation)
                messages.append(observation.to_message())
                self.last_progress_alert = ProgressAlert(
                    "tool_budget",
                    f"已达到工具调用预算 {self.max_tool_calls}。",
                )
                continue
            self.tool_calls.append((tool_call.name, dict(tool_call.args)))
            observation = None
            alert = self.progress_monitor.record_tool(tool_call.name, tool_call.args)
            self.last_progress_alert = alert
            if alert is not None and alert.kind == "duplicate_tool":
                observation = ToolObservation(
                    tool_call.name,
                    False,
                    alert.message,
                    tool_call.id,
                )
            if tool_guard is not None:
                if observation is None:
                    observation = tool_guard(tool_call)
            elif self.tool_policy is not None:
                context = tool_context(tool_call) if tool_context else {}
                policy = self.tool_policy.evaluate(
                    phase=str(context.get("phase", "IMPLEMENTING")),
                    tool_name=tool_call.name,
                    approval_required=bool(context.get("approval_required", False)),
                    explored=bool(context.get("explored", True)),
                    protected_test=bool(context.get("protected_test", False)),
                    duplicate=bool(context.get("duplicate", False)),
                )
                if policy.action == PolicyAction.DENY:
                    observation = ToolObservation(
                        tool_call.name, False, policy.reason, tool_call.id
                    )
                elif (
                    policy.action == PolicyAction.CONFIRM
                    and (
                        self.confirmation_callback is None
                        or not self.confirmation_callback(tool_call)
                    )
                ):
                    observation = ToolObservation(
                        tool_call.name, False, "Tool call denied by user.", tool_call.id
                    )
                else:
                    observation = None
            if observation is None:
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
            progress_alert = self.progress_monitor.record_observation(
                tool_name=tool_call.name,
                ok=bool(getattr(observation, "ok", False)),
                content=str(getattr(observation, "content", "")),
            )
            if progress_alert is not None:
                self.last_progress_alert = progress_alert

        return ControllerStep(decision=decision, observations=observations, phase=self.phase)

    def budget_exhausted(self) -> bool:
        return self.step_count >= self.max_steps

    def reset_budget(self) -> None:
        """Start another user-approved step budget without losing task state."""
        self.step_count = 0

    def persist_state(self) -> dict[str, Any]:
        """Return serializable runtime state for session persistence."""
        return {
            "task_mode": self.task_mode,
            "phase": self.phase.value,
            "step_count": self.step_count,
            "max_tool_calls": self.max_tool_calls,
            "tool_calls": [
                {"name": name, "args": args} for name, args in self.tool_calls
            ],
            "last_result": self.last_result,
            "interrupted": self.interrupted,
        }
