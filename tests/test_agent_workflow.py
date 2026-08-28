from pathlib import Path
from io import StringIO
from unittest.mock import Mock, patch

from rich.console import Console

from memcodeagent.agent import AgentConfig
from memcodeagent.agent import CodingAgent
from memcodeagent.agent import IntentRouter
from memcodeagent.agent import TaskMode
from memcodeagent.controller import AgentController
from memcodeagent.completion import CompletionGuard, CompletionState
from memcodeagent.llm import AgentDecision
from memcodeagent.memory.hybrid_retriever import RetrievalContext
from memcodeagent.policy import PolicyAction, ToolPolicy
from memcodeagent.progress import ProgressMonitor, ProgressSnapshot
from memcodeagent.verification import VerificationKind, classify_verification
from memcodeagent.runtime import InvalidTransition, Phase, RuntimeEvent, StateMachine, TransitionGuard


def test_phase_tool_permissions() -> None:
    read = {"list_files", "read_file", "search_text", "summarize_tree", "diff_summary"}
    assert all(CodingAgent._tool_allowed_in_phase("EXPLORING", name, False) for name in read)
    assert not CodingAgent._tool_allowed_in_phase("EXPLORING", "apply_patch", False)
    assert CodingAgent._tool_allowed_in_phase("IMPLEMENTING", "apply_patch", True)
    assert CodingAgent._tool_allowed_in_phase("TESTING", "run_command", True)
    assert CodingAgent._tool_allowed_in_phase("VERIFYING", "diff_summary", True)
    assert not CodingAgent._tool_allowed_in_phase("COMPLETED", "read_file", True)


def test_transition_guard_accepts_modify_workflow() -> None:
    machine = StateMachine()
    assert machine.transition(RuntimeEvent.TASK_STARTED) == Phase.UNDERSTAND
    assert machine.transition(RuntimeEvent.EXPLORATION_COMPLETE) == Phase.PLAN
    assert machine.transition(RuntimeEvent.PLAN_READY) == Phase.CONFIRM
    assert machine.transition(RuntimeEvent.USER_APPROVED) == Phase.IMPLEMENT
    assert machine.transition(RuntimeEvent.IMPLEMENTATION_DONE) == Phase.DIFF_CHECK
    assert machine.transition(RuntimeEvent.DIFF_CHECKED) == Phase.TEST
    assert machine.transition(RuntimeEvent.TEST_PASSED) == Phase.VERIFY
    assert machine.transition(RuntimeEvent.ANSWER_GENERATED) == Phase.DONE


def test_transition_guard_rejects_illegal_phase_changes() -> None:
    machine = StateMachine(Phase.PLAN)
    try:
        machine.transition(RuntimeEvent.USER_APPROVED)
    except InvalidTransition as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("illegal transition should be rejected")

    assert TransitionGuard.next_phase(Phase.TEST, RuntimeEvent.TEST_FAILED) == Phase.IMPLEMENT
    assert machine.try_transition(RuntimeEvent.USER_REJECTED) is False


def test_agent_controller_runs_one_tool_turn() -> None:
    class FakeObservation:
        ok = True
        tool_name = "list_files"
        content = "app.py"

        def to_message(self):
            return {"role": "tool", "content": self.content}

    class FakeTools:
        def __init__(self):
            self.calls = []

        def execute(self, name, args, tool_call_id):
            self.calls.append((name, args, tool_call_id))
            return FakeObservation()

    call = type("ToolCall", (), {"id": "1", "name": "list_files", "args": {"glob": "*.py"}})()
    decision = AgentDecision(
        tool_calls=[call],
        assistant_message={"role": "assistant", "content": None},
    )
    llm = type("FakeLlm", (), {"next_action": lambda self, messages: decision})()
    tools = FakeTools()
    controller = AgentController(llm=llm, tool_executor=tools, max_steps=2)
    messages = [{"role": "user", "content": "inspect"}]

    controller.handle_user_request("ANSWER", messages)
    result = controller.step(messages)

    assert result.finished is False
    assert result.phase == Phase.ANSWER
    assert tools.calls == [("list_files", {"glob": "*.py"}, "1")]
    assert messages[-1] == {"role": "tool", "content": "app.py"}
    assert controller.persist_state()["step_count"] == 1


def test_agent_controller_final_answer_transitions_to_done() -> None:
    decision = AgentDecision(
        content="done",
        assistant_message={"role": "assistant", "content": "done"},
    )
    llm = type("FakeLlm", (), {"next_action": lambda self, messages: decision})()
    controller = AgentController(llm=llm, tool_executor=object())
    messages = [{"role": "user", "content": "why"}]

    controller.handle_user_request("ANSWER", messages)
    result = controller.step(messages)

    assert result.finished is True
    assert controller.phase == Phase.DONE
    assert controller.last_result == "done"


def test_tool_policy_permission_matrix() -> None:
    policy = ToolPolicy()
    assert policy.evaluate(
        phase="INSPECTING",
        tool_name="read_file",
        approval_required=True,
    ).action == PolicyAction.ALLOW
    assert policy.evaluate(
        phase="INSPECTING",
        tool_name="apply_patch",
        approval_required=True,
    ).action == PolicyAction.DENY
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="apply_patch",
        approval_required=True,
    ).action == PolicyAction.CONFIRM
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="apply_patch",
        approval_required=False,
    ).action == PolicyAction.ALLOW
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="write_file",
        approval_required=False,
        protected_test=True,
    ).action == PolicyAction.DENY


def test_controller_policy_can_deny_without_executing_tool() -> None:
    class FakeObservation:
        ok = True
        tool_name = "apply_patch"
        content = "should not run"

        def to_message(self):
            return {"role": "tool", "content": self.content}

    class FakeTools:
        def __init__(self):
            self.calls = 0

        def execute(self, *_args):
            self.calls += 1
            return FakeObservation()

    call = type("ToolCall", (), {"id": "1", "name": "apply_patch", "args": {}})()
    decision = AgentDecision(
        tool_calls=[call],
        assistant_message={"role": "assistant", "content": None},
    )
    llm = type("FakeLlm", (), {"next_action": lambda self, messages: decision})()
    tools = FakeTools()
    controller = AgentController(
        llm=llm,
        tool_executor=tools,
        tool_policy=ToolPolicy(),
    )
    messages = [{"role": "user", "content": "inspect"}]
    controller.handle_user_request("ANSWER", messages)

    result = controller.step(
        messages,
        tool_context=lambda _tool_call: {
            "phase": "ANSWERING",
            "approval_required": True,
        },
    )

    assert result.observations[0].ok is False
    assert tools.calls == 0


def test_completion_guard_requires_verification_for_modify() -> None:
    complete = CompletionState(
        diff_checked=True,
        verification_done=True,
        verification_passed=True,
    )
    assert CompletionGuard.can_finish(TaskMode.MODIFY, complete) is True
    assert CompletionGuard.can_finish(
        TaskMode.MODIFY,
        CompletionState(diff_checked=True, verification_done=False),
    ) is False
    assert CompletionGuard.can_finish(
        TaskMode.MODIFY,
        CompletionState(
            diff_checked=True,
            verification_done=True,
            verification_passed=False,
        ),
    ) is False


def test_completion_guard_read_only_modes_do_not_allow_changes() -> None:
    assert CompletionGuard.can_finish(
        TaskMode.ANSWER,
        CompletionState(answer_generated=True),
    ) is True
    assert CompletionGuard.can_finish(
        TaskMode.ANSWER,
        CompletionState(answer_generated=True, files_changed=True),
    ) is False
    assert CompletionGuard.can_finish(
        TaskMode.PLAN,
        CompletionState(plan_generated=True),
    ) is True


def test_progress_monitor_detects_duplicate_and_monotony() -> None:
    monitor = ProgressMonitor(duplicate_limit=1, monotony_limit=3)
    assert monitor.record_tool("read_file", {"path": "a.py"}) is None
    duplicate = monitor.record_tool("read_file", {"path": "a.py"})
    assert duplicate is not None
    assert duplicate.kind == "duplicate_tool"

    monitor.reset()
    assert monitor.record_tool("read_file", {"path": "a.py"}) is None
    assert monitor.record_tool("read_file", {"path": "b.py"}) is None
    monotony = monitor.record_tool("read_file", {"path": "c.py"})
    assert monotony is not None
    assert monotony.kind == "tool_monotony"


def test_progress_monitor_detects_no_progress() -> None:
    monitor = ProgressMonitor(no_progress_limit=2)
    snapshot = ProgressSnapshot()
    assert monitor.record_snapshot(snapshot) is None
    alert = monitor.record_snapshot(snapshot)
    assert alert is not None
    assert alert.kind == "no_progress"


def test_controller_enforces_tool_call_budget() -> None:
    call = type("ToolCall", (), {"id": "1", "name": "read_file", "args": {"path": "a.py"}})()
    decision = AgentDecision(
        tool_calls=[call],
        assistant_message={"role": "assistant", "content": None},
    )
    llm = type("FakeLlm", (), {"next_action": lambda self, messages: decision})()

    class Tools:
        def execute(self, *_args):
            raise AssertionError("tool must not execute after budget is exhausted")

    controller = AgentController(
        llm=llm,
        tool_executor=Tools(),
        max_tool_calls=1,
    )
    controller.handle_user_request("ANSWER", [{"role": "user", "content": "inspect"}])
    controller.tool_calls.append(("previous", {}))
    first = controller.step([{"role": "user", "content": "inspect"}])
    assert len(first.observations) == 1
    assert controller.last_progress_alert is not None
    assert controller.last_progress_alert.kind == "tool_budget"


def test_verification_results_are_classified() -> None:
    assert classify_verification(True, "exit_code=0").kind == VerificationKind.PASS
    assert classify_verification(False, "exit_code=1\nAssertionError: bad").kind == VerificationKind.ASSERTION_FAILURE
    assert classify_verification(False, "exit_code=1\nSyntaxError: invalid syntax").kind == VerificationKind.COMPILE_ERROR
    assert classify_verification(False, "ModuleNotFoundError: openpyxl").kind == VerificationKind.ENVIRONMENT_ERROR
    assert classify_verification(False, "command not found").kind == VerificationKind.COMMAND_ERROR


def test_actionable_task_detection() -> None:
    assert CodingAgent._should_plan([{"role": "user", "content": "你好"}]) is False
    assert CodingAgent._should_plan([{"role": "user", "content": "请修复登录模块并补充测试"}]) is True


def test_intent_router_answer_requests_are_read_only() -> None:
    examples = [
        "你读一下 src 里的代码，告诉我核心思想是什么",
        "为什么这个函数会返回 None？",
        "你看看目前这个 dense_only 方法还有什么可改进的点",
        "分析一下仓颉版本是否和 Python 版本对齐",
    ]

    for text in examples:
        assert IntentRouter.resolve(text).mode == TaskMode.ANSWER


def test_intent_router_plan_requests_are_plan_only() -> None:
    examples = [
        "这个模块应该怎么重构？",
        "给我一个修改计划，先不要改代码",
        "设计一下后续实现方案",
    ]

    for text in examples:
        assert IntentRouter.resolve(text).mode == TaskMode.PLAN


def test_intent_router_explicit_mutation_wins() -> None:
    examples = [
        "帮我分析一下这个 bug，然后修复它",
        "把 quicksort.cpp 改成归并排序",
        "实现分页功能并补充测试",
        "删除重复代码然后运行测试",
    ]

    for text in examples:
        assert IntentRouter.resolve(text).mode == TaskMode.MODIFY


def test_low_confidence_intent_requires_user_choice(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path), console=Mock())
    with patch("memcodeagent.agent.questionary.select") as select:
        select.return_value.ask.return_value = "直接修改并验证"
        intent = agent._resolve_intent("请处理一下这个问题")

    assert intent is not None
    assert intent.mode == TaskMode.MODIFY
    assert intent.confidence == "clarified"
    select.assert_called_once()


def test_single_turn_answer_mode_does_not_execute_tools(tmp_path: Path) -> None:
    console = Console(file=StringIO())
    agent = CodingAgent(AgentConfig(workspace=tmp_path, max_steps=2), console=console)
    agent.retriever.retrieve = Mock(return_value=RetrievalContext())
    agent.retriever.remember_task = Mock()
    decision = AgentDecision(
        content="这是只读分析结果。",
        assistant_message={"role": "assistant", "content": "这是只读分析结果。"},
    )

    with patch.object(agent.llm, "next_action", return_value=decision), patch.object(
        agent.tools, "execute"
    ) as execute:
        result = agent.run("为什么这个函数会返回 None？")

    assert result == "这是只读分析结果。"
    execute.assert_not_called()


def test_single_turn_plan_mode_rejects_edit_tools(tmp_path: Path) -> None:
    console = Console(file=StringIO())
    agent = CodingAgent(AgentConfig(workspace=tmp_path, max_steps=2), console=console)
    agent.retriever.retrieve = Mock(return_value=RetrievalContext())
    agent.retriever.remember_task = Mock()
    edit = Mock()
    edit.id = "edit-1"
    edit.name = "apply_patch"
    edit.args = {"path": "src/app.py", "edits": [{"old": "a", "new": "b"}]}
    decisions = [
        AgentDecision(
            tool_calls=[edit],
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "edit-1", "function": {"name": "apply_patch"}}],
            },
        ),
        AgentDecision(
            content="计划已生成。",
            assistant_message={"role": "assistant", "content": "计划已生成。"},
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions), patch.object(
        agent.tools, "execute"
    ) as execute:
        result = agent.run("给我一个修改计划，先不要改代码")

    assert result == "计划已生成。"
    execute.assert_not_called()


def test_session_state_defaults(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    assert agent._phase == "IDLE"
    assert agent._plan_text == ""


def test_existing_tests_are_protected(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_existing.py").write_text("def test_ok(): assert True")
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    agent._protected_test_files = agent._snapshot_test_files()

    assert agent._is_protected_test_edit("apply_patch", {"path": "tests/test_existing.py"}) is True
    assert agent._is_protected_test_edit("write_file", {"path": "tests/test_existing.py"}) is True
    assert agent._is_protected_test_edit("write_file", {"path": "tests/test_new.py"}) is False
