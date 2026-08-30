from pathlib import Path
from io import StringIO
from unittest.mock import Mock, patch

from rich.console import Console

from memcodeagent.agent import AgentConfig
from memcodeagent.agent import CodingAgent
from memcodeagent.agent import IntentRouter
from memcodeagent.agent import TaskMode
from memcodeagent.controller import AgentController, _close_unanswered_tool_calls
from memcodeagent.completion import CompletionGuard, CompletionState
from memcodeagent.llm import AgentDecision, ToolCall
from memcodeagent.tools import ToolObservation
from memcodeagent.memory.hybrid_retriever import RetrievalContext
from memcodeagent.policy import PolicyAction, ToolPolicy
from memcodeagent.progress import ProgressMonitor, ProgressSnapshot
from memcodeagent.ui import AgentEvent, AgentEventKind, ToolEvent, ToolEventKind, format_agent_event
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


def test_controller_records_invalid_transition_errors() -> None:
    llm = type("FakeLlm", (), {"next_action": lambda self, messages: None})()
    controller = AgentController(llm=llm, tool_executor=object())
    controller.state_machine.phase = Phase.PLAN

    ok = controller.transition(RuntimeEvent.USER_APPROVED)

    assert ok is False
    assert controller.last_transition_error is not None


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


def test_controller_closes_all_tool_calls_when_execution_is_interrupted() -> None:
    call_1 = type("ToolCall", (), {"id": "call-1", "name": "first", "args": {}})()
    call_2 = type("ToolCall", (), {"id": "call-2", "name": "second", "args": {}})()
    decision = AgentDecision(
        tool_calls=[call_1, call_2],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1"},
                {"id": "call-2"},
            ],
        },
    )

    class InterruptingTools:
        def execute(self, name, args, tool_call_id):
            raise KeyboardInterrupt

    llm = type("FakeLlm", (), {"next_action": lambda self, messages: decision})()
    messages = [{"role": "user", "content": "continue"}]
    controller = AgentController(
        llm=llm,
        tool_executor=InterruptingTools(),
        max_steps=1,
    )

    result = controller.step(messages)

    assert result.interrupted is True
    assert [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ] == ["call-1", "call-2"]


def test_close_unanswered_tool_calls_repairs_existing_history() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1"}, {"id": "call-2"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]

    repaired = _close_unanswered_tool_calls(messages)

    assert repaired == 1
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call-2"


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
    ).action == PolicyAction.CONFIRM
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="apply_patch",
        approval_required=False,
    ).action == PolicyAction.ALLOW
    assert policy.evaluate(
        phase="VERIFYING",
        tool_name="apply_patch",
        approval_required=False,
    ).action == PolicyAction.ALLOW
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="write_file",
        approval_required=False,
        protected_test=True,
    ).action == PolicyAction.DENY
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="run_command",
        approval_required=True,
    ).action == PolicyAction.CONFIRM
    assert policy.evaluate(
        phase="INSPECTING",
        tool_name="read_file_range",
        approval_required=True,
    ).action == PolicyAction.ALLOW
    assert policy.evaluate(
        phase="INSPECTING",
        tool_name="summarize_symbols",
        approval_required=True,
    ).action == PolicyAction.ALLOW
    assert policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="write_file",
        approval_required=True,
    ).action == PolicyAction.CONFIRM
    assert "创建或覆盖" in policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="write_file",
        approval_required=True,
    ).reason
    assert "修改工作区文件" in policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="apply_patch",
        approval_required=True,
    ).reason


def test_tool_policy_auto_allows_safe_verification_commands() -> None:
    policy = ToolPolicy()
    decision = policy.evaluate(
        phase="TESTING",
        tool_name="run_command",
        approval_required=True,
        command="mvn test",
    )
    assert decision.action == PolicyAction.ALLOW


def test_tool_policy_denies_destructive_command_chains() -> None:
    policy = ToolPolicy()
    decision = policy.evaluate(
        phase="TESTING",
        tool_name="run_command",
        approval_required=True,
        command="mvn test && Remove-Item -Recurse -Force .",
    )
    assert decision.action == PolicyAction.DENY


def test_tool_policy_approved_scope_skips_normal_confirmation() -> None:
    policy = ToolPolicy()
    decision = policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="apply_patch",
        approval_required=True,
        approved=True,
    )
    assert decision.action == PolicyAction.ALLOW


def test_tool_policy_keeps_network_commands_confirmed_after_grant() -> None:
    policy = ToolPolicy()
    decision = policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="run_command",
        approval_required=True,
        command="curl https://example.com",
        approved=True,
    )
    assert decision.action == PolicyAction.CONFIRM


def test_filesystem_edit_batch_is_limited_per_task(tmp_path: Path) -> None:
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path, filesystem_edit_batch_size=4),
        console=Mock(),
    )
    agent._filesystem_edits_remaining = 0
    assert agent._filesystem_edit_approval_active("apply_patch") is False
    agent._filesystem_edits_remaining = 3
    assert agent._filesystem_edit_approval_active("apply_patch") is True
    assert agent._filesystem_edit_approval_active("apply_patch") is True
    assert agent._filesystem_edit_approval_active("apply_patch") is True
    assert agent._filesystem_edit_approval_active("apply_patch") is False


def test_filesystem_batch_preview_shows_compact_badge_and_hint(tmp_path: Path) -> None:
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path, filesystem_edit_batch_size=4),
        console=Mock(),
    )
    agent._current_filesystem_batch = [
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/java/C.java",
        "src/main/java/D.java",
    ]
    preview = agent._format_filesystem_batch_preview(
        "write_file",
        {"path": "src/main.java", "content": "class A {}"},
    )
    assert "+4" in preview
    assert "write_file" in preview

    details = agent._filesystem_batch_details(
        "write_file",
        {"path": "src/main.java", "content": "class A {}"},
    )
    assert details == [
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/java/C.java",
        "src/main/java/D.java",
    ]


def test_filesystem_batch_preview_omits_badge_for_single_file(tmp_path: Path) -> None:
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path, filesystem_edit_batch_size=4),
        console=Mock(),
    )
    agent._current_filesystem_batch = ["src/main/java/A.java"]
    preview = agent._format_filesystem_batch_preview(
        "write_file",
        {"path": "src/main.java", "content": "class A {}"},
    )
    assert "+1" not in preview
    assert "+4" not in preview


def test_filesystem_batch_hint_matches_actual_batch_size(tmp_path: Path) -> None:
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path, filesystem_edit_batch_size=4),
        console=Mock(),
    )
    agent._current_filesystem_batch = ["src/main/java/A.java"]
    assert agent._filesystem_batch_hint(
        "write_file",
        {"path": "src/main.java", "content": "class A {}"},
    ) == ""

    agent._current_filesystem_batch = [
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/java/C.java",
        "src/main/java/D.java",
    ]
    assert agent._filesystem_batch_hint(
        "write_file",
        {"path": "src/main.java", "content": "class A {}"},
    ) == "+4"


def test_tool_targets_are_concise() -> None:
    assert "src/app.py:10-20" in CodingAgent._tool_target(
        "read_file",
        {"path": "src/app.py", "start_line": 10, "end_line": 20},
    )
    target = CodingAgent._tool_target(
        "run_command",
        {"command": "python -m pytest tests/test_agent_workflow.py -q --tb=short"},
    )
    assert target.startswith(" [python -m pytest")
    assert len(target) < 120


def test_agent_event_formatting_is_concise() -> None:
    event = AgentEvent(AgentEventKind.PHASE, "阶段：检查项目", "步骤 1/4")
    assert format_agent_event(event) == "阶段：检查项目 — 步骤 1/4"


def test_history_defaults_to_user_summary_and_supports_raw_mode(tmp_path: Path) -> None:
    output = StringIO()
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path),
        console=Console(file=output),
    )
    messages = [
        {"role": "user", "content": "修复登录并运行测试"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "apply_patch",
                        "arguments": '{"path":"src/app.py","edits":[]}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "(error) patch failed"},
        {"role": "assistant", "content": "## 修复结果\n\n**仍有一个补丁需要修复。**"},
    ]
    agent._current_task = "修复登录并运行测试"

    agent._print_history(messages)
    summary_output = output.getvalue()
    assert "Current task" in summary_output
    assert "Recent actions" in summary_output
    assert "Unresolved issues" in summary_output
    assert "Raw Conversation History" not in summary_output

    output.seek(0)
    output.truncate(0)
    agent._print_history(messages, raw=True)
    raw_output = output.getvalue()
    assert "Raw Conversation History" in raw_output
    assert "TOOLS:" in raw_output
    assert "## 修复结果" not in raw_output
    assert "仍有一个补丁需要修复。" in raw_output


def test_final_answer_is_rendered_as_markdown(tmp_path: Path) -> None:
    output = StringIO()
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path),
        console=Console(file=output),
    )
    agent._run_loop = Mock(return_value="## Heading\n\n**important**")

    agent._run_loop_interactive([{"role": "user", "content": "explain"}])

    rendered = output.getvalue()
    assert "Heading" in rendered
    assert "important" in rendered
    assert "## Heading" not in rendered


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


def test_progress_monitor_allows_retry_after_an_intervening_tool() -> None:
    monitor = ProgressMonitor(duplicate_limit=1)

    assert monitor.record_tool("run_command", {"command": "mvn test"}) is None
    assert monitor.record_tool("apply_patch", {"path": "src/App.java"}) is None
    assert monitor.record_tool("run_command", {"command": "mvn test"}) is None


def test_progress_monitor_detects_no_progress() -> None:
    monitor = ProgressMonitor(no_progress_limit=2)
    snapshot = ProgressSnapshot()
    assert monitor.record_snapshot(snapshot) is None
    alert = monitor.record_snapshot(snapshot)
    assert alert is not None
    assert alert.kind == "no_progress"


def test_repeated_tool_call_warns_before_pausing() -> None:
    agent = CodingAgent(
        AgentConfig(workspace=Path.cwd(), max_steps=3, max_duplicate_attempts=2),
        console=Mock(),
    )
    agent.retriever.remember_task = Mock()
    agent.tools.execute = Mock(
        return_value=ToolObservation("run_command", True, "exit_code=0")
    )
    call_1 = ToolCall("call-1", "run_command", {"command": "mvn -q package -DskipTests"})
    call_2 = ToolCall("call-2", "run_command", {"command": "mvn -q package -DskipTests"})
    decisions = [
        AgentDecision(
            tool_calls=[call_1],
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1"}],
            },
        ),
        AgentDecision(
            tool_calls=[call_2],
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-2"}],
            },
        ),
        AgentDecision(
            content="done",
            assistant_message={"role": "assistant", "content": "done"},
        ),
    ]

    messages = [{"role": "user", "content": "build the project"}]
    with patch.object(agent.llm, "next_action", side_effect=decisions):
        result = agent._run_loop(messages, "build the project")

    assert result == "done"
    assert len(agent.tools.execute.call_args_list) == 2
    assert any(
        "intentional and approved" in str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )


def test_duplicate_policy_does_not_override_confirmation() -> None:
    decision = ToolPolicy().evaluate(
        phase="IMPLEMENTING",
        tool_name="run_command",
        approval_required=True,
        duplicate=True,
        command="mvn test",
    )
    assert decision.action == PolicyAction.CONFIRM


def test_controller_executes_duplicate_after_confirmation() -> None:
    call = ToolCall("call-1", "run_command", {"command": "mvn test"})
    decision = AgentDecision(
        tool_calls=[call],
        assistant_message={"role": "assistant", "content": None},
    )

    class FakeLlm:
        def next_action(self, _messages):
            return decision

    class FakeTools:
        def __init__(self):
            self.calls = 0

        def execute(self, *_args):
            self.calls += 1
            return ToolObservation("run_command", True, "ok", call.id)

    tools = FakeTools()
    controller = AgentController(
        llm=FakeLlm(),
        tool_executor=tools,
        tool_policy=ToolPolicy(),
        confirmation_callback=lambda *_args: True,
    )
    controller.progress_monitor.record_tool("run_command", {"command": "mvn test"})
    result = controller.step(
        [{"role": "user", "content": "run the test again"}],
        tool_context=lambda _call: {
            "phase": "IMPLEMENTING",
            "approval_required": True,
            "duplicate": True,
        },
    )

    assert result.observations[0].ok is True
    assert tools.calls == 1


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


def test_controller_keyboard_interrupt_sets_persisted_flag() -> None:
    class FakeTools:
        def execute(self, *_args):
            raise AssertionError("tool should not run")

    def raising_next_action(_messages):
        raise KeyboardInterrupt

    llm = type("FakeLlm", (), {"next_action": staticmethod(raising_next_action)})()
    controller = AgentController(llm=llm, tool_executor=FakeTools())
    controller.handle_user_request("ANSWER", [{"role": "user", "content": "inspect"}])
    result = controller.step([{"role": "user", "content": "inspect"}])

    assert result.interrupted is True
    assert controller.interrupted is True
    assert controller.persist_state()["interrupted"] is True


def test_verification_results_are_classified() -> None:
    assert classify_verification(True, "exit_code=0").kind == VerificationKind.PASS
    assert classify_verification(False, "exit_code=1\nAssertionError: bad").kind == VerificationKind.ASSERTION_FAILURE
    assert classify_verification(False, "exit_code=1\nSyntaxError: invalid syntax").kind == VerificationKind.COMPILE_ERROR
    assert classify_verification(False, "ModuleNotFoundError: openpyxl").kind == VerificationKind.ENVIRONMENT_ERROR
    assert classify_verification(False, "command not found").kind == VerificationKind.COMMAND_ERROR


def test_completion_guard_blocks_unresolved_errors() -> None:
    assert CompletionGuard.can_finish(
        TaskMode.MODIFY,
        CompletionState(
            diff_checked=True,
            verification_done=True,
            verification_passed=True,
            unresolved_errors=True,
        ),
    ) is False


def test_completion_guard_requires_diff_and_verification() -> None:
    assert CompletionGuard.can_finish(
        TaskMode.MODIFY,
        CompletionState(diff_checked=False, verification_done=True, verification_passed=True),
    ) is False
    assert CompletionGuard.can_finish(
        TaskMode.MODIFY,
        CompletionState(diff_checked=True, verification_done=False, verification_passed=True),
    ) is False
    assert CompletionGuard.can_finish(
        TaskMode.MODIFY,
        CompletionState(diff_checked=True, verification_done=True, verification_passed=False),
    ) is False


def test_verification_environment_error_blocks_completion() -> None:
    assert classify_verification(False, "ModuleNotFoundError: openpyxl").kind == VerificationKind.ENVIRONMENT_ERROR


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


def test_low_confidence_intent_is_auto_resolved(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path), console=Mock())
    intent = agent._resolve_intent("请处理一下这个问题")

    assert intent is not None
    assert intent.mode in {TaskMode.ANSWER, TaskMode.PLAN}


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


def test_explicit_plan_mode_rejects_edit_tools(tmp_path: Path) -> None:
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
        result = agent._run_read_only_interactive(
            [{"role": "user", "content": "给我一个修改计划，先不要改代码"}],
            TaskMode.PLAN,
            "给我一个修改计划，先不要改代码",
            display_final=False,
        )

    assert result == "计划已生成。"
    execute.assert_not_called()


def test_read_only_mode_can_execute_safe_shell_command(tmp_path: Path) -> None:
    console = Console(file=StringIO())
    agent = CodingAgent(AgentConfig(workspace=tmp_path, max_steps=2), console=console)
    agent.retriever.retrieve = Mock(return_value=RetrievalContext())
    agent.retriever.remember_task = Mock()

    class FakeObservation:
        ok = True
        tool_name = "run_command"
        content = "exit_code=0\nSTDOUT:\n1\nSTDERR:\n"

        def to_message(self):
            return {"role": "tool", "content": self.content}

    run_call = Mock()
    run_call.id = "run-1"
    run_call.name = "run_command"
    run_call.args = {"command": "python -c \"print(1)\""}

    decisions = [
        AgentDecision(
            tool_calls=[run_call],
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "run-1", "function": {"name": "run_command"}}],
            },
        ),
        AgentDecision(
            content="当前时间应由系统命令返回。",
            assistant_message={"role": "assistant", "content": "当前时间应由系统命令返回。"},
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions), patch.object(
        agent.tools, "execute"
    ) as execute:
        execute.return_value = FakeObservation()
        result = agent._run_read_only_interactive(
            [{"role": "user", "content": "现在几点了"}],
            TaskMode.ANSWER,
            "现在几点了",
            display_final=False,
        )

    assert result == "当前时间应由系统命令返回。"
    execute.assert_any_call("run_command", {"command": "python -c \"print(1)\""}, "run-1")


def test_session_state_defaults(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    assert agent._phase == "IDLE"
    assert agent._plan_text == ""


def test_session_state_restores_runtime(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    agent.controller.handle_user_request("MODIFY", [{"role": "user", "content": "fix"}])
    agent.controller.step_count = 3
    agent.controller.tool_calls.append(("read_file", {"path": "a.py"}))
    agent._verification_done = True
    agent._verification_passed = True
    agent._last_verification_kind = VerificationKind.PASS
    agent._test_attempts = 2
    messages = [{"role": "user", "content": "hello"}]
    agent._save_session(messages)

    restored = agent._load_session()

    assert restored is not None
    assert agent.controller.step_count == 3
    assert agent.controller.tool_calls == [("read_file", {"path": "a.py"})]
    assert agent._verification_done is True
    assert agent._verification_passed is True
    assert agent._last_verification_kind == VerificationKind.PASS
    assert agent._test_attempts == 2


def test_diff_summary_is_cached(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path), console=Mock())
    agent.tools.execute = Mock(
        return_value=Mock(ok=True, content="diff --git a/x b/x", tool_name="diff_summary")
    )
    messages = [{"role": "user", "content": "hello"}]
    agent._print_diff_summary(messages)

    assert agent._last_diff_summary == "diff --git a/x b/x"


def test_existing_tests_are_protected(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_existing.py").write_text("def test_ok(): assert True")
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    agent._protected_test_files = agent._snapshot_test_files()

    assert agent._is_protected_test_edit("apply_patch", {"path": "tests/test_existing.py"}) is True
    assert agent._is_protected_test_edit("write_file", {"path": "tests/test_existing.py"}) is True
    assert agent._is_protected_test_edit("write_file", {"path": "tests/test_new.py"}) is False


def test_tool_policy_classifies_dangerous_commands() -> None:
    policy = ToolPolicy()
    assert policy.command_risk("rm -rf build")[0] == "destructive"
    assert policy.command_risk("pip install -r requirements.txt")[0] == "environment"
    assert policy.command_risk("git reset --hard HEAD")[0] == "destructive"
    decision = policy.evaluate(
        phase="IMPLEMENTING",
        tool_name="run_command",
        approval_required=True,
        command="rm -rf build",
    )
    assert decision.action == PolicyAction.CONFIRM
    assert decision.risk == "destructive"
    assert "删除" in decision.reason
    maven_decision = policy.evaluate(
        phase="TESTING",
        tool_name="run_command",
        approval_required=True,
        command="mvn test",
    )
    assert "Maven 测试" in maven_decision.reason


def test_controller_emits_separate_tool_events() -> None:
    events = []

    class FakeTools:
        def execute(self, name, args, call_id):
            return type("Obs", (), {
                "tool_name": name, "ok": True, "content": "ok",
                "to_message": lambda self: {"role": "tool", "content": "ok"},
            })()

    call = type("Call", (), {"id": "1", "name": "read_file", "args": {"path": "a.py"}})()
    decision = AgentDecision(
        tool_calls=[call],
        assistant_message={"role": "assistant", "content": None, "tool_calls": []},
    )
    llm = type("Llm", (), {"next_action": staticmethod(lambda _messages: decision)})()
    controller = AgentController(
        llm=llm, tool_executor=FakeTools(), event_callback=events.append
    )
    controller.handle_user_request("ANSWER", [{"role": "user", "content": "inspect"}])
    controller.step([{"role": "user", "content": "inspect"}])
    assert [event.kind for event in events] == [ToolEventKind.CALL, ToolEventKind.RESULT]
    assert all(isinstance(event, ToolEvent) for event in events)


def test_progress_monitor_detects_repeated_final_answer() -> None:
    monitor = ProgressMonitor()
    assert monitor.record_final_answer("完成了。") is None
    alert = monitor.record_final_answer("  完成了。 ")
    assert alert is not None
    assert alert.kind == "final_repetition"


def test_progress_monitor_detects_phase_monotony() -> None:
    monitor = ProgressMonitor(phase_monotony_limit=3)
    assert monitor.record_phase("INSPECTING") is None
    assert monitor.record_phase("FIXING") is None
    alert = monitor.record_phase("INSPECTING")
    assert alert is not None
    assert alert.kind == "phase_monotony"


def test_tool_executor_truncates_large_read_output(tmp_path: Path) -> None:
    from memcodeagent.tools import ToolExecutor
    from memcodeagent.workspace import Workspace

    big = "x" * 40000
    (tmp_path / "large.txt").write_text(big, encoding="utf-8")
    executor = ToolExecutor(Workspace(tmp_path), dry_run=False, max_read_bytes=2048)
    obs = executor.execute("read_file", {"path": "large.txt"}, None)
    assert obs.ok is True
    assert len(obs.content.encode("utf-8")) < len(big.encode("utf-8"))
    assert "truncated" in obs.content.lower()


def test_completion_report_includes_verification_details(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path), console=Mock())
    agent._last_diff_summary = "diff --git a/app.py b/app.py"
    agent._last_verification_command = "python -m pytest -q"
    agent._last_verification_kind = VerificationKind.PASS
    report = agent._build_completion_report()
    assert "diff --git a/app.py b/app.py" in report
    assert "python -m pytest -q" in report
    assert "pass" in report.lower()


def test_controller_lifecycle_methods_use_guarded_transitions() -> None:
    controller = AgentController(llm=Mock(), tool_executor=Mock())
    controller.handle_user_request("MODIFY", [{"role": "user", "content": "fix"}])
    assert controller.mark_implementation_done() is False
    assert controller.last_transition_error
    controller.state_machine.phase = Phase.IMPLEMENT
    assert controller.mark_implementation_done() is True
    assert controller.phase == Phase.DIFF_CHECK
    assert controller.mark_diff_checked() is True
    assert controller.phase == Phase.TEST
    assert controller.mark_test_result(True) is True
    assert controller.phase == Phase.VERIFY


def test_controller_persists_plan_text() -> None:
    controller = AgentController(llm=Mock(), tool_executor=Mock())
    controller.set_plan("先读源码，再修复，再测")
    state = controller.persist_state()
    assert state["plan_text"] == "先读源码，再修复，再测"
    restored = AgentController(llm=Mock(), tool_executor=Mock())
    restored.restore_state(state)
    assert restored.plan_text == "先读源码，再修复，再测"


def test_session_restore_keeps_plan_text(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    agent.controller.set_plan("先探索，再修改")
    agent._plan_text = "先探索，再修改"
    agent._save_session([{"role": "user", "content": "fix"}])
    restored = agent._load_session()
    assert restored is not None
    assert agent._plan_text == "先探索，再修改"
