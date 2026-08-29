from pathlib import Path
from io import StringIO
from unittest.mock import Mock, patch

from rich.console import Console

from memcodeagent.agent import AgentConfig, CodingAgent
from memcodeagent.llm import AgentDecision
from memcodeagent.memory.hybrid_retriever import RetrievalContext
from memcodeagent.policy import PolicyAction, ToolPolicy


def test_single_turn_long_request_enters_react_loop(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path, max_steps=1), console=Mock())
    agent.retriever.retrieve = Mock(return_value=RetrievalContext())
    agent.retriever.remember_task = Mock()
    agent._run_loop = Mock(return_value="完成")

    with patch.object(agent, "_run_read_only_interactive") as readonly:
        result = agent.run(
            "请分析当前项目的实现问题，修改相关代码，补充必要测试并运行完整验证"
        )

    assert result == "完成"
    agent._run_loop.assert_called_once()
    readonly.assert_not_called()


def test_interactive_ordinary_request_skips_forced_plan(tmp_path: Path) -> None:
    agent = CodingAgent(
        AgentConfig(workspace=tmp_path, max_steps=1),
        console=Console(file=StringIO()),
    )
    agent.controller.step = Mock(side_effect=RuntimeError("stop after first step"))
    agent._save_session = Mock()

    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "先看看这个问题，如果需要就直接修改代码并运行测试",
        },
    ]

    with patch.object(agent, "_prepare_task") as prepare:
        agent._run_loop_interactive(messages)

    prepare.assert_not_called()


def test_inspecting_phase_does_not_block_a_safe_edit() -> None:
    decision = ToolPolicy().evaluate(
        phase="INSPECTING",
        tool_name="apply_patch",
        approval_required=False,
    )

    assert decision.action == PolicyAction.ALLOW


def test_read_only_mode_does_not_execute_shell_commands(tmp_path: Path) -> None:
    console = Console(file=StringIO())
    agent = CodingAgent(AgentConfig(workspace=tmp_path, max_steps=1), console=console)
    agent.tools.execute = Mock()

    call = Mock()
    call.id = "run-1"
    call.name = "run_command"
    call.args = {"command": "echo unsafe"}
    decision = AgentDecision(
        tool_calls=[call],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "run-1", "function": {"name": "run_command"}}],
        },
    )

    with patch.object(agent.llm, "next_action", return_value=decision):
        agent._run_read_only_interactive(
            [{"role": "user", "content": "explain"}],
            TaskMode.ANSWER,
            "explain",
            display_final=False,
        )

    agent.tools.execute.assert_not_called()
