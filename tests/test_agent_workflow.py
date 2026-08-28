from pathlib import Path

from memcodeagent.agent import CodingAgent
from memcodeagent.agent import AgentConfig


def test_phase_tool_permissions() -> None:
    read = {"list_files", "read_file", "search_text"}
    assert all(CodingAgent._tool_allowed_in_phase("EXPLORING", name, False) for name in read)
    assert not CodingAgent._tool_allowed_in_phase("EXPLORING", "apply_patch", False)
    assert CodingAgent._tool_allowed_in_phase("IMPLEMENTING", "apply_patch", True)
    assert CodingAgent._tool_allowed_in_phase("TESTING", "run_command", True)
    assert not CodingAgent._tool_allowed_in_phase("COMPLETED", "read_file", True)


def test_actionable_task_detection() -> None:
    assert CodingAgent._should_plan([{"role": "user", "content": "你好"}]) is False
    assert CodingAgent._should_plan([{"role": "user", "content": "请修复登录模块并补充测试"}]) is True


def test_session_state_defaults(tmp_path: Path) -> None:
    agent = CodingAgent(AgentConfig(workspace=tmp_path))
    assert agent._phase == "IDLE"
    assert agent._plan_text == ""
