"""Tests for tool retry and automated test verification (Solution 3)."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from memcodeagent.agent import AgentConfig, CodingAgent
from memcodeagent.llm import ToolCall
from memcodeagent.runtime import Phase
from memcodeagent.tools import ToolExecutor
from memcodeagent.workspace import Workspace


def test_tool_retry_on_transient_failure(tmp_path: Path) -> None:
    """run_command retries on failure up to max_tool_retries times."""
    workspace = Workspace(tmp_path)
    executor = ToolExecutor(workspace, dry_run=False, max_tool_retries=2)

    call_count = 0

    def flaky_command(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Transient error")
        return "exit_code=0\nSTDOUT:\nSuccess\nSTDERR:\n"

    with patch.object(executor, "_tool_run_command", flaky_command):
        obs = executor.execute("run_command", {"command": "echo test"}, None)
        assert obs.ok
        assert call_count == 3
        assert "Success" in obs.content


def test_tool_retry_exhausted(tmp_path: Path) -> None:
    """After max retries, the tool returns an error observation."""
    workspace = Workspace(tmp_path)
    executor = ToolExecutor(workspace, dry_run=False, max_tool_retries=2)

    def always_fails(**kwargs):
        raise RuntimeError("Persistent error")

    with patch.object(executor, "_tool_run_command", always_fails):
        obs = executor.execute("run_command", {"command": "echo test"}, None)
        assert not obs.ok
        assert "failed after 3 attempt(s)" in obs.content
        assert "Persistent error" in obs.content


def test_run_command_executes_with_platform_default_shell() -> None:
    executor = ToolExecutor(Workspace(Path.cwd()), dry_run=False, max_tool_retries=0)

    observation = executor.execute("run_command", {"command": "echo test"}, None)

    assert observation.ok
    assert "exit_code=0" in observation.content
    assert "test" in observation.content.lower()


def test_deterministic_tools_do_not_retry(tmp_path: Path) -> None:
    """write_file and apply_patch should not auto-retry (they're deterministic)."""
    workspace = Workspace(tmp_path)
    executor = ToolExecutor(workspace, dry_run=False, max_tool_retries=2)

    call_count = 0

    def counting_write(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Bad path")

    with patch.object(executor, "_tool_write_file", counting_write):
        obs = executor.execute("write_file", {"path": "test.py", "content": "x=1"}, None)
        assert not obs.ok
        assert call_count == 1  # Only 1 attempt, no retry
        assert "failed after" not in obs.content


def test_verification_tests_run_after_code_edit(tmp_path: Path) -> None:
    """After write_file or apply_patch, the agent runs verification tests automatically."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_dummy.py").write_text("def test_pass(): assert True")

    config = AgentConfig(workspace=tmp_path, max_steps=2, run_tests_after_edit=True)
    agent = CodingAgent(config, console=Mock())

    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Write a simple file."},
    ]

    # Mock LLM to return a write_file tool call, then a final answer
    tool_call_mock = Mock()
    tool_call_mock.name = "write_file"
    tool_call_mock.args = {"path": "hello.py", "content": "print('hi')"}
    tool_call_mock.id = "1"

    decisions = [
        Mock(
            is_final=False,
            assistant_message={"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "write_file"}}]},
            tool_calls=[tool_call_mock],
        ),
        Mock(
            is_final=True,
            content="Done.",
            assistant_message={"role": "assistant", "content": "Done."},
            tool_calls=[],
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions):
        result = agent._run_loop(messages, "Write a simple file.")
        assert result == "Done."

    # Check that the messages now contain a test verification observation
    test_msg = [m for m in messages if m.get("content") and "Automated test verification" in m["content"]]
    assert len(test_msg) == 1
    assert "PASSED" in test_msg[0]["content"] or "FAILED" in test_msg[0]["content"]


def test_verification_disabled_when_no_tests_dir(tmp_path: Path) -> None:
    """If tests/ doesn't exist and test_command is None, no verification runs."""
    config = AgentConfig(workspace=tmp_path, max_steps=2, run_tests_after_edit=True, test_command=None)
    agent = CodingAgent(config, console=Mock())

    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Write a simple file."},
    ]

    tool_call_mock = Mock()
    tool_call_mock.name = "write_file"
    tool_call_mock.args = {"path": "hello.py", "content": "print('hi')"}
    tool_call_mock.id = "1"

    decisions = [
        Mock(
            is_final=False,
            assistant_message={"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "write_file"}}]},
            tool_calls=[tool_call_mock],
        ),
        Mock(
            is_final=True,
            content="Done.",
            assistant_message={"role": "assistant", "content": "Done."},
            tool_calls=[],
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions):
        result = agent._run_loop(messages, "Write a simple file.")
        assert result == "Done."

    # No test verification message should appear
    test_msg = [m for m in messages if m.get("content") and "Automated test verification" in m["content"]]
    assert len(test_msg) == 0


def test_verification_respects_run_tests_after_edit_flag(tmp_path: Path) -> None:
    """If run_tests_after_edit=False, no verification runs even if tests/ exists."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_dummy.py").write_text("def test_pass(): assert True")

    config = AgentConfig(workspace=tmp_path, max_steps=2, run_tests_after_edit=False)
    agent = CodingAgent(config, console=Mock())

    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Write a simple file."},
    ]

    tool_call_mock = Mock()
    tool_call_mock.name = "write_file"
    tool_call_mock.args = {"path": "hello.py", "content": "print('hi')"}
    tool_call_mock.id = "1"

    decisions = [
        Mock(
            is_final=False,
            assistant_message={"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "write_file"}}]},
            tool_calls=[tool_call_mock],
        ),
        Mock(
            is_final=True,
            content="Done.",
            assistant_message={"role": "assistant", "content": "Done."},
            tool_calls=[],
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions):
        result = agent._run_loop(messages, "Write a simple file.")
        assert result == "Done."

    test_msg = [m for m in messages if m.get("content") and "Automated test verification" in m["content"]]
    assert len(test_msg) == 0


def test_runtime_recovers_from_failed_tests_then_finishes(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_add.py").write_text(
        "from app import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    (tmp_path / "app.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n"
    )

    config = AgentConfig(
        workspace=tmp_path,
        max_steps=4,
        run_tests_after_edit=True,
        approval_required=False,
    )
    agent = CodingAgent(config, console=Mock())
    agent.retriever.retrieve = Mock(return_value=Mock())
    agent.retriever.remember_task = Mock()

    bad_patch = ToolCall(
        id="1",
        name="apply_patch",
        args={
            "path": "app.py",
            "edits": [{"old": "return a - b", "new": "return a + b + 1"}],
        },
    )
    fix_patch = ToolCall(
        id="2",
        name="apply_patch",
        args={
            "path": "app.py",
            "edits": [{"old": "return a + b + 1", "new": "return a + b"}],
        },
    )
    decisions = [
        Mock(
            is_final=False,
            assistant_message={"role": "assistant", "content": None},
            tool_calls=[bad_patch],
        ),
        Mock(
            is_final=False,
            assistant_message={"role": "assistant", "content": None},
            tool_calls=[fix_patch],
        ),
        Mock(
            is_final=True,
            content="修复完成。",
            assistant_message={"role": "assistant", "content": "修复完成。"},
            tool_calls=[],
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions):
        result = agent._run_loop([
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the add function and verify it."},
        ], "Fix the add function and verify it.")

    assert result == "修复完成。"
    assert agent._verification_done is True
    assert agent._verification_passed is True
    assert agent.controller.phase == Phase.DONE


def test_runtime_blocks_on_verification_environment_error(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_add.py").write_text(
        "from app import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    (tmp_path / "app.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n"
    )

    config = AgentConfig(
        workspace=tmp_path,
        max_steps=2,
        run_tests_after_edit=True,
        approval_required=False,
        test_command='python -c "import definitely_missing_pkg_zz"',
    )
    agent = CodingAgent(config, console=Mock())
    agent.retriever.retrieve = Mock(return_value=Mock())
    agent.retriever.remember_task = Mock()

    patch_call = ToolCall(
        id="1",
        name="apply_patch",
        args={
            "path": "app.py",
            "edits": [{"old": "return a - b", "new": "return a + b"}],
        },
    )
    decisions = [
        Mock(
            is_final=False,
            assistant_message={"role": "assistant", "content": None},
            tool_calls=[patch_call],
        ),
        Mock(
            is_final=True,
            content="我已经完成。",
            assistant_message={"role": "assistant", "content": "我已经完成。"},
            tool_calls=[],
        ),
    ]

    with patch.object(agent.llm, "next_action", side_effect=decisions):
        result = agent._run_loop([
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the add function and verify it."},
        ], "Fix the add function and verify it.")

    assert "测试环境或命令错误" in result
    assert agent._verification_unresolved_error is True
    assert agent.controller.phase == Phase.PAUSED
