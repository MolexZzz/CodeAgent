from pathlib import Path

from memcodeagent.agent import AgentConfig
from memcodeagent.agent import CodingAgent
from memcodeagent.agent import IntentRouter
from memcodeagent.agent import TaskMode


def test_phase_tool_permissions() -> None:
    read = {"list_files", "read_file", "search_text", "summarize_tree", "diff_summary"}
    assert all(CodingAgent._tool_allowed_in_phase("EXPLORING", name, False) for name in read)
    assert not CodingAgent._tool_allowed_in_phase("EXPLORING", "apply_patch", False)
    assert CodingAgent._tool_allowed_in_phase("IMPLEMENTING", "apply_patch", True)
    assert CodingAgent._tool_allowed_in_phase("TESTING", "run_command", True)
    assert CodingAgent._tool_allowed_in_phase("VERIFYING", "diff_summary", True)
    assert not CodingAgent._tool_allowed_in_phase("COMPLETED", "read_file", True)


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
