from pathlib import Path

import pytest

from memcodeagent.tools import ToolExecutor
from memcodeagent.workspace import Workspace


def make_executor(tmp_path: Path) -> ToolExecutor:
    return ToolExecutor(Workspace(tmp_path))


def test_summarize_tree_returns_compact_file_list(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text("print('hi')", encoding="utf-8")
    observation = make_executor(tmp_path).execute("summarize_tree", {"max_files": 10})
    assert observation.ok
    assert "pkg/app.py" in observation.content


def test_diff_summary_handles_non_git_workspace(tmp_path: Path) -> None:
    observation = make_executor(tmp_path).execute("diff_summary", {})
    assert observation.ok
    assert "No git repository found" in observation.content


def test_write_file_creates_new_file(tmp_path: Path) -> None:
    executor = make_executor(tmp_path)
    observation = executor.execute("write_file", {"path": "hello.py", "content": "print('hi')\n"})
    assert observation.ok
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_write_file_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("old\n", encoding="utf-8")
    executor = make_executor(tmp_path)
    observation = executor.execute("write_file", {"path": "hello.py", "content": "new\n"})
    assert not observation.ok
    assert "already exists" in observation.content
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "old\n"


def test_write_file_overwrites_when_flagged(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("old\n", encoding="utf-8")
    executor = make_executor(tmp_path)
    observation = executor.execute(
        "write_file", {"path": "hello.py", "content": "new\n", "overwrite": True}
    )
    assert observation.ok
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "new\n"


def test_apply_patch_single_edit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    executor = make_executor(tmp_path)
    observation = executor.execute(
        "apply_patch",
        {"path": "a.py", "edits": [{"old": "x = 1", "new": "x = 100"}]},
    )
    assert observation.ok
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 100\ny = 2\n"
    assert "Patched" in observation.content


def test_apply_patch_multiple_edits_in_order(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    executor = make_executor(tmp_path)
    observation = executor.execute(
        "apply_patch",
        {
            "path": "a.py",
            "edits": [
                {"old": "x = 1", "new": "x = 100"},
                {"old": "y = 2", "new": "y = 200"},
            ],
        },
    )
    assert observation.ok
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 100\ny = 200\n"


def test_apply_patch_reports_missing_text_with_snippet(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    executor = make_executor(tmp_path)
    observation = executor.execute(
        "apply_patch",
        {"path": "a.py", "edits": [{"old": "z = 9", "new": "z = 10"}]},
    )
    assert not observation.ok
    assert "text not found" in observation.content
    assert "current file content" in observation.content


def test_apply_patch_rejects_ambiguous_match(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    executor = make_executor(tmp_path)
    observation = executor.execute(
        "apply_patch",
        {"path": "a.py", "edits": [{"old": "x = 1", "new": "x = 2"}]},
    )
    assert not observation.ok
    assert "matched 2 times" in observation.content


def test_execute_unknown_tool_returns_error(tmp_path: Path) -> None:
    executor = make_executor(tmp_path)
    observation = executor.execute("does_not_exist", {})
    assert not observation.ok
    assert "Unknown tool" in observation.content


def test_tool_observation_to_message_uses_tool_call_id(tmp_path: Path) -> None:
    executor = make_executor(tmp_path)
    observation = executor.execute(
        "write_file", {"path": "f.txt", "content": "hi"}, tool_call_id="call_123"
    )
    message = observation.to_message()
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_123"
