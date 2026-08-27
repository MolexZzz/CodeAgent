from pathlib import Path

import pytest

from memcodeagent.memory.retriever import SimpleRetriever
from memcodeagent.memory.schema import TaskRecord
from memcodeagent.workspace import Workspace


def test_remember_task_persists_to_json(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.remember_task("Create a file", "File created successfully")
    store_path = tmp_path / ".memcode" / "memory.json"
    assert store_path.exists()


def test_retrieve_returns_empty_when_no_history(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    context = retriever.retrieve("Create a Python file")
    assert len(context.items) == 0


def test_retrieve_matches_by_keyword_overlap(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.remember_task("Create hello.py", "Created Python file")
    retriever.remember_task("Delete backup files", "Removed old backups")

    context = retriever.retrieve("Make a new Python script")
    assert len(context.items) >= 1
    assert "hello.py" in context.items[0].text


def test_retrieve_scores_by_token_overlap(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.remember_task("Fix authentication bug", "Fixed login error")
    retriever.remember_task("Refactor database queries", "Improved performance")

    context = retriever.retrieve("Debug authentication issue")
    assert len(context.items) >= 1
    # Higher overlap with "authentication" should rank first
    assert context.items[0].score > 0


def test_retrieve_limits_to_max_results(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    for i in range(10):
        retriever.remember_task(f"Task {i} with python code", f"Outcome {i}")

    context = retriever.retrieve("python")
    # _MAX_RETRIEVED is 5
    assert len(context.items) <= 5


def test_record_tool_result_tracks_changed_files(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.record_tool_result("write_file", True, {"path": "test.py"}, "Created test.py")
    retriever.record_tool_result("apply_patch", True, {"path": "main.py"}, "Patched main.py")
    retriever.remember_task("Modify files", "Done")

    context = retriever.retrieve("test.py")
    assert len(context.items) == 1
    assert "test.py" in context.items[0].text
    assert "main.py" in context.items[0].text


def test_record_tool_result_tracks_failed_commands(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.record_tool_result("run_command", False, {"command": "pytest"}, "Error")
    retriever.remember_task("Run tests", "Tests failed")

    context = retriever.retrieve("pytest")
    assert len(context.items) == 1
    assert "pytest" in context.items[0].text
    assert "Failed commands" in context.items[0].text


def test_record_tool_result_tracks_nonzero_exit(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.record_tool_result(
        "run_command", True, {"command": "npm test"}, "exit_code=1\nSTDOUT:\nFailed"
    )
    retriever.remember_task("Run npm test", "Tests failed")

    context = retriever.retrieve("npm test")
    assert len(context.items) == 1
    assert "npm test" in context.items[0].text


def test_task_record_serialization_roundtrip() -> None:
    record = TaskRecord(
        task="Test task",
        summary="Test summary",
        changed_files=["a.py", "b.py"],
        failed_commands=["pytest"],
        timestamp="2026-08-27T10:00:00+00:00",
    )
    data = record.to_dict()
    restored = TaskRecord.from_dict(data)
    assert restored.task == record.task
    assert restored.summary == record.summary
    assert restored.changed_files == record.changed_files
    assert restored.failed_commands == record.failed_commands
    assert restored.timestamp == record.timestamp


def test_retrieval_context_to_prompt_formats_items(tmp_path: Path) -> None:
    retriever = SimpleRetriever(Workspace(tmp_path))
    retriever.remember_task("Create file", "File created")
    context = retriever.retrieve("file")
    prompt = context.to_prompt()
    assert "[task_history]" in prompt
    assert "Previous task" in prompt
