from pathlib import Path

from memcodeagent.memory_manager import MemoryManager
from memcodeagent.memory_manager import TranscriptStore
from memcodeagent.memory.hybrid_retriever import RetrievalContext
from memcodeagent.memory.schema import MemoryItem
from memcodeagent.workspace import Workspace


def test_memory_manager_persists_session_and_context_state(tmp_path: Path) -> None:
    manager = MemoryManager(Workspace(tmp_path))
    manager.context_manager._summary_cache = "older task summary"
    manager.context_manager._summary_dropped_count = 3

    manager.save_session({"messages": [{"role": "user", "content": "continue"}]})

    restored = MemoryManager(Workspace(tmp_path))
    payload = restored.load_session()

    assert payload is not None
    assert payload["messages"][0]["content"] == "continue"
    assert restored.context_manager._summary_cache == "older task summary"
    assert restored.context_manager._summary_dropped_count == 3


def test_memory_manager_keeps_retriever_compatibility_alias(tmp_path: Path) -> None:
    manager = MemoryManager(Workspace(tmp_path))

    assert manager.retriever is manager.code_retriever
    assert manager.context_manager.max_turns == 20


def test_memory_manager_composes_code_and_task_memory(tmp_path: Path) -> None:
    manager = MemoryManager(Workspace(tmp_path))
    code_item = MemoryItem("code_function", "code result")
    task_item = MemoryItem("task_history", "task result")
    manager.code_retriever.retrieve_code = lambda _query: [code_item]
    manager.task_memory.retrieve = lambda _query: [task_item]

    context = manager.retrieve("authentication")

    assert isinstance(context, RetrievalContext)
    assert context.items == [code_item, task_item]


def test_clear_working_memory_does_not_delete_task_memory(tmp_path: Path) -> None:
    manager = MemoryManager(Workspace(tmp_path))
    manager.task_memory._save_records([])
    manager.working_memory.current_task = "old task"
    manager.working_memory.phase = "PAUSED"
    manager.working_memory.history_summary = "old summary"

    manager.reset_working_memory()

    assert manager.working_memory.current_task == ""
    assert manager.working_memory.phase == "IDLE"
    assert manager.working_memory.history_summary == ""
    assert manager.task_memory.path.name == "memory.json"


def test_task_record_legacy_shape_is_read_with_defaults(tmp_path: Path) -> None:
    manager = MemoryManager(Workspace(tmp_path))
    manager.task_memory.path.parent.mkdir(parents=True, exist_ok=True)
    manager.task_memory.path.write_text(
        '{"records": [{"task": "old", "summary": "done"}]}',
        encoding="utf-8",
    )

    items = manager.task_memory.retrieve("old")

    assert len(items) == 1
    assert items[0].kind == "task_history"


def test_transcript_store_appends_only_new_messages(tmp_path: Path) -> None:
    store = TranscriptStore(Workspace(tmp_path))
    first = {"messages": [{"role": "user", "content": "one"}]}
    second = {
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
    }

    store.save(first)
    store.save(second)

    lines = store.transcript_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"content": "two"' in lines[-1]


def test_transcript_store_handles_reset_snapshot_without_duplicate_count(
    tmp_path: Path,
) -> None:
    store = TranscriptStore(Workspace(tmp_path))
    store.save({"messages": [{"role": "user", "content": "old"}]})
    store.save({"messages": []})
    store.save({"messages": [{"role": "user", "content": "new"}]})

    lines = store.transcript_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"content": "new"' in lines[-1]


def test_project_memory_is_injected_into_context(tmp_path: Path) -> None:
    manager = MemoryManager(Workspace(tmp_path))
    manager.project_memory.save({"rules": ["Run tests with rtk pytest -q."]})

    trimmed = manager.trim_context([{"role": "user", "content": "check tests"}])

    assert any(
        message.get("role") == "system"
        and "Run tests with rtk pytest -q." in message.get("content", "")
        for message in trimmed
    )
