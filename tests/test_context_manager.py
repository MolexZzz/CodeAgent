from memcodeagent.context_manager import ContextManager


def test_trim_keeps_system_and_recent_turns() -> None:
    cm = ContextManager(max_turns=2, max_tokens=1000000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn1"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "turn2"},
        {"role": "assistant", "content": "answer2"},
        {"role": "user", "content": "turn3"},
        {"role": "assistant", "content": "answer3"},
    ]
    trimmed = cm.trim(messages)
    assert len(trimmed) == 5
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["content"] == "turn2"
    assert trimmed[4]["content"] == "answer3"


def test_trim_does_not_split_tool_calls_from_results() -> None:
    cm = ContextManager(max_turns=1, max_tokens=1000000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "file content"},
        {"role": "user", "content": "turn2"},
        {"role": "assistant", "content": "answer2"},
    ]
    trimmed = cm.trim(messages)
    # Should keep system + turn2 only (turn1 + its tool_calls/tool are dropped as a unit)
    assert len(trimmed) == 3
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["content"] == "turn2"
    assert trimmed[2]["content"] == "answer2"
    # turn1's tool message should not be in trimmed
    assert not any(m.get("role") == "tool" for m in trimmed)


def test_trim_enforces_token_budget_by_dropping_oldest_turns() -> None:
    cm = ContextManager(max_turns=10, max_tokens=50)
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
    ]
    trimmed = cm.trim(messages)
    # Token budget forces dropping the oldest turn (turn1), keeps system + turn2
    assert len(trimmed) == 3
    assert trimmed[0]["role"] == "system"
    assert "c" in trimmed[1]["content"]
    assert "d" in trimmed[2]["content"]


def test_stats_reports_full_and_kept_counts() -> None:
    cm = ContextManager(max_turns=1, max_tokens=1000000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn1"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "turn2"},
        {"role": "assistant", "content": "answer2"},
    ]
    stats = cm.stats(messages)
    assert stats["total_messages"] == 5
    assert stats["total_turns"] == 2
    assert stats["kept_messages"] == 3  # system + turn2 + answer2
    assert stats["kept_turns"] == 1
    assert stats["estimated_tokens_full"] > stats["estimated_tokens_kept"]
