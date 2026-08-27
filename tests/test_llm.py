import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memcodeagent.llm import AgentDecision, LlmClient, ToolCall


def test_agent_decision_is_final_when_no_tool_calls() -> None:
    decision = AgentDecision(content="Done", tool_calls=[])
    assert decision.is_final


def test_agent_decision_is_not_final_when_tool_calls_present() -> None:
    decision = AgentDecision(
        tool_calls=[ToolCall(id="call_1", name="read_file", args={"path": "a.py"})]
    )
    assert not decision.is_final


def test_llm_client_returns_placeholder_when_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary directory with no .env file and clear all API keys
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MEMCODE_MODEL", raising=False)

    # Also ensure load_dotenv doesn't find the real .env
    with patch("memcodeagent.llm.load_dotenv"):
        client = LlmClient()
        decision = client.next_action([{"role": "user", "content": "Hello"}])
        assert decision.is_final
        assert "not configured" in decision.content


@patch("memcodeagent.llm.OpenAI")
def test_llm_client_parses_tool_calls_from_response(mock_openai_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Simulate OpenAI tool-calling response
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "read_file"
    mock_tool_call.function.arguments = json.dumps({"path": "test.py"})

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    client = LlmClient()
    client.api_key = "test-key"
    decision = client.next_action([{"role": "user", "content": "Read test.py"}])

    assert not decision.is_final
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].id == "call_abc123"
    assert decision.tool_calls[0].name == "read_file"
    assert decision.tool_calls[0].args == {"path": "test.py"}
    assert "tool_calls" in decision.assistant_message


@patch("memcodeagent.llm.OpenAI")
def test_llm_client_parses_final_answer_from_response(mock_openai_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_message = MagicMock()
    mock_message.content = "Task completed successfully"
    mock_message.tool_calls = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    client = LlmClient()
    client.api_key = "test-key"
    decision = client.next_action([{"role": "user", "content": "Finish"}])

    assert decision.is_final
    assert decision.content == "Task completed successfully"
    assert len(decision.tool_calls) == 0


@patch("memcodeagent.llm.OpenAI")
def test_llm_client_passes_tools_to_api(mock_openai_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_message = MagicMock()
    mock_message.content = "Done"
    mock_message.tool_calls = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    client = LlmClient()
    client.api_key = "test-key"
    client.next_action([{"role": "user", "content": "Test"}])

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["tools"] is not None
    assert len(call_args.kwargs["tools"]) == 6  # All 6 tools
    assert call_args.kwargs["tool_choice"] == "auto"


def test_tool_call_dataclass_structure() -> None:
    tc = ToolCall(id="call_1", name="write_file", args={"path": "test.py", "content": "hi"})
    assert tc.id == "call_1"
    assert tc.name == "write_file"
    assert tc.args["path"] == "test.py"
