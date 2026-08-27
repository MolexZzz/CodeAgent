"""Manual test script to verify per-error retry counter behavior."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from memcodeagent.agent import AgentConfig, CodingAgent


def test_multiple_errors_each_get_retries():
    """Verify that multiple independent errors each get their own retry budget."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create a tests directory so verification runs
        (workspace / "tests").mkdir()
        (workspace / "tests" / "test_dummy.py").write_text("def test_pass(): assert True")

        config = AgentConfig(
            workspace=workspace,
            max_error_retries=3,  # Only 3 retries per error for faster testing
            run_tests_after_edit=True
        )
        agent = CodingAgent(config, console=Mock())

        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Test task."},
        ]

        # Simulate: error 1 fails twice then succeeds, error 2 fails twice then succeeds
        tool_calls = []

        # Error 1: First failure
        tc1 = Mock()
        tc1.name = "run_command"
        tc1.args = {"command": "exit 1"}
        tc1.id = "call_1"
        tool_calls.append(tc1)

        # Error 1: Second failure
        tc2 = Mock()
        tc2.name = "run_command"
        tc2.args = {"command": "exit 1"}
        tc2.id = "call_2"
        tool_calls.append(tc2)

        # Error 1: Success (counter should reset)
        tc3 = Mock()
        tc3.name = "run_command"
        tc3.args = {"command": "echo success"}
        tc3.id = "call_3"
        tool_calls.append(tc3)

        # Error 2: First failure (new error, counter resets to 1)
        tc4 = Mock()
        tc4.name = "run_command"
        tc4.args = {"command": "exit 1"}
        tc4.id = "call_4"
        tool_calls.append(tc4)

        # Error 2: Success
        tc5 = Mock()
        tc5.name = "run_command"
        tc5.args = {"command": "echo success2"}
        tc5.id = "call_5"
        tool_calls.append(tc5)

        # Final response
        decisions = []
        for i, tc in enumerate(tool_calls):
            decisions.append(Mock(
                is_final=False,
                assistant_message={"role": "assistant", "content": None, "tool_calls": [{"id": tc.id}]},
                tool_calls=[tc],
            ))

        # Final decision
        decisions.append(Mock(
            is_final=True,
            content="Task completed successfully.",
            assistant_message={"role": "assistant", "content": "Task completed successfully."},
            tool_calls=[],
        ))

        from unittest.mock import patch
        with patch.object(agent.llm, "next_action", side_effect=decisions):
            result = agent._run_loop(messages, "Test task.")

        # Should complete successfully (not stopped after max_error_retries)
        assert result == "Task completed successfully."
        print("Test passed: Multiple errors each got their own retry budget")


def test_error_exhaustion():
    """Verify that exceeding max_error_retries stops the loop."""
    from memcodeagent.tools import ToolObservation

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        config = AgentConfig(
            workspace=workspace,
            max_error_retries=2,
            run_tests_after_edit=False
        )
        agent = CodingAgent(config, console=Mock())

        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Test task."},
        ]

        # Create 3 consecutive tool calls that will be mocked to fail
        tool_calls = []
        for i in range(3):
            tc = Mock()
            tc.name = "run_command"
            tc.args = {"command": "exit 1"}
            tc.id = f"call_{i}"
            tool_calls.append(tc)

        decisions = [
            Mock(
                is_final=False,
                assistant_message={"role": "assistant", "content": None, "tool_calls": [{"id": tc.id}]},
                tool_calls=[tc],
            )
            for tc in tool_calls
        ]

        from unittest.mock import patch

        # Mock the tool executor to return observation.ok=False for failures
        def mock_execute(tool_name, args, tool_call_id):
            return ToolObservation(tool_name, False, "Mocked tool failure", tool_call_id)

        call_count = 0
        def tracked_next_action(msgs):
            nonlocal call_count
            call_count += 1
            print(f"\n=== next_action call #{call_count} ===")
            print(f"Messages count: {len(msgs)}")
            if call_count > len(decisions):
                raise StopIteration(f"Ran out of mock decisions after {len(decisions)} calls")
            decision = decisions[call_count - 1]
            print(f"Returning: is_final={decision.is_final}, tool_calls={len(decision.tool_calls)}")
            return decision

        import os
        os.environ['DEBUG_RETRY'] = '1'

        with patch.object(agent.llm, "next_action", side_effect=tracked_next_action):
            with patch.object(agent.tools, "execute", side_effect=mock_execute):
                result = agent._run_loop(messages, "Test task.")
                print(f"\n=== Final result ===")
                print(f"Total next_action calls: {call_count}")
                print(f"Result: {result}")

        # Should stop after 2 retries with error message
        assert "failed attempts" in result.lower()
        assert call_count == 2, f"Expected 2 next_action calls but got {call_count}"
        print("Test passed: Loop stopped after max_error_retries exceeded")


if __name__ == "__main__":
    test_multiple_errors_each_get_retries()
    test_error_exhaustion()
    print("\nAll manual tests passed!")
