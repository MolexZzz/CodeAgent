from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from memcodeagent.llm import LlmClient
from memcodeagent.memory.retriever import RetrievalContext, SimpleRetriever
from memcodeagent.tools import ToolExecutor
from memcodeagent.workspace import Workspace


@dataclass(slots=True)
class AgentConfig:
    workspace: Path
    max_steps: int = 8
    dry_run: bool = False


class CodingAgent:
    """Coordinates retrieval, LLM decisions, local tools, and observations."""

    def __init__(self, config: AgentConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.workspace = Workspace(config.workspace)
        self.llm = LlmClient()
        self.tools = ToolExecutor(self.workspace, dry_run=config.dry_run)
        self.retriever = SimpleRetriever(self.workspace)

    def run(self, task: str) -> str:
        self.workspace.ensure_exists()
        retrieval_context = self.retriever.retrieve(task)
        self._print_context(retrieval_context)

        messages = self._initial_messages(task, retrieval_context)
        for step in range(1, self.config.max_steps + 1):
            self.console.rule(f"[bold blue]Step {step}")
            decision = self.llm.next_action(messages)
            self.console.print(decision.to_display())

            # Append the assistant's message (content + tool_calls if any) to conversation history.
            messages.append(decision.assistant_message)

            if decision.is_final:
                self.retriever.remember_task(task, decision.content or "(empty response)")
                return decision.content or "(empty response)"

            # Execute all tool calls (parallel if multiple) and collect observations.
            for tool_call in decision.tool_calls:
                observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                self.console.print(observation.to_display())
                messages.append(observation.to_message())
                # Track what changed so it can be persisted when the task completes.
                self.retriever.record_tool_result(tool_call.name, observation.ok, tool_call.args, observation.content)

        final = "Stopped because the maximum number of steps was reached."
        self.retriever.remember_task(task, final)
        return final

    def index_workspace(self) -> str:
        self.workspace.ensure_exists()
        return self.retriever.index_workspace()

    def _initial_messages(self, task: str, retrieval_context: RetrievalContext) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are MemCodeAgent, a CLI coding agent. Use the provided tools to complete "
                    "programming tasks: list_files, read_file, search_text, write_file, apply_patch, "
                    "run_command. You may call multiple tools in parallel. Observe each result and "
                    "continue until the task is complete, then respond with a final natural-language answer."
                ),
            },
            {
                "role": "user",
                "content": f"Task:\n{task}\n\nRetrieved context:\n{retrieval_context.to_prompt()}",
            },
        ]

    def _print_context(self, retrieval_context: RetrievalContext) -> None:
        self.console.rule("[bold cyan]Retrieved Context")
        self.console.print(retrieval_context.to_display())
