from dataclasses import dataclass
from pathlib import Path

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

            if decision.final_answer:
                self.retriever.remember_task(task, decision.final_answer)
                return decision.final_answer

            observation = self.tools.execute(decision.tool_name, decision.tool_args)
            self.console.print(observation.to_display())
            messages.append(decision.to_message())
            messages.append(observation.to_message())

        final = "Stopped because the maximum number of steps was reached."
        self.retriever.remember_task(task, final)
        return final

    def index_workspace(self) -> str:
        self.workspace.ensure_exists()
        return self.retriever.index_workspace()

    def _initial_messages(self, task: str, retrieval_context: RetrievalContext) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are MemCodeAgent, a CLI coding agent. Decide one local tool call at a time, "
                    "observe the result, and continue until the programming task is complete."
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
