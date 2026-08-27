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
        """Single-turn execution: retrieve context, run agent loop, persist memory."""
        self.workspace.ensure_exists()
        retrieval_context = self.retriever.retrieve(task)
        self._print_context(retrieval_context)

        messages = self._initial_messages(task, retrieval_context)
        result = self._run_loop(messages, task)
        return result

    def chat(self) -> None:
        """Multi-turn interactive REPL: maintains conversation history across user inputs."""
        self.workspace.ensure_exists()
        self.console.print("[bold cyan]MemCodeAgent REPL[/bold cyan]")
        self.console.print("Commands: /exit, /clear, /history, /help")
        self.console.print()

        messages = self._initial_messages_chat()

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Exiting...[/yellow]")
                break

            if not user_input:
                continue

            if user_input == "/exit":
                break
            elif user_input == "/clear":
                messages = self._initial_messages_chat()
                self.console.print("[yellow]Conversation cleared.[/yellow]")
                continue
            elif user_input == "/history":
                self._print_history(messages)
                continue
            elif user_input == "/help":
                self.console.print("[cyan]Commands:[/cyan] /exit, /clear, /history, /help")
                continue

            # Regular user message: add to conversation and run agent loop.
            messages.append({"role": "user", "content": user_input})
            self._run_loop_interactive(messages)

    def _run_loop(self, messages: list[dict[str, Any]], task: str) -> str:
        """Core agent loop for single-turn mode: returns final answer or stops at max_steps."""
        for step in range(1, self.config.max_steps + 1):
            self.console.rule(f"[bold blue]Step {step}")
            decision = self.llm.next_action(messages)
            self.console.print(decision.to_display())

            messages.append(decision.assistant_message)

            if decision.is_final:
                self.retriever.remember_task(task, decision.content or "(empty response)")
                return decision.content or "(empty response)"

            # Execute all tool calls and collect observations.
            for tool_call in decision.tool_calls:
                observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                self.console.print(observation.to_display())
                messages.append(observation.to_message())
                self.retriever.record_tool_result(tool_call.name, observation.ok, tool_call.args, observation.content)

        final = "Stopped because the maximum number of steps was reached."
        self.retriever.remember_task(task, final)
        return final

    def _run_loop_interactive(self, messages: list[dict[str, Any]]) -> None:
        """Core agent loop for interactive mode: modifies messages in-place, no memory persistence per turn."""
        for step in range(1, self.config.max_steps + 1):
            self.console.rule(f"[bold blue]Step {step}")
            decision = self.llm.next_action(messages)
            self.console.print(decision.to_display())

            messages.append(decision.assistant_message)

            if decision.is_final:
                # In interactive mode, just print and return; memory is built across the full session.
                self.console.print(f"[green]{decision.content}[/green]")
                return

            # Execute all tool calls and collect observations.
            for tool_call in decision.tool_calls:
                observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                self.console.print(observation.to_display())
                messages.append(observation.to_message())

        self.console.print("[yellow]Stopped because the maximum number of steps was reached.[/yellow]")

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

    def _initial_messages_chat(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are MemCodeAgent, a CLI coding agent. Use the provided tools to complete "
                    "programming tasks: list_files, read_file, search_text, write_file, apply_patch, "
                    "run_command. You may call multiple tools in parallel. Observe each result and "
                    "continue until the task is complete, then respond with a final natural-language answer. "
                    "You are in an interactive chat session, so the user may ask follow-up questions or "
                    "refine their requests across multiple turns."
                ),
            },
        ]

    def _print_context(self, retrieval_context: RetrievalContext) -> None:
        self.console.rule("[bold cyan]Retrieved Context")
        self.console.print(retrieval_context.to_display())

    def _print_history(self, messages: list[dict[str, Any]]) -> None:
        self.console.rule("[bold cyan]Conversation History")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                self.console.print(f"[dim]SYSTEM: {content[:100]}...[/dim]")
            elif role == "user":
                self.console.print(f"[cyan]USER: {content}[/cyan]")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    self.console.print(f"[yellow]ASSISTANT: (called {len(tool_calls)} tool(s))[/yellow]")
                else:
                    self.console.print(f"[green]ASSISTANT: {content}[/green]")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "?")
                self.console.print(f"[dim]TOOL({tool_call_id}): {content[:80]}...[/dim]")
