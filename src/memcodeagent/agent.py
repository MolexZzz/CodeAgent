from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from memcodeagent.context_manager import ContextManager
from memcodeagent.llm import LlmClient
from memcodeagent.memory.retriever import RetrievalContext, SimpleRetriever
from memcodeagent.tools import ToolExecutor
from memcodeagent.workspace import Workspace


@dataclass(slots=True)
class AgentConfig:
    workspace: Path
    max_steps: int = 8
    dry_run: bool = False
    max_context_turns: int = 20
    max_context_tokens: int = 24000


class CodingAgent:
    """Coordinates retrieval, LLM decisions, local tools, and observations."""

    def __init__(self, config: AgentConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.workspace = Workspace(config.workspace)
        self.llm = LlmClient()
        self.tools = ToolExecutor(self.workspace, dry_run=config.dry_run)
        self.retriever = SimpleRetriever(self.workspace)
        self.context_manager = ContextManager(
            max_turns=config.max_context_turns,
            max_tokens=config.max_context_tokens,
        )
        self.verbose = False

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
        self.console.print(f"model={self.llm.model}  workspace={self.workspace.root}")
        self.console.print("Type /help to see available commands.")
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

            if user_input in {"/exit", "/quit"}:
                break
            elif user_input == "/clear":
                messages = self._initial_messages_chat()
                self.console.print("[yellow]Conversation cleared.[/yellow]")
                continue
            elif user_input == "/history":
                self._print_history(messages)
                continue
            elif user_input == "/help":
                self._print_help()
                continue
            elif user_input.startswith("/model"):
                self._handle_model_command(user_input)
                continue
            elif user_input.startswith("/verbose"):
                self._handle_verbose_command(user_input)
                continue
            elif user_input == "/workspace":
                self.console.print(f"[cyan]Workspace:[/cyan] {self.workspace.root}")
                continue
            elif user_input == "/context":
                self._print_context_stats(messages)
                continue
            elif user_input.startswith("/"):
                self.console.print(f"[red]Unknown command: {user_input}[/red] (type /help for a list)")
                continue

            # Regular user message: add to conversation and run agent loop.
            messages.append({"role": "user", "content": user_input})
            self._run_loop_interactive(messages)

    def _run_loop(self, messages: list[dict[str, Any]], task: str) -> str:
        """Core agent loop for single-turn mode: returns final answer or stops at max_steps."""
        for step in range(1, self.config.max_steps + 1):
            self.console.rule(f"[bold blue]Step {step}")
            trimmed = self.context_manager.trim(messages)
            decision = self.llm.next_action(trimmed)
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
            trimmed = self.context_manager.trim(messages)
            decision = self.llm.next_action(trimmed)

            if decision.is_final:
                # In interactive mode, just print and return; memory is built across the full session.
                self.console.print(f"[green]{decision.content}[/green]")
                messages.append(decision.assistant_message)
                return

            # Tool calls: show simplified or detailed view based on verbose flag
            if self.verbose:
                self.console.rule(f"[bold blue]Step {step}")
                self.console.print(decision.to_display())
            else:
                tool_names = [tc.name for tc in decision.tool_calls]
                self.console.print(f"[dim]→ calling {', '.join(tool_names)}...[/dim]")

            messages.append(decision.assistant_message)

            # Execute all tool calls and collect observations.
            for tool_call in decision.tool_calls:
                observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                if self.verbose:
                    self.console.print(observation.to_display())
                else:
                    status_icon = "✓" if observation.ok else "✗"
                    self.console.print(f"[dim]  {status_icon} {observation.tool_name}[/dim]")
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

    def _print_help(self) -> None:
        """Print all available REPL slash commands."""
        self.console.print("[bold cyan]Available commands:[/bold cyan]")
        rows = [
            ("/help", "Show this help message"),
            ("/exit, /quit", "Exit the REPL"),
            ("/clear", "Clear conversation history and start fresh"),
            ("/history", "Show the full conversation history"),
            ("/model", "Show the current model"),
            ("/model <name>", "Switch to a different model, e.g. /model deepseek-chat"),
            ("/verbose", "Toggle verbose mode (show full step/tool details)"),
            ("/workspace", "Show the current workspace root path"),
            ("/context", "Show sliding-window context stats (turns/tokens kept vs total)"),
        ]
        for cmd, desc in rows:
            self.console.print(f"  [cyan]{cmd:<16}[/cyan] {desc}")

    def _print_context_stats(self, messages: list[dict[str, Any]]) -> None:
        """Show how the sliding window is trimming the conversation history."""
        stats = self.context_manager.stats(messages)
        self.console.rule("[bold cyan]Context Window (sliding window)")
        self.console.print(
            f"turns: {stats['kept_turns']} kept / {stats['total_turns']} total  "
            f"(max_turns={self.context_manager.max_turns})"
        )
        self.console.print(
            f"messages sent to model: {stats['kept_messages']} / {stats['total_messages']} in full history"
        )
        self.console.print(
            f"estimated tokens sent: ~{stats['estimated_tokens_kept']} / "
            f"~{stats['estimated_tokens_full']} full (max_tokens={self.context_manager.max_tokens})"
        )
        if stats["kept_turns"] < stats["total_turns"]:
            dropped = stats["total_turns"] - stats["kept_turns"]
            self.console.print(f"[yellow]{dropped} oldest turn(s) dropped from model context (still kept in /history)[/yellow]")

    def _handle_model_command(self, user_input: str) -> None:
        """Handle `/model` (show current) and `/model <name>` (switch model)."""
        parts = user_input.split(maxsplit=1)
        if len(parts) == 1:
            self.console.print(f"[cyan]Current model:[/cyan] {self.llm.model}")
            return
        new_model = parts[1].strip()
        if not new_model:
            self.console.print(f"[cyan]Current model:[/cyan] {self.llm.model}")
            return
        old_model = self.llm.model
        self.llm.model = new_model
        self.console.print(f"[green]Model switched:[/green] {old_model} -> {new_model}")

    def _handle_verbose_command(self, user_input: str) -> None:
        """Handle `/verbose` (toggle) and `/verbose on|off` (explicit set)."""
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            arg = parts[1].strip().lower()
            if arg in {"on", "true", "1"}:
                self.verbose = True
            elif arg in {"off", "false", "0"}:
                self.verbose = False
            else:
                self.console.print(f"[red]Unknown argument: {arg}[/red] (use on/off)")
                return
        else:
            self.verbose = not self.verbose
        state = "on" if self.verbose else "off"
        self.console.print(f"[cyan]Verbose mode:[/cyan] {state}")
