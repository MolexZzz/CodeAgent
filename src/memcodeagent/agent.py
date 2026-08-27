from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import questionary
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.status import Status

from memcodeagent.context_manager import ContextManager
from memcodeagent.llm import LlmClient
from memcodeagent.memory.hybrid_retriever import HybridRetriever, RetrievalContext
from memcodeagent.tools import ToolExecutor
from memcodeagent.workspace import Workspace


@dataclass(slots=True)
class AgentConfig:
    workspace: Path
    max_steps: int = 8  # Deprecated: use max_error_retries instead
    max_error_retries: int = 10  # Maximum retry attempts per error before giving up
    dry_run: bool = False
    max_context_turns: int = 20
    max_context_tokens: int = 24000
    max_tool_retries: int = 2
    run_tests_after_edit: bool = True
    test_command: str | None = None  # None = auto-detect (pytest if a tests/ dir exists)


# Tools that modify code on disk and should trigger a verification test run.
_CODE_EDIT_TOOLS = {"write_file", "apply_patch"}


class CodingAgent:
    """Coordinates retrieval, LLM decisions, local tools, and observations."""

    def __init__(self, config: AgentConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.workspace = Workspace(config.workspace)
        self.llm = LlmClient()
        self.tools = ToolExecutor(self.workspace, dry_run=config.dry_run, max_tool_retries=config.max_tool_retries)
        self.retriever = HybridRetriever(self.workspace)
        self.context_manager = ContextManager(
            max_turns=config.max_context_turns,
            max_tokens=config.max_context_tokens,
            enable_summarization=True,
            llm_client=self.llm,
        )
        self.verbose = False
        self.streaming_mode = False
        # Session-level token tracking
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0

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

        # Setup autocomplete for slash commands
        slash_commands = [
            "/help", "/exit", "/quit", "/clear", "/history",
            "/model", "/models", "/verbose", "/workspace", "/context", "/tokens", "/streaming"
        ]
        completer = WordCompleter(slash_commands, ignore_case=True, sentence=True)
        session = PromptSession(completer=completer)

        while True:
            try:
                user_input = session.prompt(">>> ").strip()
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
            elif user_input == "/models":
                self._handle_models_command()
                continue
            elif user_input.startswith("/model"):
                self._handle_model_command(user_input)
                continue
            elif user_input.startswith("/verbose"):
                self._handle_verbose_command(user_input)
                continue
            elif user_input.startswith("/streaming"):
                self._handle_streaming_command(user_input)
                continue
            elif user_input == "/workspace":
                self.console.print(f"[cyan]Workspace:[/cyan] {self.workspace.root}")
                continue
            elif user_input == "/context":
                self._print_context_stats(messages)
                continue
            elif user_input == "/tokens":
                self._print_token_stats()
                continue
            elif user_input.startswith("/"):
                self.console.print(f"[red]Unknown command: {user_input}[/red] (type /help for a list)")
                continue

            # Regular user message: add to conversation and run agent loop.
            messages.append({"role": "user", "content": user_input})
            self._run_loop_interactive(messages)

    def _run_loop(self, messages: list[dict[str, Any]], task: str) -> str:
        """Core agent loop for single-turn mode: uses per-error retry counter.

        When an error occurs (tool failure or test failure), the agent gets up to
        max_error_retries attempts to fix it. Once fixed successfully, the counter
        resets to 0 for the next error.
        """
        error_retry_count = 0
        last_error_detected = False
        step = 0

        while True:
            step += 1
            self.console.rule(f"[bold blue]Step {step}")
            trimmed = self.context_manager.trim(messages)
            decision = self.llm.next_action(trimmed)
            self.console.print(decision.to_display())

            messages.append(decision.assistant_message)

            if decision.is_final:
                self.retriever.remember_task(task, decision.content or "(empty response)")
                return decision.content or "(empty response)"

            # Execute all tool calls and collect observations.
            code_changed = False
            current_step_has_error = False

            for tool_call in decision.tool_calls:
                observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                self.console.print(observation.to_display())
                messages.append(observation.to_message())
                self.retriever.record_tool_result(tool_call.name, observation.ok, tool_call.args, observation.content)

                if not observation.ok:
                    current_step_has_error = True

                if observation.ok and tool_call.name in _CODE_EDIT_TOOLS:
                    code_changed = True

            # Run verification tests after code changes
            if code_changed:
                test_failed = self._run_verification_tests(messages)
                if test_failed:
                    current_step_has_error = True

            # Update error retry counter based on current step outcome
            if current_step_has_error:
                if last_error_detected:
                    # Still in error state, increment retry count
                    error_retry_count += 1
                else:
                    # New error detected, reset counter to 1
                    error_retry_count = 1
                    last_error_detected = True

                # Debug output for testing
                import os
                if os.environ.get('DEBUG_RETRY'):
                    print(f"[DEBUG] Step {step}: error detected, retry_count={error_retry_count}, max={self.config.max_error_retries}")

                if error_retry_count >= self.config.max_error_retries:
                    final = f"Stopped after {error_retry_count} failed attempts to fix the error."
                    self.retriever.remember_task(task, final)
                    return final
            else:
                # No error in this step - reset counter if we were in error state
                if last_error_detected:
                    self.console.print(f"[green]✓ Error fixed after {error_retry_count} attempt(s). Counter reset.[/green]")
                    error_retry_count = 0
                    last_error_detected = False

    def _resolve_test_command(self) -> str | None:
        """Pick the test command to run after a code edit, or None if tests are disabled/absent."""
        if not self.config.run_tests_after_edit:
            return None
        if self.config.test_command:
            return self.config.test_command
        if (self.workspace.root / "tests").is_dir():
            return "python -m pytest tests/ -q --tb=short"
        return None

    def _run_verification_tests(self, messages: list[dict[str, Any]]) -> bool:
        """Run the project's test suite after a code-modifying tool call and feed the
        result back into the conversation as an observation, so the model can react to
        regressions in the same step loop (the automated test-verification part of the
        retry loop).

        Returns True if tests failed, False if tests passed or were skipped.
        """
        command = self._resolve_test_command()
        if not command:
            return False
        self.console.rule("[bold magenta]Verification tests")
        observation = self.tools.execute("run_command", {"command": command}, tool_call_id=None)
        self.console.print(observation.to_display())
        passed = observation.ok and "exit_code=0" in observation.content
        status = "PASSED" if passed else "FAILED"
        summary = (
            f"Automated test verification after code edit: {status}.\n"
            f"{observation.content}"
        )
        messages.append({"role": "user", "content": summary})
        return not passed  # Return True if tests failed

    def _run_loop_interactive(self, messages: list[dict[str, Any]]) -> None:
        """Core agent loop for interactive mode: uses per-error retry counter."""
        error_retry_count = 0
        last_error_detected = False
        step = 0

        while True:
            step += 1
            trimmed = self.context_manager.trim(messages)

            if self.streaming_mode:
                # Stream tokens to the terminal as they arrive instead of blocking silently.
                self.console.print("[green]Streaming:[/green]")

                def _on_chunk(text: str) -> None:
                    print(text, end="", flush=True)

                decision = self.llm.next_action(trimmed, stream=True, on_chunk=_on_chunk)
                if decision.content:
                    print()  # newline after streamed content
            else:
                # Show spinner during LLM call
                with Status("[cyan]Thinking...[/cyan]", console=self.console):
                    decision = self.llm.next_action(trimmed)

            # Update session token counters
            self.session_prompt_tokens += decision.usage.prompt_tokens
            self.session_completion_tokens += decision.usage.completion_tokens
            self.session_total_tokens += decision.usage.total_tokens

            # Display token usage
            self.console.print(
                f"[dim]Tokens: {decision.usage.total_tokens:,} "
                f"(prompt: {decision.usage.prompt_tokens:,}, "
                f"completion: {decision.usage.completion_tokens:,}) | "
                f"Session total: {self.session_total_tokens:,}[/dim]"
            )

            if decision.is_final:
                # In interactive mode, just print and return; memory is built across the full session.
                if not self.streaming_mode:
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
            code_changed = False
            current_step_has_error = False

            for tool_call in decision.tool_calls:
                observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                if self.verbose:
                    self.console.print(observation.to_display())
                else:
                    status_icon = "✓" if observation.ok else "✗"
                    self.console.print(f"[dim]  {status_icon} {observation.tool_name}[/dim]")
                messages.append(observation.to_message())

                if not observation.ok:
                    current_step_has_error = True

                if observation.ok and tool_call.name in _CODE_EDIT_TOOLS:
                    code_changed = True

            # Run verification tests after code changes
            if code_changed:
                test_failed = self._run_verification_tests(messages)
                if test_failed:
                    current_step_has_error = True

            # Update error retry counter based on current step outcome
            if current_step_has_error:
                if last_error_detected:
                    # Still in error state, increment retry count
                    error_retry_count += 1
                else:
                    # New error detected, reset counter to 1
                    error_retry_count = 1
                    last_error_detected = True

                if error_retry_count >= self.config.max_error_retries:
                    self.console.print(f"[yellow]Stopped after {error_retry_count} failed attempts to fix the error.[/yellow]")
                    return
            else:
                # No error in this step - reset counter if we were in error state
                if last_error_detected:
                    self.console.print(f"[green]✓ Error fixed after {error_retry_count} attempt(s). Counter reset.[/green]")
                    error_retry_count = 0
                    last_error_detected = False

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
            ("/model", "Show current model and open an interactive menu to switch"),
            ("/model <name>", "Switch to a different model, e.g. /model deepseek-chat"),
            ("/models", "List preset models available for selection"),
            ("/verbose", "Toggle verbose mode (show full step/tool details)"),
            ("/streaming", "Toggle streaming mode (show model thinking in real-time)"),
            ("/workspace", "Show the current workspace root path"),
            ("/context", "Show sliding-window context stats (turns/tokens kept vs total)"),
            ("/tokens", "Show session-level token usage statistics"),
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

    def _handle_models_command(self) -> None:
        """Handle `/models`: list all known models grouped by provider, with credential status."""
        self.console.print("[bold cyan]Available models:[/bold cyan]")
        by_provider: dict[str, list[str]] = {}
        for name, config in self.llm.MODEL_REGISTRY.items():
            by_provider.setdefault(config["provider"], []).append(name)

        for provider, names in by_provider.items():
            configured = self.llm._model_is_configured(names[0])
            status = "[green]configured[/green]" if configured else "[red]no credentials[/red]"
            self.console.print(f"[bold]{provider}[/bold] ({status})")
            for name in names:
                marker = "[green]*[/green]" if name == self.llm.model else " "
                self.console.print(f"  {marker} {name}")

        self.console.print("[dim]Use /model to open an interactive picker, or /model <name> to switch directly.[/dim]")

    def _handle_model_command(self, user_input: str) -> None:
        """Handle `/model` (interactive picker) and `/model <name>` (direct switch)."""
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            new_model = parts[1].strip()
            old_model = self.llm.model
            try:
                self.llm.set_model(new_model, persist=True)
            except ValueError as exc:
                self.console.print(f"[red]{exc}[/red]")
                return
            self.console.print(f"[green]Model switched:[/green] {old_model} -> {new_model} [dim](saved as default)[/dim]")
            return

        # No argument: show an interactive menu with arrow-key navigation.
        choices = list(self.llm.AVAILABLE_MODELS)
        if self.llm.model not in choices:
            choices.append(self.llm.model)

        if not choices:
            self.console.print(
                "[red]No models are configured yet.[/red] Set at least one provider's API key "
                "(e.g. DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY) in your .env file."
            )
            return

        self.console.print(f"[cyan]Current model:[/cyan] {self.llm.model}")
        try:
            selected = questionary.select(
                "Select a model (this will be saved as default):",
                choices=choices,
                default=self.llm.model if self.llm.model in choices else None,
            ).ask()
        except Exception:
            selected = None

        if not selected or selected == self.llm.model:
            self.console.print("[dim]Model unchanged.[/dim]")
            return

        old_model = self.llm.model
        try:
            self.llm.set_model(selected, persist=True)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        self.console.print(f"[green]Model switched:[/green] {old_model} -> {selected} [dim](saved as default)[/dim]")

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

    def _print_token_stats(self) -> None:
        """Display session-level token usage statistics."""
        self.console.rule("[bold cyan]Session Token Usage")
        self.console.print(f"Prompt tokens:     {self.session_prompt_tokens:,}")
        self.console.print(f"Completion tokens: {self.session_completion_tokens:,}")
        self.console.print(f"Total tokens:      {self.session_total_tokens:,}")

    def _handle_streaming_command(self, user_input: str) -> None:
        """Handle `/streaming` (toggle) and `/streaming on|off` (explicit set)."""
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            arg = parts[1].strip().lower()
            if arg in {"on", "true", "1"}:
                self.streaming_mode = True
            elif arg in {"off", "false", "0"}:
                self.streaming_mode = False
            else:
                self.console.print(f"[red]Unknown argument: {arg}[/red] (use on/off)")
                return
        else:
            self.streaming_mode = not self.streaming_mode
        state = "on" if self.streaming_mode else "off"
        self.console.print(f"[cyan]Streaming mode:[/cyan] {state}")
