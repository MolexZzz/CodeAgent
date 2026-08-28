from __future__ import annotations

import json
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
from memcodeagent.tools import ToolExecutor, ToolObservation
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
    approval_required: bool = False
    max_continuations: int = 3


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
        self._session_path = self.workspace.root / ".memcode" / "session.json"
        self._stop_requested = False

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
        if self.config.approval_required:
            self.console.print("[dim]Read-only tools run automatically; file edits and commands require approval.[/dim]")
        self.console.print()

        messages = self._load_session() or self._initial_messages_chat()
        if len(messages) > 1:
            self.console.print("[dim]Resumed persisted session from .memcode/session.json[/dim]")

        # Setup autocomplete for slash commands
        slash_commands = [
            "/help", "/exit", "/quit", "/clear", "/history",
            "/model", "/models", "/verbose", "/workspace", "/context", "/tokens", "/streaming", "/plan", "/cache", "/save"
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
                self._save_session(messages)
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
            elif user_input == "/cache":
                self._print_cache_stats(messages)
                continue
            elif user_input == "/save":
                self._save_session(messages)
                self.console.print("[green]Session saved.[/green]")
                continue
            elif user_input.startswith("/plan"):
                request = user_input[5:].strip()
                if not request:
                    self.console.print("[yellow]用法：/plan <要规划的任务>。不会读取或修改文件。[/yellow]")
                else:
                    self._run_plan(request, messages)
                continue
            elif user_input.startswith("/"):
                self.console.print(f"[red]Unknown command: {user_input}[/red] (type /help for a list)")
                continue

            # Regular user message: add to conversation and run agent loop.
            messages.append({"role": "user", "content": user_input})
            self._save_session(messages)
            self._run_loop_interactive(messages)
            self._save_session(messages)

    def _run_loop(self, messages: list[dict[str, Any]], task: str) -> str:
        """Core agent loop for single-turn mode: uses per-error retry counter.

        When an error occurs (tool failure or test failure), the agent gets up to
        max_error_retries attempts to fix it. Once fixed successfully, the counter
        resets to 0 for the next error.
        """
        error_retry_count = 0
        last_error_detected = False
        step = 0

        while step < self.config.max_steps:
            step += 1
            self.console.rule(f"[bold blue]Step {step}")
            trimmed = self.context_manager.trim(messages)
            try:
                decision = self.llm.next_action(trimmed)
            except KeyboardInterrupt:
                final = "Interrupted by user. No further tool calls were executed."
                self.retriever.remember_task(task, final)
                return final
            self.console.print(decision.to_display())

            messages.append(decision.assistant_message)
            self._save_session(messages)

            if decision.is_final:
                self.retriever.remember_task(task, decision.content or "(empty response)")
                return decision.content or "(empty response)"

            # Execute all tool calls and collect observations.
            code_changed = False
            current_step_has_error = False

            for tool_call in decision.tool_calls:
                try:
                    observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                except KeyboardInterrupt:
                    final = "Interrupted by user during tool execution."
                    self.retriever.remember_task(task, final)
                    return final
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
        final = f"Stopped after reaching the maximum of {self.config.max_steps} agent steps."
        self.retriever.remember_task(task, final)
        return final

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
        if self._should_plan(messages) and not self._confirm_interactive_plan(messages):
            return

        error_retry_count = 0
        last_error_detected = False
        self._stop_requested = False
        continuation_count = 0
        seen_calls: set[tuple[str, str]] = set()
        phase = "INSPECTING"
        pending_edits = 0

        while True:
            for step in range(1, self.config.max_steps + 1):
                trimmed = self.context_manager.trim(messages)
                if self.context_manager.last_trim_notice:
                    self.console.print(f"[yellow]{self.context_manager.last_trim_notice}[/yellow]")

                if self.streaming_mode:
                    self.console.print("[green]Streaming:[/green]")

                    def _on_chunk(text: str) -> None:
                        print(text, end="", flush=True)

                    try:
                        decision = self.llm.next_action(trimmed, stream=True, on_chunk=_on_chunk)
                    except KeyboardInterrupt:
                        self.console.print("[yellow]Interrupted by user.[/yellow]")
                        return
                    if decision.content:
                        print()
                else:
                    with Status("[cyan]Thinking...[/cyan]", console=self.console):
                        try:
                            decision = self.llm.next_action(trimmed)
                        except KeyboardInterrupt:
                            self.console.print("[yellow]Interrupted by user.[/yellow]")
                            return

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
                    if pending_edits:
                        self._print_phase("TESTING", step, self.config.max_steps)
                        self._run_verification_tests(messages)
                        pending_edits = 0
                    if not self.streaming_mode:
                        self.console.print(f"[green]{decision.content}[/green]")
                    messages.append(decision.assistant_message)
                    return

                # Tool calls: show simplified or detailed view based on verbose flag
                tool_names = [tc.name for tc in decision.tool_calls]
                if any(name in _CODE_EDIT_TOOLS for name in tool_names):
                    phase = "IMPLEMENTING"
                elif pending_edits and "run_command" in tool_names:
                    phase = "TESTING"
                else:
                    phase = "INSPECTING"
                self._print_phase(phase, step, self.config.max_steps)
                if self.verbose:
                    self.console.rule(f"[bold blue]Step {step}")
                    self.console.print(decision.to_display())
                else:
                    tool_names = [tc.name for tc in decision.tool_calls]
                    self.console.print(f"[dim]→ calling {', '.join(tool_names)}...[/dim]")

                messages.append(decision.assistant_message)
                self._save_session(messages)

                code_changed = False
                current_step_has_error = False

                for tool_call in decision.tool_calls:
                    call_key = (tool_call.name, json.dumps(tool_call.args, sort_keys=True, ensure_ascii=False))
                    if call_key in seen_calls:
                        observation = ToolObservation(tool_call.name, False, "Duplicate tool call suppressed; use a different path, range, query, or command.", tool_call.id)
                    elif self._requires_approval(tool_call.name) and not self._confirm_tool(tool_call.name, tool_call.args):
                        seen_calls.add(call_key)
                        observation = ToolObservation(tool_call.name, False, "Tool call denied by user.", tool_call.id)
                    else:
                        seen_calls.add(call_key)
                        try:
                            observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                        except KeyboardInterrupt:
                            self.console.print("[yellow]Interrupted during tool execution.[/yellow]")
                            return
                    if self.verbose:
                        self.console.print(observation.to_display())
                    else:
                        status_icon = "✓" if observation.ok else "✗"
                        self.console.print(f"[dim]  {status_icon} {observation.tool_name}[/dim]")
                    messages.append(observation.to_message())
                    self._save_session(messages)

                    if not observation.ok:
                        current_step_has_error = True

                    if observation.ok and tool_call.name in _CODE_EDIT_TOOLS:
                        code_changed = True
                        phase = "IMPLEMENTING"
                        pending_edits += 1

                # Validate after a small batch of edits, or when the model has
                # moved on from editing to another kind of action. This avoids
                # running the full suite after every single patch while still
                # feeding failures back before the task drifts too far.
                should_verify = code_changed and (
                    pending_edits >= 2
                    or not any(name in _CODE_EDIT_TOOLS for name in tool_names)
                )
                if should_verify:
                    phase = "TESTING"
                    self._print_phase(phase, step, self.config.max_steps)
                    test_failed = self._run_verification_tests(messages)
                    pending_edits = 0
                    if test_failed:
                        current_step_has_error = True
                        phase = "FIXING"

                if current_step_has_error:
                    if last_error_detected:
                        error_retry_count += 1
                    else:
                        error_retry_count = 1
                        last_error_detected = True

                    if error_retry_count >= self.config.max_error_retries:
                        self.console.print(f"[yellow]Stopped after {error_retry_count} failed attempts to fix the error.[/yellow]")
                        return
                else:
                    if last_error_detected:
                        self.console.print(f"[green]✓ Error fixed after {error_retry_count} attempt(s). Counter reset.[/green]")
                        error_retry_count = 0
                        last_error_detected = False

            self.console.print(f"[yellow]Paused after {step} agent steps ({self.config.max_steps}-step budget reached).[/yellow]")
            if continuation_count >= self.config.max_continuations:
                self.console.print("[dim]Continuation limit reached. Task paused; start a new message to continue.[/dim]")
                return
            try:
                continue_work = questionary.confirm("Task may be incomplete. Continue with another step budget?", default=False).ask()
            except (KeyboardInterrupt, EOFError):
                continue_work = False
            if not continue_work:
                self.console.print("[dim]Task paused. No further tools will be executed.[/dim]")
                return
            continuation_count += 1
            self.console.print(f"[cyan]Continuing with budget {continuation_count}/{self.config.max_continuations}...[/cyan]")

    def _confirm_interactive_plan(self, messages: list[dict[str, Any]]) -> bool:
        """Force a tool-free planning turn before repository changes begin."""
        plan_instruction = {
            "role": "system",
            "content": (
                "You are in the planning phase for a repository task. Do not call tools. "
                "Return a concise plan with: (1) understood goal, (2) likely files/modules "
                "to inspect or change, (3) implementation order, (4) verification command, "
                "and (5) risks or compatibility concerns. Do not claim anything was inspected "
                "unless it is already present in the conversation."
            ),
        }
        try:
            try:
                decision = self.llm.next_action([plan_instruction, *messages], tools_enabled=False)
            except TypeError:
                decision = self.llm.next_action([plan_instruction, *messages])
        except KeyboardInterrupt:
            self.console.print("[yellow]Planning interrupted.[/yellow]")
            return False

        self.console.rule("[bold cyan]Plan")
        self.console.print(decision.content or "(No plan returned.)")
        self.console.print("[dim]No files or commands have been changed during planning.[/dim]")
        try:
            approved = questionary.confirm("开始按这个计划执行？", default=True).ask()
        except (KeyboardInterrupt, EOFError):
            approved = False
        if not approved:
            self.console.print("[dim]Plan accepted? No. Task remains unchanged.[/dim]")
            return False

        # Keep the approved plan in the conversation so later tool decisions
        # can follow the same outline without relying on hidden state.
        if decision.content:
            messages.append({"role": "assistant", "content": f"计划：\n{decision.content}"})
        messages.append({"role": "user", "content": "计划已确认。现在开始执行，按计划检查、修改并验证项目。"})
        self._save_session(messages)
        return True

    @staticmethod
    def _should_plan(messages: list[dict[str, Any]]) -> bool:
        """Use the approval gate for actionable tasks, not short chat replies."""
        user_messages = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        ]
        if not user_messages:
            return False
        latest = user_messages[-1].strip().lower()
        conversational = {
            "你好", "hello", "hi", "谢谢", "thanks", "谢谢你",
            "继续", "继续吧", "可以", "好的", "ok",
        }
        if latest in conversational:
            return False
        action_words = (
            "改", "修", "实现", "增加", "删除", "重构", "测试", "运行",
            "检查", "分析", "开发", "修改", "fix", "implement", "refactor",
            "test", "run", "debug", "add", "change",
        )
        return len(latest) >= 12 or any(word in latest for word in action_words)

    def _print_phase(self, phase: str, step: int, limit: int) -> None:
        labels = {
            "INSPECTING": "检查项目",
            "IMPLEMENTING": "实现修改",
            "TESTING": "运行验证",
            "FIXING": "修复失败",
            "COMPLETING": "整理结果",
        }
        self.console.print(
            f"[cyan]阶段：{labels.get(phase, phase)}[/cyan] "
            f"[dim]步骤 {step}/{limit}[/dim]"
        )

    def _requires_approval(self, tool_name: str) -> bool:
        """Require confirmation only for operations that can change state."""
        return self.config.approval_required and tool_name in {"write_file", "apply_patch", "run_command"}

    def _confirm_tool(self, name: str, args: dict[str, Any]) -> bool:
        try:
            answer = questionary.confirm(
                f"Approve {self._tool_summary(name, args)}?",
                default=False,
            ).ask()
            return bool(answer)
        except (KeyboardInterrupt, EOFError):
            self._stop_requested = True
            return False

    @staticmethod
    def _tool_summary(name: str, args: dict[str, Any]) -> str:
        """Keep approval prompts readable while retaining the actionable details."""
        if name == "write_file":
            path = args.get("path", "<unknown>")
            size = len(args.get("content", ""))
            mode = "overwrite" if args.get("overwrite") else "create"
            return f"write_file ({mode}) {path} [{size:,} chars]"
        if name == "apply_patch":
            path = args.get("path", "<unknown>")
            edits = args.get("edits") or []
            return f"apply_patch {path} [{len(edits)} edit(s)]"
        if name == "run_command":
            command = " ".join(str(args.get("command", "")).split())
            if len(command) > 160:
                command = command[:157] + "..."
            return f"run_command: {command}"
        return name

    def _run_plan(self, request: str, messages: list[dict[str, Any]]) -> None:
        plan_messages = [
            {"role": "system", "content": "You are in PLAN-ONLY mode. Analyze the request and provide an ordered implementation plan, risks, files to inspect, and verification steps. Do not call tools or modify files."},
            {"role": "user", "content": request},
        ]
        try:
            decision = self.llm.next_action(plan_messages, tools_enabled=False)
        except TypeError:
            decision = self.llm.next_action(plan_messages)
        self.console.rule("[bold cyan]Plan (no changes made)")
        self.console.print(decision.content or "(no plan returned)")

    def _save_session(self, messages: list[dict[str, Any]]) -> None:
        import json
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_path.write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_session(self) -> list[dict[str, Any]] | None:
        import json
        try:
            data = json.loads(self._session_path.read_text(encoding="utf-8"))
            messages = data.get("messages")
            return messages if isinstance(messages, list) and messages else None
        except (OSError, json.JSONDecodeError):
            return None

    def _print_cache_stats(self, messages: list[dict[str, Any]]) -> None:
        stats = self.context_manager.stats(messages)
        summary_state = "已生成（后续裁剪会复用）" if self.context_manager._summary_cache else "尚未触发（对话还没超出窗口）"
        session_state = "存在，可恢复" if self._session_path.exists() else "不存在"
        self.console.rule("[bold cyan]缓存与持久化状态")
        self.console.print(f"上下文摘要缓存：{summary_state}")
        self.console.print(f"会话历史文件：{session_state}（.memcode/session.json）")
        self.console.print(f"当前请求预计 token：{stats['estimated_tokens_kept']} / {self.context_manager.max_tokens}")

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
                    f"run_command. The workspace root is {self.workspace.root}. Tools already execute "
                    "inside this directory; never add `cd /workspace`, `/e/workspace`, or switch to "
                    "another project. Use relative paths. Before the first tool call, briefly state a "
                    "3-5 step plan in your assistant content. Inspect only files relevant to the task, "
                    "use line ranges for large files, avoid repeating an identical tool call, and "
                    "finish with verification and a concise summary."
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
                    f"run_command. The workspace root is {self.workspace.root}. Tools already execute "
                    "inside this directory; never add `cd /workspace`, `/e/workspace`, or switch to "
                    "another project. Use relative paths. Before the first tool call, briefly state a "
                    "3-5 step plan in your assistant content. Inspect only files relevant to the task, "
                    "use line ranges for large files, avoid repeating an identical tool call, and "
                    "finish with verification and a concise summary. "
                    "You are in an interactive chat session, so the user may ask follow-up questions or "
                    "refine their requests across multiple turns."
                ),
            },
        ]

    def _print_context(self, retrieval_context: RetrievalContext) -> None:
        self.console.rule("[bold cyan]Retrieved Context")
        self.console.print(retrieval_context.to_display())

    def _print_history(self, messages: list[dict[str, Any]]) -> None:
        self.console.rule("[bold cyan]Conversation Summary")
        self.console.print("[dim]只显示用户消息、最终回复和工具摘要；完整原始内容保存在 session.json。[/dim]")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = self._clean_display_text(msg.get("content", ""))
            if role == "system":
                continue
            if role == "user" and content.startswith("Automated test verification"):
                self.console.print(f"[dim]VERIFICATION: {content.splitlines()[0]}[/dim]")
            elif role == "user":
                self.console.print(f"[cyan]USER:[/cyan] {self._clip(content, 240)}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                    self.console.print(f"[yellow]TOOLS:[/yellow] {', '.join(names)}")
                else:
                    self.console.print(f"[green]ASSISTANT:[/green] {self._clip(content, 600)}")
            elif role == "tool":
                self.console.print(f"[dim]TOOL RESULT:[/dim] {self._clip(content, 180)}")

    @staticmethod
    def _clean_display_text(value: Any) -> str:
        text = value if isinstance(value, str) else str(value)
        return text.replace("\x00", "")

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

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
            ("/plan [task]", "Plan only; do not call tools or modify files"),
            ("/cache", "Show context compression and session persistence status"),
            ("/save", "Persist the current conversation immediately"),
        ]
        for cmd, desc in rows:
            self.console.print(f"  [cyan]{cmd:<16}[/cyan] {desc}")

    def _print_context_stats(self, messages: list[dict[str, Any]]) -> None:
        """Show how the sliding window is trimming the conversation history."""
        stats = self.context_manager.stats(messages)
        self.console.rule("[bold cyan]上下文状态")
        self.console.print(
            f"当前发送给模型：{stats['kept_turns']} / {stats['total_turns']} 轮对话 "
            f"（最多保留 {self.context_manager.max_turns} 轮）"
        )
        self.console.print(
            f"消息数：发送 {stats['kept_messages']} / 完整历史 {stats['total_messages']}"
        )
        self.console.print(
            f"本次请求预计 token：{stats['estimated_tokens_kept']} / 上限 {self.context_manager.max_tokens}"
        )
        if stats["kept_turns"] < stats["total_turns"]:
            dropped = stats["total_turns"] - stats["kept_turns"]
            self.console.print(f"[yellow]已暂时省略较早的 {dropped} 轮；完整历史仍保存在会话文件中。[/yellow]")

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
