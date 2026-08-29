from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import questionary
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.status import Status

from memcodeagent.context_manager import ContextManager
from memcodeagent.completion import CompletionGuard, CompletionState
from memcodeagent.controller import AgentController
from memcodeagent.llm import LlmClient
from memcodeagent.memory.hybrid_retriever import HybridRetriever, RetrievalContext
from memcodeagent.policy import ToolPolicy
from memcodeagent.runtime import RuntimeEvent
from memcodeagent.verification import VerificationKind, classify_verification
from memcodeagent.tools import ToolExecutor, ToolObservation
from memcodeagent.ui import AgentEvent, AgentEventKind, ToolEvent, ToolEventKind, format_agent_event
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
    max_read_bytes: int = 24000
    max_replan_count: int = 3
    run_tests_after_edit: bool = True
    test_command: str | None = None  # None = auto-detect (pytest if a tests/ dir exists)
    approval_required: bool = False
    max_continuations: int = 3
    protect_existing_tests: bool = True
    max_tool_calls: int = 64
    max_test_attempts: int = 5
    max_duplicate_attempts: int = 2


class TaskMode(str, Enum):
    ANSWER = "ANSWER"
    PLAN = "PLAN"
    MODIFY = "MODIFY"


@dataclass(frozen=True, slots=True)
class IntentResolution:
    mode: TaskMode
    confidence: str
    reason: str


class IntentRouter:
    """Classify the user's request before entering the agent loop.

    The router is deliberately rule-first. Explicit mutation requests must not
    be downgraded by the model into read-only advice, while explanation and
    plan-only requests should not enter the code-changing workflow.
    """

    CONVERSATIONAL = {
        "你好", "hello", "hi", "谢谢", "thanks", "谢谢你",
        "继续", "继续吧", "可以", "好的", "ok",
    }
    MODIFY_SIGNALS = (
        "修复", "修改", "改成", "改为", "实现", "添加", "增加", "删除",
        "重构", "补充测试", "补测试", "改代码", "写代码", "创建", "提交",
        "fix", "implement", "refactor", "add", "change", "delete", "remove",
        "write", "create",
    )
    PLAN_ONLY_GUARDS = (
        "先不要改", "先别改", "不要改代码", "暂不修改", "暂时不要修改",
        "只给计划", "只给方案", "只分析", "先不要动代码",
        "do not modify", "don't modify", "plan only",
    )
    PLAN_SIGNALS = (
        "计划", "方案", "怎么改", "如何改", "怎么重构", "如何重构",
        "怎么实现", "如何实现", "设计一下", "给我一个计划", "给个计划",
        "roadmap", "plan",
    )
    ANSWER_SIGNALS = (
        "解释", "说明", "分析", "看看", "读一下", "告诉我", "为什么",
        "是什么", "是否", "对齐", "评价", "有什么问题", "有什么可改进",
        "explain", "analyze", "why", "what", "review",
    )

    @classmethod
    def resolve(cls, text: str) -> IntentResolution:
        normalized = text.strip().lower()
        if not normalized or normalized in cls.CONVERSATIONAL:
            return IntentResolution(TaskMode.ANSWER, "explicit", "寒暄或空请求，按普通回答处理")

        if any(signal in normalized for signal in cls.PLAN_ONLY_GUARDS):
            return IntentResolution(TaskMode.PLAN, "explicit", "检测到只规划/不修改的明确约束")

        if any(signal in normalized for signal in cls.PLAN_SIGNALS):
            return IntentResolution(TaskMode.PLAN, "high", "检测到方案或计划类意图")

        if any(signal in normalized for signal in cls.MODIFY_SIGNALS):
            return IntentResolution(TaskMode.MODIFY, "explicit", "检测到明确修改/实现/提交类意图")

        if any(signal in normalized for signal in cls.ANSWER_SIGNALS):
            return IntentResolution(TaskMode.ANSWER, "high", "检测到解释、分析或代码理解类意图")

        if len(normalized) >= 24:
            return IntentResolution(TaskMode.PLAN, "low", "长请求但没有明确修改词，先按只读规划处理")

        return IntentResolution(TaskMode.ANSWER, "low", "未检测到修改词，默认只读回答")


# Tools that modify code on disk and should trigger a verification test run.
_CODE_EDIT_TOOLS = {"write_file", "apply_patch"}
_READ_ONLY_TOOLS = {"list_files", "read_file", "search_text", "summarize_tree", "diff_summary"}
_PHASE_LABELS = {
    "PLANNING": "制定计划",
    "ANSWERING": "只读分析",
    "EXPLORING": "检查项目",
    "IMPLEMENTING": "实现修改",
    "TESTING": "运行验证",
    "FIXING": "修复失败",
    "VERIFYING": "最终检查",
    "COMPLETED": "已完成",
    "PAUSED": "已暂停",
}


class CodingAgent:
    """Coordinates retrieval, LLM decisions, local tools, and observations."""

    def __init__(self, config: AgentConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.workspace = Workspace(config.workspace)
        self.llm = LlmClient()
        self.tools = ToolExecutor(
            self.workspace,
            dry_run=config.dry_run,
            max_tool_retries=config.max_tool_retries,
            max_read_bytes=config.max_read_bytes,
        )
        self.retriever = HybridRetriever(self.workspace)
        self.context_manager = ContextManager(
            max_turns=config.max_context_turns,
            max_tokens=config.max_context_tokens,
            enable_summarization=True,
            llm_client=self.llm,
        )
        self.controller = AgentController(
            llm=self.llm,
            tool_executor=self.tools,
            context_manager=self.context_manager,
            max_steps=config.max_steps,
            max_tool_calls=config.max_tool_calls,
            tool_policy=ToolPolicy(),
            confirmation_callback=lambda tool_call, policy=None: self._confirm_tool(
                tool_call.name, tool_call.args, policy
            ),
            event_callback=self._handle_runtime_event,
        )
        self.verbose = False
        self.streaming_mode = False
        # Session-level token tracking
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self._session_path = self.workspace.root / ".memcode" / "session.json"
        self._stop_requested = False
        self._phase = "IDLE"
        self._plan_text = ""
        self._protected_test_files: set[str] = set()
        self._verification_done = False
        self._verification_passed = False
        self._last_verification_kind: VerificationKind | None = None
        self._test_attempts = 0
        self._verification_unresolved_error = False
        self._last_diff_summary = ""
        self._last_verification_command = ""
        self._last_completion_report = ""
        self._current_task = ""
        self._pending_approval: dict[str, Any] | None = None

    def run(self, task: str) -> str:
        """Single-turn execution through the normal ReAct loop."""
        self.workspace.ensure_exists()
        self._reset_task_state()
        self._current_task = task
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
            "/model", "/models", "/verbose", "/workspace", "/context", "/tokens",
            "/streaming", "/plan", "/explain", "/cache", "/save"
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
            elif user_input == "/plan" or user_input.startswith("/plan "):
                request = user_input[len("/plan"):].strip()
                if not request:
                    self.console.print("[yellow]用法：/plan <要规划的任务>。不会读取或修改文件。[/yellow]")
                else:
                    self._run_plan(request, messages)
                continue
            elif user_input == "/explain" or user_input.startswith("/explain "):
                request = user_input[len("/explain"):].strip()
                if not request:
                    self.console.print("[yellow]用法：/explain <要解释的问题>。不会修改文件。[/yellow]")
                else:
                    messages.append({"role": "user", "content": request})
                    self._run_read_only_interactive(
                        messages,
                        TaskMode.ANSWER,
                        request,
                    )
                    self._save_session(messages)
                continue
            elif user_input.startswith("/"):
                self.console.print(f"[red]Unknown command: {user_input}[/red] (type /help for a list)")
                continue

            # Ordinary messages always use the model-driven ReAct loop.
            messages.append({"role": "user", "content": user_input})
            self._reset_task_state()
            self._current_task = user_input
            self._save_session(messages)
            self._run_loop_interactive(messages)
            self._save_session(messages)

    def _run_loop(self, messages: list[dict[str, Any]], task: str) -> str:
        """Run a single task through the Controller-owned loop."""
        self.controller.handle_user_request("NORMAL", messages)
        self.controller.mark_implementation_started()
        self._stop_requested = False
        self._phase = "IMPLEMENTING"
        last_result = ""
        has_changes = False
        error_retry_count = 0
        last_error_detected = False
        seen_calls: set[tuple[str, str]] = set()
        for _ in range(self.config.max_steps):
            # Duplicate protection applies within one model turn. A later
            # identical command may be a valid retry after an edit or failure.
            seen_calls.clear()
            result = self._run_model_turn(
                messages,
                tool_context=lambda call: {
                    "phase": "IMPLEMENTING",
                    "approval_required": self.config.approval_required,
                    "protected_test": self._is_protected_test_edit(call.name, call.args),
                    "duplicate": self._is_duplicate_call(seen_calls, call),
                },
            )
            if result is None:
                if self.controller.interrupted:
                    last_result = "任务已被用户中断。"
                    break
                last_result = "任务尚未完成，已暂停。"
                break
            last_result = self._limit_response(
                self._strip_tool_protocol(result.decision.content or last_result)
            )
            if result.interrupted:
                last_result = "任务已被用户中断。"
                break
            progress_alert = self.controller.last_progress_alert
            if progress_alert is not None:
                self.controller.last_progress_alert = None
                if progress_alert.kind == "duplicate_tool":
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Runtime warning: {progress_alert.message} "
                                "This call is identical to the previous one. "
                                "If the repetition is intentional and approved, continue; "
                                "otherwise inspect its result or choose a different action."
                            ),
                        }
                    )
                elif progress_alert.kind in {"no_progress", "final_repetition"}:
                    last_result = (
                        f"任务已暂停：{progress_alert.message}"
                    )
                    self._phase = "PAUSED"
                    self._save_session(messages)
                    return last_result
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Runtime progress warning: {progress_alert.message} "
                                "This is only a warning; continue if the current exploration "
                                "is producing new information, otherwise choose a different action."
                            ),
                        }
                    )
            changed = any(
                obs.ok and call.name in _CODE_EDIT_TOOLS
                for call, obs in zip(result.decision.tool_calls, result.observations)
            )
            current_step_has_error = any(not getattr(obs, "ok", False) for obs in result.observations)
            if changed:
                has_changes = True
                test_failed = self._run_verification_tests(messages)
                if self._verification_unresolved_error:
                    self.controller.mark_blocked("verification environment error")
                    last_result = "测试环境或命令错误，任务已暂停。"
                    break
                current_step_has_error = current_step_has_error or test_failed
                self.controller.mark_implementation_done()
                self.controller.mark_diff_checked()
                self.controller.mark_test_result(self._verification_passed)
                if not test_failed and not self._verification_unresolved_error:
                    self.controller.mark_modify_completed()
            if result.decision.is_final:
                if not has_changes:
                    if current_step_has_error:
                        error_retry_count, last_error_detected, should_stop = self._update_error_retry_state(
                            True,
                            error_retry_count,
                            last_error_detected,
                        )
                        if should_stop:
                            self._phase = "PAUSED"
                            self._save_session(messages)
                            return (
                                f"Stopped after {error_retry_count} failed attempts to fix the error."
                            )
                        continue
                    self.retriever.remember_task(task, last_result)
                    self.controller.mark_task_completed()
                    self._phase = "COMPLETED"
                    self._save_session(messages)
                    return last_result
                if self._final_completion_checks(messages):
                    self.retriever.remember_task(task, last_result)
                    self._phase = "COMPLETED"
                    self._save_session(messages)
                    return last_result
            error_retry_count, last_error_detected, should_stop = self._update_error_retry_state(
                current_step_has_error,
                error_retry_count,
                last_error_detected,
            )
            if should_stop:
                self.retriever.remember_task(task, f"Stopped after {error_retry_count} failed attempts to fix the error.")
                self._phase = "PAUSED"
                self._save_session(messages)
                return f"Stopped after {error_retry_count} failed attempts to fix the error."
        self._phase = "PAUSED"
        self.retriever.remember_task(task, last_result)
        self._save_session(messages)
        return last_result or "任务尚未完成，已暂停。"

    @staticmethod
    def _is_duplicate_call(
        seen_calls: set[tuple[str, str]],
        tool_call: Any,
    ) -> bool:
        key = (
            str(tool_call.name),
            json.dumps(tool_call.args or {}, sort_keys=True, ensure_ascii=False),
        )
        duplicate = key in seen_calls
        seen_calls.add(key)
        return duplicate

    def _reset_task_state(self) -> None:
        """Clear per-request verification and progress state before a new task."""
        self._stop_requested = False
        self._phase = "IDLE"
        self._verification_done = False
        self._verification_passed = False
        self._last_verification_kind = None
        self._test_attempts = 0
        self._verification_unresolved_error = False
        self._last_diff_summary = ""
        self._last_verification_command = ""
        self._last_completion_report = ""
        self._protected_test_files = (
            self._snapshot_test_files()
            if self.config.protect_existing_tests
            else set()
        )
        self.controller.reset_budget()

    def _run_loop_legacy(self, messages: list[dict[str, Any]], task: str) -> str:
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
                final = self._strip_tool_protocol(decision.content or "(empty response)")
                self.retriever.remember_task(task, final)
                return final

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

    def _update_error_retry_state(
        self,
        current_step_has_error: bool,
        error_retry_count: int,
        last_error_detected: bool,
    ) -> tuple[int, bool, bool]:
        """Advance the step-level retry counters and report whether the loop should stop."""
        if current_step_has_error:
            if last_error_detected:
                error_retry_count += 1
            else:
                error_retry_count = 1
                last_error_detected = True
            if error_retry_count >= self.config.max_error_retries:
                return error_retry_count, last_error_detected, True
            return error_retry_count, last_error_detected, False
        if last_error_detected:
            self.console.print(f"[green]✓ Error fixed after {error_retry_count} attempt(s). Counter reset.[/green]")
            return 0, False, False
        return error_retry_count, last_error_detected, False

    def _run_model_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        before_tools: Callable[[Any], None] | None = None,
        tool_context: Callable[[Any], dict[str, Any]] | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> ControllerStep | None:
        """Run one model decision and its local tool calls."""
        try:
            if stream:
                result = self.controller.step(
                    messages,
                    before_tools=before_tools,
                    tool_context=tool_context,
                    stream=True,
                    on_chunk=on_chunk,
                )
                if result.decision.content and on_chunk is None:
                    print()
            else:
                if isinstance(self.console, Console):
                    self.console.print("[dim]Thinking...[/dim]")
                result = self.controller.step(
                    messages,
                    before_tools=before_tools,
                    tool_context=tool_context,
                )
        except RuntimeError:
            self.controller.mark_budget_exhausted()
            return None
        except KeyboardInterrupt:
            self.controller.mark_interrupted()
            return None

        self._record_usage(result.decision)
        return result

    def _run_verification_tests(self, messages: list[dict[str, Any]]) -> bool:
        """Run the project's test suite after a code-modifying tool call and feed the
        result back into the conversation as an observation, so the model can react to
        regressions in the same step loop (the automated test-verification part of the
        retry loop).

        Returns True if tests failed, False if tests passed or were skipped.
        """
        command = self._resolve_test_command()
        if not command:
            # A repository without a configured/available test command has no
            # executable verification target; treat this as a deliberate skip,
            # not as a failing verification that can never be satisfied.
            self._verification_done = True
            self._verification_passed = True
            self._last_verification_kind = None
            self._verification_unresolved_error = False
            self._last_verification_command = ""
            return False
        if self._test_attempts >= self.config.max_test_attempts:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Automated test verification: TEST_ATTEMPT_BUDGET_EXHAUSTED. "
                        f"已达到最大测试尝试次数 {self.config.max_test_attempts}。"
                    ),
                }
            )
            self._last_verification_kind = VerificationKind.COMMAND_ERROR
            return True
        self._test_attempts += 1
        self._last_verification_command = command
        self.console.rule("[bold magenta]Verification tests")
        observation = self.tools.execute("run_command", {"command": command}, tool_call_id=None)
        self.console.print(observation.to_display())
        verification = classify_verification(observation.ok, observation.content)
        passed = verification.passed
        self._verification_done = True
        self._verification_passed = passed
        self._last_verification_kind = verification.kind
        self._verification_unresolved_error = verification.kind in {
            VerificationKind.ENVIRONMENT_ERROR,
            VerificationKind.COMMAND_ERROR,
        }
        status = "PASSED" if verification.passed else "FAILED"
        summary = (
            f"Automated test verification after code edit: {status} "
            f"[{verification.kind.value}] ({verification.summary}).\n"
            f"{self._summarize_test_output(observation.content)}"
        )
        self._last_completion_report = summary
        messages.append({"role": "user", "content": summary})
        return not passed  # Return True if tests failed

    def _run_read_only_interactive(
        self,
        messages: list[dict[str, Any]],
        mode: TaskMode,
        request: str,
        *,
        display_final: bool = True,
    ) -> str | None:
        """Run a bounded read-only loop for ANSWER and PLAN tasks.

        This is the first runtime split: code-understanding and plan-only tasks
        may explore the repository, but edits and commands are denied before
        they can reach the local tool executor.
        """
        self._stop_requested = False
        self._phase = "ANSWERING" if mode == TaskMode.ANSWER else "PLANNING"
        mode_instruction = {
            "role": "system",
            "content": (
                f"当前任务模式是 {mode.value}。你可以使用只读工具 list_files、read_file、"
                "search_text、summarize_tree、diff_summary。"
                "不要调用 write_file 或 apply_patch。"
                if mode == TaskMode.ANSWER
                else
                f"当前任务模式是 {mode.value}。你可以使用只读工具 list_files、read_file、"
                "search_text、summarize_tree、diff_summary。"
                "不要修改文件。"
                "最终回答必须是具体计划，包含目标理解、相关文件、修改步骤、风险和验证建议。"
            ),
        }
        readonly_messages = [mode_instruction, *messages]
        seen_calls: set[tuple[str, str]] = set()

        for step in range(1, self.config.max_steps + 1):
            self._print_phase(self._phase, step, self.config.max_steps)
            trimmed = self.context_manager.trim(readonly_messages)
            if self.context_manager.last_trim_notice:
                self.console.print(f"[yellow]{self.context_manager.last_trim_notice}[/yellow]")

            try:
                if self.streaming_mode:
                    self.console.print("[green]Streaming:[/green]")

                    def _on_chunk(text: str) -> None:
                        print(text, end="", flush=True)

                    decision = self.llm.next_action(trimmed, stream=True, on_chunk=_on_chunk)
                    if decision.content:
                        print()
                else:
                    with Status("[cyan]Thinking...[/cyan]", console=self.console):
                        decision = self.llm.next_action(trimmed)
            except KeyboardInterrupt:
                self.console.print("[yellow]已中断。[/yellow]")
                self._phase = "PAUSED"
                return None

            self._record_usage(decision)

            if decision.is_final:
                content = self._clean_display_text(decision.content or "(empty response)")
                if display_final and not self.streaming_mode:
                    self.console.print(f"[green]{content}[/green]")
                messages.append({"role": "assistant", "content": content})
                self._phase = "COMPLETED"
                self.retriever.remember_task(request, content)
                self._save_session(messages)
                return content

            readonly_messages.append(decision.assistant_message)
            messages.append(decision.assistant_message)
            self._save_session(messages)
            tool_names = [tc.name for tc in decision.tool_calls]
            self.console.print(f"[dim]→ readonly: {', '.join(tool_names)}[/dim]")

            for tool_call in decision.tool_calls:
                call_key = (tool_call.name, json.dumps(tool_call.args, sort_keys=True, ensure_ascii=False))
                allowed_tools = _READ_ONLY_TOOLS
                if tool_call.name not in allowed_tools:
                    observation = ToolObservation(
                        tool_call.name,
                        False,
                        f"{mode.value} 模式只允许只读工具；{tool_call.name} 已被 Runtime 拒绝。",
                        tool_call.id,
                    )
                else:
                    seen_calls.add(call_key)
                    try:
                        observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                    except KeyboardInterrupt:
                        self.console.print("[yellow]已在工具执行期间中断。[/yellow]")
                        self._phase = "PAUSED"
                        return None

                if self.verbose:
                    self.console.print(observation.to_display())
                else:
                    status_icon = "✓" if observation.ok else "✗"
                    self.console.print(
                        f"[dim]  {status_icon} {observation.tool_name}"
                        f"{self._tool_target(observation.tool_name, tool_call.args)}[/dim]"
                    )
                readonly_messages.append(observation.to_message())
                messages.append(observation.to_message())
                self._save_session(messages)

        self._print_agent_event(
            AgentEventKind.PAUSED,
            f"{mode.value} 模式已达到 {self.config.max_steps} 步预算",
            "任务暂停，可继续提问或改成明确的修改请求",
        )
        self._phase = "PAUSED"
        return None

    def _run_loop_interactive(self, messages: list[dict[str, Any]]) -> None:
        """Run the same ReAct core as single-turn mode and render its result."""
        user_messages = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        ]
        task = user_messages[-1] if user_messages else ""
        result = self._run_loop(messages, task)
        if result:
            self.console.print(f"[green]{result}[/green]")

    def _run_loop_interactive_legacy(self, messages: list[dict[str, Any]]) -> None:
        """Core agent loop for interactive mode: uses per-error retry counter."""
        self.controller.handle_user_request("MODIFY", messages)

        error_retry_count = 0
        last_error_detected = False
        self._stop_requested = False
        continuation_count = 0
        replan_count = 0
        seen_calls: set[tuple[str, str]] = set()
        phase = "INSPECTING"
        pending_edits = 0
        explored = True
        has_changes = False
        while True:
            self.controller.reset_budget()
            for step in range(1, self.config.max_steps + 1):
                step_meta: dict[str, Any] = {"tool_names": []}

                def before_tools(decision: Any) -> None:
                    tool_names = [tc.name for tc in decision.tool_calls]
                    step_meta["tool_names"] = tool_names
                    nonlocal phase
                    if any(name in _CODE_EDIT_TOOLS for name in tool_names) and explored:
                        phase = "IMPLEMENTING"
                    elif pending_edits and "run_command" in tool_names:
                        phase = "TESTING"
                    elif any(name in _READ_ONLY_TOOLS for name in tool_names):
                        phase = "INSPECTING"
                    self._print_phase(phase, step, self.config.max_steps)
                    if self.verbose:
                        self.console.rule(f"[bold blue]Step {step}")
                        self.console.print(decision.to_display())
                    else:
                        self.console.print(f"[dim]→ calling {', '.join(tool_names)}...[/dim]")

                def tool_context(tool_call: Any) -> dict[str, Any]:
                    call_key = (
                        tool_call.name,
                        json.dumps(tool_call.args, sort_keys=True, ensure_ascii=False),
                    )
                    duplicate = call_key in seen_calls
                    if not duplicate:
                        seen_calls.add(call_key)
                    return {
                        "phase": phase,
                        "approval_required": self.config.approval_required,
                        "explored": explored,
                        "protected_test": self._is_protected_test_edit(
                            tool_call.name, tool_call.args
                        ),
                        "duplicate": duplicate,
                    }

                if self.streaming_mode:
                    self.console.print("[green]Streaming:[/green]")

                    def _on_chunk(text: str) -> None:
                        print(text, end="", flush=True)

                    result = self._run_model_turn(
                        messages,
                        before_tools=before_tools,
                        tool_context=tool_context,
                        stream=True,
                        on_chunk=_on_chunk,
                    )
                    if result is not None and result.decision.content:
                        print()
                else:
                    result = self._run_model_turn(
                        messages,
                        before_tools=before_tools,
                        tool_context=tool_context,
                    )
                if result is None:
                    if self.controller.interrupted:
                        self._print_agent_event(AgentEventKind.PAUSED, "任务已中断", "已保存当前状态")
                    else:
                        self.controller.mark_budget_exhausted()
                        self._print_agent_event(
                            AgentEventKind.PAUSED,
                            "任务尚未完成，已暂停",
                            f"已执行 {self.config.max_steps}/{self.config.max_steps} 步；可继续执行",
                        )
                    self._phase = "PAUSED"
                    self._save_session(messages)
                    return

                decision = result.decision
                if self.controller.last_progress_alert is not None and self.controller.last_progress_alert.kind == "final_repetition":
                    alert = self.controller.last_progress_alert
                    self._print_agent_event(AgentEventKind.ALERT, "重复回答检测", alert.message)
                    self._print_agent_event(AgentEventKind.PAUSED, "任务已暂停", "请补充约束或继续执行")
                    self._phase = "PAUSED"
                    self._save_session(messages)
                    return
                if self.context_manager.last_trim_notice:
                    self._print_agent_event(
                        AgentEventKind.ALERT,
                        "上下文压缩",
                        self.context_manager.last_trim_notice,
                    )

                if decision.is_final:
                    if pending_edits:
                        self._print_phase("TESTING", step, self.config.max_steps)
                        test_failed = self._run_verification_tests(messages)
                        pending_edits = 0
                        if test_failed:
                            replan_count += 1
                            if replan_count >= self.config.max_replan_count:
                                self._print_agent_event(
                                    AgentEventKind.PAUSED,
                                    "重规划次数已耗尽",
                                    "测试失败后已多次重新回到实现，任务暂停",
                                )
                                self._phase = "PAUSED"
                                self._save_session(messages)
                                return
                            phase = "FIXING"
                            self._print_phase(phase, step, self.config.max_steps)
                            continue
                        phase = "VERIFYING"
                        self._print_phase(phase, step, self.config.max_steps)
                    if has_changes:
                        self._print_diff_summary(messages)
                    if not self._final_completion_checks(messages):
                        replan_count += 1
                        if replan_count >= self.config.max_replan_count:
                            self._print_agent_event(
                                AgentEventKind.PAUSED,
                                "重规划次数已耗尽",
                                "验收条件仍未满足，任务暂停",
                            )
                            self._phase = "PAUSED"
                            self._save_session(messages)
                            return
                        self.console.print("[yellow]最终完成条件未满足，任务继续。[/yellow]")
                        phase = "FIXING"
                        continue
                    if not self.streaming_mode:
                        self._print_agent_event(
                            AgentEventKind.DONE,
                            "任务已完成",
                            self._build_completion_report(),
                        )
                    self._phase = "COMPLETED"
                    self._save_session(messages)
                    return
                self._save_session(messages)

                code_changed = False
                current_step_has_error = False

                tool_names = step_meta["tool_names"]
                for tool_call, observation in zip(decision.tool_calls, result.observations):
                    if self.verbose:
                        self.console.print(observation.to_display())
                    else:
                        status_icon = "✓" if observation.ok else "✗"
                        self.console.print(f"[dim]  {status_icon} {observation.tool_name}[/dim]")
                    self._save_session(messages)

                    if not observation.ok:
                        current_step_has_error = True

                    if observation.ok and tool_call.name in _CODE_EDIT_TOOLS:
                        code_changed = True
                        phase = "IMPLEMENTING"
                        pending_edits += 1
                        has_changes = True
                    if observation.ok and tool_call.name in _READ_ONLY_TOOLS:
                        explored = True

                if self.controller.last_progress_alert is not None:
                    alert = self.controller.last_progress_alert
                    self._print_agent_event(AgentEventKind.ALERT, "进度监控", alert.message)
                    self.controller.last_progress_alert = None
                    if alert.kind in {"no_progress", "tool_monotony", "phase_monotony"}:
                        self._print_agent_event(
                            AgentEventKind.PAUSED,
                            "任务因连续无进展而暂停",
                            "可继续或调整请求",
                        )
                        self._phase = "PAUSED"
                        return

                # Validate after a small batch of edits, or when the model has
                # moved on from editing to another kind of action. This avoids
                # running the full suite after every single patch while still
                # feeding failures back before the task drifts too far.
                should_verify = code_changed and (
                    pending_edits >= 2
                    or not any(name in _CODE_EDIT_TOOLS for name in tool_names)
                )
                if should_verify:
                    self.controller.mark_implementation_done()
                    phase = "TESTING"
                    self.controller.mark_diff_checked()
                    self._print_phase(phase, step, self.config.max_steps)
                    test_failed = self._run_verification_tests(messages)
                    pending_edits = 0
                    if self._verification_unresolved_error:
                        self.controller.mark_blocked("verification environment error")
                        self._print_agent_event(
                            AgentEventKind.ALERT,
                            "测试环境异常",
                            "命令或依赖错误，已停止继续修改并等待修复环境",
                        )
                        self._phase = "PAUSED"
                        return
                    if test_failed:
                        current_step_has_error = True
                        replan_count += 1
                        if replan_count >= self.config.max_replan_count:
                            self._print_agent_event(
                                AgentEventKind.PAUSED,
                                "重规划次数已耗尽",
                                "连续测试失败后已超过允许的重新规划次数",
                            )
                            self._phase = "PAUSED"
                            self._save_session(messages)
                            return
                        phase = "FIXING"
                        self.controller.mark_test_result(False)
                    else:
                        phase = "VERIFYING"
                        self.controller.mark_test_result(True)

                if current_step_has_error:
                    error_retry_count, last_error_detected, should_stop = self._update_error_retry_state(
                        True,
                        error_retry_count,
                        last_error_detected,
                    )
                    if should_stop:
                        self.console.print(
                            f"[yellow]Stopped after {error_retry_count} failed attempts to fix the error.[/yellow]"
                        )
                        return
                else:
                    error_retry_count, last_error_detected, _ = self._update_error_retry_state(
                        False,
                        error_retry_count,
                        last_error_detected,
                    )

            self.controller.mark_budget_exhausted()
            self.console.print(f"[yellow]任务尚未完成，已暂停。已执行 {step}/{self.config.max_steps} 步。[/yellow]")
            self._phase = "PAUSED"
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
        self.console.print(self._strip_tool_protocol(decision.content or "(No plan returned.)"))
        self.console.print("[dim]No files or commands have been changed during planning.[/dim]")
        self.controller.transition(RuntimeEvent.PLAN_READY)
        try:
            approved = questionary.confirm("开始按这个计划执行？", default=True).ask()
        except (KeyboardInterrupt, EOFError):
            approved = False
        if not approved:
            self.console.print("[dim]Plan accepted? No. Task remains unchanged.[/dim]")
            self.controller.transition(RuntimeEvent.USER_REJECTED)
            return False

        # Keep the approved plan in the conversation so later tool decisions
        # can follow the same outline without relying on hidden state.
        if decision.content:
            cleaned_plan = self._strip_tool_protocol(decision.content)
            self._plan_text = cleaned_plan
            self.controller.set_plan(cleaned_plan)
            messages.append({"role": "assistant", "content": f"计划：\n{cleaned_plan}"})
        messages.append({"role": "user", "content": "计划已确认。现在开始执行，按计划检查、修改并验证项目。"})
        self.controller.transition(RuntimeEvent.USER_APPROVED)
        self._phase = "IMPLEMENTING"
        self._save_session(messages)
        return True

    def _prepare_task(self, messages: list[dict[str, Any]]) -> bool:
        """Explore with read-only tools, then generate and approve a grounded plan."""
        if self.config.protect_existing_tests:
            self._protected_test_files = self._snapshot_test_files()
        self._phase = "EXPLORING"
        self.console.rule("[bold cyan]探索项目")
        exploration_instruction = {
            "role": "system",
            "content": (
                f"当前 workspace 是 {self.workspace.root}。你现在处于只读探索阶段，只能调用 "
                "list_files、read_file、search_text。请先阅读需求文档、项目说明、相关源码和现有测试，"
                "不要修改文件、不要运行命令。每次只读取与任务直接相关的内容；读取完成后用一句话说明 "
                "已掌握的文件和发现。"
            ),
        }
        explored = False
        for step in range(1, 5):
            self._print_phase("EXPLORING", step, 4)
            trimmed = self.context_manager.trim([exploration_instruction, *messages])
            try:
                decision = self.llm.next_action(trimmed)
            except KeyboardInterrupt:
                self.console.print("[yellow]探索已中断。[/yellow]")
                return False
            self._record_usage(decision)
            if decision.is_final:
                if decision.content:
                    messages.append(decision.assistant_message)
                    self.console.print(f"[dim]{self._strip_tool_protocol(decision.content)}[/dim]")
                break
            messages.append(decision.assistant_message)
            for tool_call in decision.tool_calls:
                if tool_call.name not in _READ_ONLY_TOOLS:
                    observation = ToolObservation(
                        tool_call.name,
                        False,
                        "探索阶段只允许读取和搜索，暂不执行修改或命令。",
                        tool_call.id,
                    )
                else:
                    try:
                        observation = self.tools.execute(tool_call.name, tool_call.args, tool_call.id)
                    except KeyboardInterrupt:
                        self.console.print("[yellow]探索已中断。[/yellow]")
                        return False
                    explored = explored or observation.ok
                self.console.print(
                    f"[dim]  {'✓' if observation.ok else '✗'} {observation.tool_name}"
                    f"{self._tool_target(observation.tool_name, tool_call.args)}[/dim]"
                )
                messages.append(observation.to_message())
        if not explored:
            self.console.print("[yellow]未获得有效的项目探索结果，任务暂停。[/yellow]")
            return False
        self.controller.transition(RuntimeEvent.EXPLORATION_COMPLETE)
        return self._confirm_interactive_plan(messages)

    def _print_diff_summary(self, messages: list[dict[str, Any]]) -> None:
        observation = self.tools.execute("diff_summary", {}, tool_call_id=None)
        if observation.ok and "No git repository found" not in observation.content:
            self.console.print("[dim]  ✓ diff summary[/dim]")
            self._last_diff_summary = observation.content
            messages.append({"role": "user", "content": f"Final diff summary:\n{observation.content}"})

    def _final_completion_checks(self, messages: list[dict[str, Any]]) -> bool:
        diff_obs = self.tools.execute("diff_summary", {}, tool_call_id=None)
        if not diff_obs.ok:
            return False
        self._last_diff_summary = diff_obs.content
        messages.append({"role": "user", "content": f"Completion diff summary:\n{diff_obs.content}"})
        files_changed = "No changes" not in diff_obs.content and "no changes" not in diff_obs.content
        state = CompletionState(
            diff_checked=True,
            verification_done=self._verification_done,
            verification_passed=self._verification_passed,
            unresolved_errors=self._verification_unresolved_error,
            files_changed=files_changed,
        )
        return CompletionGuard.can_finish(TaskMode.MODIFY, state)

    def _snapshot_test_files(self) -> set[str]:
        files: list[Path] = []
        for dirname in ("tests", "test"):
            root = self.workspace.root / dirname
            if root.is_dir():
                files.extend(path for path in root.rglob("*") if path.is_file())
        files.extend(path for path in self.workspace.root.glob("test_*.py") if path.is_file())
        return {
            path.relative_to(self.workspace.root).as_posix()
            for path in files
            if not self.workspace.should_ignore(path)
        }

    def _is_protected_test_edit(self, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name not in _CODE_EDIT_TOOLS or not self._protected_test_files:
            return False
        path = str(args.get("path", "")).replace("\\", "/").lstrip("./")
        return path in self._protected_test_files

    def _record_usage(self, decision: Any) -> None:
        usage = getattr(decision, "usage", None)
        if not usage:
            return
        def number(name: str) -> int:
            value = getattr(usage, name, 0)
            return int(value) if isinstance(value, (int, float)) else 0
        prompt_tokens = number("prompt_tokens")
        completion_tokens = number("completion_tokens")
        total_tokens = number("total_tokens")
        self.session_prompt_tokens += prompt_tokens
        self.session_completion_tokens += completion_tokens
        self.session_total_tokens += total_tokens
        self.console.print(
            f"[dim]Tokens: {total_tokens:,} "
            f"(prompt: {prompt_tokens:,}, "
            f"completion: {completion_tokens:,}) | "
            f"Session total: {self.session_total_tokens:,}[/dim]"
        )

    def _resolve_intent(self, text: str) -> IntentResolution | None:
        """Resolve an intent automatically from the user's text."""
        return IntentRouter.resolve(text)

    @staticmethod
    def _summarize_test_output(content: str, limit: int = 2000) -> str:
        lines = content.splitlines()
        interesting = [
            line for line in lines
            if (
                "FAILED " in line
                or "ERROR " in line
                or "passed" in line
                or "failed" in line
                or "exit_code=" in line
                or line.strip().startswith(("E   ", ">"))
            )
        ]
        text = "\n".join(interesting or lines[-40:])
        return text if len(text) <= limit else text[:limit] + "\n...[test output shortened]"

    @staticmethod
    def _tool_target(name: str, args: dict[str, Any]) -> str:
        if name == "read_file":
            path = args.get("path") or ""
            start = args.get("start_line")
            end = args.get("end_line")
            suffix = f":{start}-{end}" if start or end else ""
            return f" [{path}{suffix}]"
        if name in {"list_files", "summarize_tree"}:
            return f" [{args.get('path') or args.get('glob') or ''}]"
        if name == "search_text":
            return f" [{args.get('query', '')}]"
        if name == "run_command":
            command = " ".join(str(args.get("command", "")).split())
            if len(command) > 96:
                command = command[:93] + "..."
            return f" [{command}]"
        if name == "diff_summary":
            return " [git diff]"
        return ""

    @staticmethod
    def _tool_allowed_in_phase(phase: str, tool_name: str, explored: bool) -> bool:
        if phase in {"PLANNING", "COMPLETED", "PAUSED"}:
            return False
        if phase in {"EXPLORING", "INSPECTING"}:
            return tool_name in _READ_ONLY_TOOLS
        if phase in {"IMPLEMENTING", "FIXING"}:
            return tool_name in _READ_ONLY_TOOLS | _CODE_EDIT_TOOLS | {"run_command"}
        if phase in {"TESTING", "VERIFYING"}:
            return tool_name in {"read_file", "search_text", "diff_summary", "run_command"}
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
        self._phase = phase
        event = AgentEvent(
            AgentEventKind.PHASE,
            f"阶段：{_PHASE_LABELS.get(phase, phase)}",
            f"步骤 {step}/{limit}",
        )
        self.console.print(
            f"[cyan]{format_agent_event(event)}[/cyan]"
        )

    def _print_agent_event(self, kind: AgentEventKind, message: str, detail: str = "") -> None:
        self.console.print(f"[yellow]{format_agent_event(AgentEvent(kind, message, detail))}[/yellow]")

    def _handle_runtime_event(self, event: Any) -> None:
        if not isinstance(event, ToolEvent):
            return
        if event.kind == ToolEventKind.CALL:
            payload = event.payload or {}
            args = payload.get("args") if isinstance(payload, dict) else {}
            self.console.print(
                f"[dim]→ tool {event.tool}{self._tool_target(event.tool, args if isinstance(args, dict) else {})}[/dim]"
            )

    @staticmethod
    def _strip_tool_protocol(text: str) -> str:
        cleaned = text.replace("<tool_call>", "").replace("</tool_call>", "")
        cleaned = cleaned.replace("<tool-result>", "").replace("</tool-result>", "")
        return cleaned

    @staticmethod
    def _limit_response(text: str, limit: int = 12000) -> str:
        """Keep abnormal model output from flooding the terminal or session UI."""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[final response truncated by runtime]"

    def _build_completion_report(self) -> str:
        parts: list[str] = []
        if self._last_diff_summary:
            parts.append(self._last_diff_summary[:800])
        if self._last_verification_command:
            parts.append(f"Verification command: {self._last_verification_command}")
        if self._last_verification_kind is not None:
            parts.append(f"Verification result: {self._last_verification_kind.value}")
        if self._verification_unresolved_error:
            parts.append("Unresolved issue: verification environment or command error")
        if self._last_completion_report:
            parts.append(self._last_completion_report[:300])
        return " | ".join(parts) if parts else "No completion report available."

    def _requires_approval(self, tool_name: str) -> bool:
        """Require confirmation only for operations that can change state."""
        return self.config.approval_required and tool_name in {"write_file", "apply_patch", "run_command"}

    def _confirm_tool(self, name: str, args: dict[str, Any], policy: Any | None = None) -> bool:
        self._pending_approval = {
            "tool": name,
            "args": dict(args),
            "reason": getattr(policy, "reason", "") if policy is not None else "",
        }
        try:
            reason = getattr(policy, "reason", "") if policy is not None else ""
            suffix = f" ({reason})" if reason else ""
            self.console.print()
            self.console.print("[bold yellow]Approval required[/bold yellow]")
            self.console.print(
                f"[yellow]{self._tool_summary(name, args)}{suffix}[/yellow]"
            )
            answer = questionary.confirm(
                "Allow this operation? (yes/no)",
                default=False,
            ).ask()
            return bool(answer)
        except (KeyboardInterrupt, EOFError):
            self._stop_requested = True
            return False
        finally:
            self._pending_approval = None

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
        self.console.print(self._strip_tool_protocol(decision.content or "(no plan returned)"))

    def _save_session(self, messages: list[dict[str, Any]]) -> None:
        import json
        import os
        import tempfile

        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "messages": messages,
            "task": self._current_task,
            "pending_approval": self._pending_approval,
            "phase": self._phase,
            "plan": self._plan_text,
            "controller": self.controller.persist_state(),
            "verification": {
                "done": self._verification_done,
                "passed": self._verification_passed,
                "kind": None if self._last_verification_kind is None else self._last_verification_kind.value,
                "attempts": self._test_attempts,
            },
        }
        fd, temp_name = tempfile.mkstemp(
            prefix="session.", suffix=".tmp", dir=self._session_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._session_path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def _load_session(self) -> list[dict[str, Any]] | None:
        import json
        try:
            data = json.loads(self._session_path.read_text(encoding="utf-8"))
            messages = data.get("messages")
            self._current_task = str(data.get("task", ""))
            pending_approval = data.get("pending_approval")
            self._pending_approval = (
                pending_approval if isinstance(pending_approval, dict) else None
            )
            self._phase = data.get("phase", "IDLE")
            self._plan_text = data.get("plan", "")
            self._restore_runtime_state(data)
            return messages if isinstance(messages, list) and messages else None
        except json.JSONDecodeError:
            self.console.print(
                "[yellow]会话文件格式不完整，已忽略损坏历史并创建新会话。[/yellow]"
            )
            return None
        except OSError:
            return None

    def _restore_runtime_state(self, data: dict[str, Any]) -> None:
        controller_state = data.get("controller")
        if isinstance(controller_state, dict):
            self.controller.restore_state(controller_state)
            self._plan_text = self.controller.plan_text
        verification = data.get("verification")
        if isinstance(verification, dict):
            self._verification_done = bool(verification.get("done", False))
            self._verification_passed = bool(verification.get("passed", False))
            kind = verification.get("kind")
            try:
                self._last_verification_kind = (
                    None if kind is None else VerificationKind(str(kind))
                )
            except ValueError:
                self._last_verification_kind = None
            self._test_attempts = int(verification.get("attempts", 0))

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
                    "use line ranges for large files, avoid repeating an identical tool call, do not "
                    "weaken or delete tests just to make them pass, and finish with verification and "
                    "a concise summary. After a code edit, the runtime automatically runs the configured "
                    "test command and adds its result to the conversation; do not immediately call the "
                    "same test command again unless the user explicitly asks for a rerun or another edit "
                    f"has happened. {self._shell_instructions()}"
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
                    "use line ranges for large files, avoid repeating an identical tool call, do not "
                    "weaken or delete tests just to make them pass, and finish with verification and "
                    "a concise summary. After a code edit, the runtime automatically runs the configured "
                    "test command and adds its result to the conversation; do not immediately call the "
                    "same test command again unless the user explicitly asks for a rerun or another edit "
                    f"has happened. {self._shell_instructions()} "
                    "You are in an interactive chat session, so the user may ask follow-up questions or "
                    "refine their requests across multiple turns."
                ),
            },
        ]

    @staticmethod
    def _shell_instructions() -> str:
        if platform.system().lower() == "windows":
            return (
                "The current operating system is Windows; use PowerShell or Windows-compatible "
                "commands, never POSIX background syntax such as `&`, `$!`, or `2>&1`."
            )
        return (
            f"The current operating system is {platform.system() or 'Unix'}; use commands "
            "compatible with that platform and do not assume Windows paths."
        )

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
            ("/explain [question]", "Read-only explanation; do not modify files"),
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

