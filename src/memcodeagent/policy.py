"""Tool authorization policy for the single-agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import re


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    CONFIRM = "CONFIRM"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str = ""
    risk: str = "normal"


class ToolPolicy:
    """Pure tool policy; it never executes tools or prompts the user."""

    READ_ONLY = {
        "list_files",
        "read_file",
        "search_text",
        "summarize_tree",
        "diff_summary",
        "read_file_range",
        "summarize_symbols",
    }
    EDIT_TOOLS = {"write_file", "apply_patch"}
    CONFIRM_TOOLS = EDIT_TOOLS | {"run_command"}

    SAFE_COMMANDS = (
        r"(?:python|python3)(?:\s+-m\s+pytest|\s+-m\s+unittest)\b",
        r"pytest\b",
        r"mvnw?\s+(?:-q\s+)?(?:test|verify|compile)\b",
        r"(?:npm|pnpm|yarn)\s+run\s+(?:test|build|lint)\b",
        r"git\s+(?:status|diff|log|show)\b",
        r"(?:dir|ls|pwd|type|cat|whoami|echo)\b",
    )

    @staticmethod
    def command_risk(command: str) -> tuple[str, str]:
        """Classify shell commands without executing them.

        Risky commands remain confirmable (rather than being silently blocked),
        matching interactive coding-agent behavior while making the prompt
        explain why confirmation is needed.
        """
        text = " ".join(str(command).split()).lower()
        if re.search(r"(^|[;&|])\s*(rm|del|erase|rmdir|remove-item|format|rd)\b", text):
            return "destructive", "可能删除文件或目录"
        if re.search(r"\b(git\s+(reset|clean)|checkout\s+--|restore\s+--)", text):
            return "destructive", "可能丢弃未提交的修改"
        if re.search(r"(?:^|[;&|])\s*(start|taskkill|kill|pkill)\b", text):
            return "high", "将启动或终止后台进程"
        if re.search(r"\b(pip|npm|pnpm|yarn|cargo|go)\s+install\b", text):
            return "environment", "可能安装依赖并修改环境"
        if re.search(r"\b(curl|wget|invoke-webrequest|irm)\b", text):
            return "network", "可能访问外部网络或下载内容"
        if any(re.search(pattern, text) for pattern in ToolPolicy.SAFE_COMMANDS):
            return "external", "将执行 Maven 测试、构建或只读检查命令"
        if re.search(r"\bmvn(?:w)?\s+(compile|package|install|deploy)\b", text):
            return "external", "将执行 Maven 构建命令并可能修改构建产物"
        if re.search(r"(^|\s)(mv|move|cp|copy|tee)\b", text):
            return "filesystem", "可能覆盖、移动或复制文件"
        if re.search(r">{1,2}|2>&1", text):
            return "redirect", "将命令输出写入日志文件"
        return "normal", "可能执行外部命令"

    def evaluate(
        self,
        *,
        phase: str,
        tool_name: str,
        approval_required: bool,
        explored: bool = True,
        protected_test: bool = False,
        duplicate: bool = False,
        command: str = "",
        approved: bool = False,
    ) -> PolicyDecision:
        if protected_test:
            return PolicyDecision(
                PolicyAction.DENY,
                "Baseline test files are protected; add a new test instead.",
            )
        # Runtime phases describe the agent's current activity for display and
        # persistence. They must not decide whether the model may inspect,
        # edit, or verify: ReAct tasks can move between those actions freely.
        if tool_name in self.CONFIRM_TOOLS and approval_required:
            if tool_name == "run_command":
                risk, explanation = self.command_risk(command)
                if duplicate:
                    return PolicyDecision(PolicyAction.CONFIRM, explanation, risk)
                if risk == "destructive" and re.search(r"[;&|]", command):
                    return PolicyDecision(PolicyAction.DENY, explanation, risk)
                # Verification and inspection commands do not need to interrupt
                # a normal edit/test loop.
                if risk == "external" and any(
                    re.search(pattern, " ".join(command.lower().split()))
                    for pattern in self.SAFE_COMMANDS
                ):
                    return PolicyDecision(PolicyAction.ALLOW, explanation, risk)
                if approved and risk not in {"network", "environment", "high"}:
                    return PolicyDecision(PolicyAction.ALLOW, explanation, risk)
                return PolicyDecision(PolicyAction.CONFIRM, explanation, risk)
            if approved:
                return PolicyDecision(PolicyAction.ALLOW)
            if tool_name == "write_file":
                return PolicyDecision(
                    PolicyAction.CONFIRM,
                    "将创建或覆盖工作区文件",
                    "filesystem",
                )
            if tool_name == "apply_patch":
                return PolicyDecision(
                    PolicyAction.CONFIRM,
                    "将修改工作区文件中的指定内容",
                    "filesystem",
                )
            return PolicyDecision(
                PolicyAction.CONFIRM,
                "该操作可能修改工作区状态",
            )
        return PolicyDecision(PolicyAction.ALLOW)
