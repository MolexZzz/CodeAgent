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

    @staticmethod
    def command_risk(command: str) -> tuple[str, str]:
        """Classify shell commands without executing them.

        Risky commands remain confirmable (rather than being silently blocked),
        matching interactive coding-agent behavior while making the prompt
        explain why confirmation is needed.
        """
        text = " ".join(str(command).split()).lower()
        if re.search(r"(^|[;&|])\s*(rm|del|erase|rmdir|remove-item|format)\b", text):
            return "destructive", "可能删除文件或目录"
        if re.search(r"\b(git\s+(reset|clean)|checkout\s+--|restore\s+--)", text):
            return "destructive", "可能丢弃未提交的修改"
        if re.search(r"\b(pip|npm|pnpm|yarn|cargo|go)\s+install\b", text):
            return "environment", "可能安装依赖并修改环境"
        if re.search(r"\b(curl|wget|invoke-webrequest|irm)\b", text):
            return "network", "可能访问外部网络或下载内容"
        if re.search(r"(^|\s)(mv|move|cp|copy|tee)\b|>{1,2}", text):
            return "filesystem", "可能覆盖、移动或复制文件"
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
    ) -> PolicyDecision:
        if protected_test:
            return PolicyDecision(
                PolicyAction.DENY,
                "Baseline test files are protected; add a new test instead.",
            )
        if duplicate:
            return PolicyDecision(
                PolicyAction.DENY,
                "Duplicate tool call suppressed; use a different path, range, query, or command.",
            )
        if phase in {"PLANNING", "COMPLETED", "PAUSED"}:
            return PolicyDecision(
                PolicyAction.DENY,
                f"当前阶段不允许调用 {tool_name}。",
            )
        if phase in {"EXPLORING", "INSPECTING"} and tool_name not in self.READ_ONLY:
            return PolicyDecision(
                PolicyAction.DENY,
                f"当前阶段只允许只读工具，{tool_name} 已被拒绝。",
            )
        if phase in {"TESTING", "VERIFYING"} and tool_name not in (
            self.READ_ONLY | {"run_command"}
        ):
            return PolicyDecision(
                PolicyAction.DENY,
                f"当前验证阶段不允许调用 {tool_name}。",
            )
        if phase in {"IMPLEMENTING", "FIXING"} and tool_name not in (
            self.READ_ONLY | self.EDIT_TOOLS | {"run_command"}
        ):
            return PolicyDecision(
                PolicyAction.DENY,
                f"当前实现阶段不允许调用 {tool_name}。",
            )
        if tool_name in self.CONFIRM_TOOLS and approval_required:
            if tool_name == "run_command":
                risk, explanation = self.command_risk(command)
                return PolicyDecision(PolicyAction.CONFIRM, explanation, risk)
            return PolicyDecision(
                PolicyAction.CONFIRM,
                "该工具可能修改文件、环境或执行外部命令。",
            )
        return PolicyDecision(PolicyAction.ALLOW)
