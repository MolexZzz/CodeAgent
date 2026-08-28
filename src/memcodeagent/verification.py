"""Structured classification of local verification command results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VerificationKind(str, Enum):
    PASS = "PASS"
    ASSERTION_FAILURE = "ASSERTION_FAILURE"
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMMAND_ERROR = "COMMAND_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    kind: VerificationKind
    summary: str
    output: str = ""

    @property
    def passed(self) -> bool:
        return self.kind == VerificationKind.PASS


def classify_verification(ok: bool, content: str) -> VerificationResult:
    text = content or ""
    lowered = text.lower()
    if ok and "exit_code=0" in lowered:
        return VerificationResult(VerificationKind.PASS, "验证通过", text)
    if any(token in lowered for token in ("modulenotfounderror", "filenotfounderror", "no such file", "not recognized")):
        return VerificationResult(VerificationKind.ENVIRONMENT_ERROR, "验证环境或依赖不可用", text)
    if "exit_code=" not in lowered or "command not found" in lowered:
        return VerificationResult(VerificationKind.COMMAND_ERROR, "验证命令无法执行", text)
    if any(token in lowered for token in ("assertionerror", "failed", "assert ")):
        return VerificationResult(VerificationKind.ASSERTION_FAILURE, "测试断言失败", text)
    if any(token in lowered for token in ("syntaxerror", "compile error", "compilation failed", "error:")):
        return VerificationResult(VerificationKind.COMPILE_ERROR, "编译或语法错误", text)
    return VerificationResult(VerificationKind.RUNTIME_ERROR, "程序运行失败", text)

