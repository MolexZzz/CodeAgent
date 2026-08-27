from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentDecision:
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    raw: str = ""

    def to_message(self) -> dict[str, str]:
        return {"role": "assistant", "content": self.raw or self.to_json()}

    def to_json(self) -> str:
        if self.final_answer:
            return json.dumps({"final": self.final_answer}, ensure_ascii=False)
        return json.dumps({"tool": self.tool_name, "args": self.tool_args}, ensure_ascii=False)

    def to_display(self) -> str:
        if self.final_answer:
            return f"[green]Final answer:[/green] {self.final_answer}"
        return f"[yellow]Tool:[/yellow] {self.tool_name} [yellow]Args:[/yellow] {self.tool_args}"


class LlmClient:
    """Small wrapper around an OpenAI-compatible chat model."""

    def __init__(self) -> None:
        self.model = os.getenv("MEMCODE_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL") or None

    def next_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        if not self.api_key:
            return AgentDecision(
                final_answer=(
                    "LLM is not configured yet. Set OPENAI_API_KEY, then rerun the task. "
                    "The CLI framework, retrieval stub, and tool executor are ready."
                )
            )

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages + [{"role": "system", "content": self._output_contract()}],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return self._parse_decision(content)

    def _parse_decision(self, content: str) -> AgentDecision:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return AgentDecision(final_answer=f"Model returned invalid JSON: {content}", raw=content)

        if "final" in payload:
            return AgentDecision(final_answer=str(payload["final"]), raw=content)
        return AgentDecision(
            tool_name=payload.get("tool"),
            tool_args=payload.get("args") or {},
            raw=content,
        )

    def _output_contract(self) -> str:
        return (
            "Return only JSON. To call a tool, return "
            '{"tool":"tool_name","args":{"key":"value"}}. '
            'To finish, return {"final":"summary"}. Available tools: '
            "list_files, read_file, search_text, apply_patch, run_command."
        )
