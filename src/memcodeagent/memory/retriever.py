from __future__ import annotations

from dataclasses import dataclass, field

from memcodeagent.memory.schema import MemoryItem
from memcodeagent.workspace import Workspace


@dataclass(slots=True)
class RetrievalContext:
    items: list[MemoryItem] = field(default_factory=list)

    def to_prompt(self) -> str:
        if not self.items:
            return "(no retrieved context yet)"
        return "\n\n".join(f"[{item.kind}] {item.text}" for item in self.items)

    def to_display(self) -> str:
        if not self.items:
            return "(no retrieved context yet)"
        lines = []
        for item in self.items:
            location = f" {item.path}" if item.path else ""
            lines.append(f"- {item.kind}{location} score={item.score:.2f} reason={item.reason}\n  {item.text}")
        return "\n".join(lines)


class SimpleRetriever:
    """Initial retrieval stub; will evolve into layered memory and grouped code retrieval."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def retrieve(self, query: str) -> RetrievalContext:
        del query
        return RetrievalContext()

    def index_workspace(self) -> str:
        file_count = sum(
            1 for path in self.workspace.root.glob("**/*") if path.is_file() and not self.workspace.should_ignore(path)
        )
        return f"Indexed workspace placeholder: {file_count} files discovered."

    def remember_task(self, task: str, summary: str) -> None:
        del task, summary
