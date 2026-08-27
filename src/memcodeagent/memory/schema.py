from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MemoryItem:
    kind: str
    text: str
    path: Path | None = None
    score: float = 0.0
    reason: str = ""
