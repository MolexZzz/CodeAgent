from dataclasses import dataclass
from datetime import datetime

VALID_STATUSES = {"todo", "doing", "done"}
VALID_PRIORITIES = {"low", "medium", "high"}


@dataclass(slots=True)
class Task:
    id: int
    title: str
    status: str = "todo"
    priority: str = "medium"
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
