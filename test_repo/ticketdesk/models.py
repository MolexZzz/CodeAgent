from dataclasses import dataclass
from datetime import datetime

VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_STATUSES = {"open", "in_progress", "closed"}


@dataclass(slots=True)
class Ticket:
    id: int
    title: str
    description: str
    owner: str
    priority: str = "medium"
    status: str = "open"
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
