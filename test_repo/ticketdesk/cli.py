from .models import VALID_PRIORITIES
from .service import TicketService


def parse_priority(value: str) -> str:
    value = value.strip().lower()
    if value not in VALID_PRIORITIES:
        raise ValueError(f"invalid priority: {value}")
    return value


def format_ticket(ticket) -> str:
    return f"#{ticket.id} [{ticket.priority}] {ticket.title} ({ticket.status})"


def render_inbox(service: TicketService, requester: str, status: str | None = None) -> list[str]:
    return [format_ticket(ticket) for ticket in service.list_tickets(requester, status)]
