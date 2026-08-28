from ticketdesk.cli import format_ticket, parse_priority
from ticketdesk.models import Ticket


def test_parse_priority_is_case_insensitive() -> None:
    assert parse_priority(" URGENT ") == "urgent"


def test_format_ticket() -> None:
    ticket = Ticket(1, "VPN issue", "No connection", "alice", priority="high")
    assert format_ticket(ticket) == "#1 [high] VPN issue (open)"
