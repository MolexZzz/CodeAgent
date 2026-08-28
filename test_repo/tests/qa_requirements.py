import pytest

from ticketdesk.service import TicketService


def seeded_service() -> TicketService:
    service = TicketService()
    first = service.create_ticket("Printer", "Printer is offline", "alice", "low")
    second = service.create_ticket("VPN", "VPN is unavailable", "alice", "urgent")
    for ticket_id, status in ((first.id, "closed"), (second.id, "in_progress")):
        ticket = service.repository.get(ticket_id)
        ticket.status = status
        service.repository.save(ticket)
    return service


def test_status_filter_returns_matching_status() -> None:
    service = seeded_service()
    tickets = service.list_tickets("alice", status="closed")
    assert [ticket.title for ticket in tickets] == ["Printer"]


def test_list_supports_priority_sorting_and_pagination() -> None:
    service = seeded_service()
    tickets = service.list_tickets("alice", sort_by="priority", descending=True, page=1, page_size=1)
    assert [ticket.title for ticket in tickets] == ["VPN"]


def test_reopen_ticket_restores_open_status() -> None:
    service = seeded_service()
    ticket = service.list_tickets("alice")[0]
    ticket.status = "closed"
    service.repository.save(ticket)
    reopened = service.reopen_ticket(ticket.id, "alice")
    assert reopened.status == "open"


@pytest.mark.parametrize("page,page_size", [(0, 10), (1, 0), (-1, 10)])
def test_invalid_pagination_is_rejected(page: int, page_size: int) -> None:
    service = TicketService()
    with pytest.raises(ValueError):
        service.list_tickets("alice", page=page, page_size=page_size)
