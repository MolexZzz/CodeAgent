from ticketdesk.service import TicketService


def test_create_ticket_normalizes_text() -> None:
    service = TicketService()
    ticket = service.create_ticket("  Printer offline ", "  Cannot print  ", "alice")
    assert ticket.title == "Printer offline"
    assert ticket.description == "Cannot print"


def test_owner_cannot_view_another_users_ticket() -> None:
    service = TicketService()
    ticket = service.create_ticket("VPN issue", "No connection", "alice")
    try:
        service.get_ticket(ticket.id, "bob")
    except PermissionError:
        pass
    else:
        raise AssertionError("non-owner should not be allowed to view the ticket")
