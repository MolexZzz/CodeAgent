from dataclasses import replace

from .models import Ticket


class TicketRepository:
    def __init__(self) -> None:
        self._tickets: dict[int, Ticket] = {}
        self._next_id = 1

    def create(self, title: str, description: str, owner: str, priority: str) -> Ticket:
        ticket = Ticket(self._next_id, title, description, owner, priority)
        self._tickets[ticket.id] = ticket
        self._next_id += 1
        return replace(ticket)

    def get(self, ticket_id: int) -> Ticket:
        if ticket_id not in self._tickets:
            raise KeyError(f"ticket {ticket_id} not found")
        return replace(self._tickets[ticket_id])

    def list_all(self) -> list[Ticket]:
        return [replace(ticket) for ticket in self._tickets.values()]

    def save(self, ticket: Ticket) -> Ticket:
        if ticket.id not in self._tickets:
            raise KeyError(f"ticket {ticket.id} not found")
        self._tickets[ticket.id] = replace(ticket)
        return replace(ticket)
