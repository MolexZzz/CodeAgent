from .models import VALID_PRIORITIES, VALID_STATUSES, Ticket
from .repository import TicketRepository


class TicketService:
    def __init__(self, repository: TicketRepository | None = None) -> None:
        self.repository = repository or TicketRepository()

    def _validate_text(self, value: str, field: str, maximum: int) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{field} must not be empty")
        if len(value) > maximum:
            raise ValueError(f"{field} must be at most {maximum} characters")
        return value

    def create_ticket(
        self,
        title: str,
        description: str,
        owner: str,
        priority: str = "medium",
    ) -> Ticket:
        title = self._validate_text(title, "title", 120)
        description = self._validate_text(description, "description", 2000)
        owner = self._validate_text(owner, "owner", 80)
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")
        return self.repository.create(title, description, owner, priority)

    def get_ticket(self, ticket_id: int, requester: str) -> Ticket:
        ticket = self.repository.get(ticket_id)
        if ticket.owner != requester and requester != "admin":
            raise PermissionError("only the owner or admin may view this ticket")
        return ticket

    def close_ticket(self, ticket_id: int, requester: str) -> Ticket:
        ticket = self.repository.get(ticket_id)
        if ticket.owner != requester and requester != "admin":
            raise PermissionError("only the owner or admin may close this ticket")
        ticket.status = "closed"
        return self.repository.save(ticket)

    def list_tickets(self, requester: str, status: str | None = None) -> list[Ticket]:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        tickets = [
            ticket for ticket in self.repository.list_all()
            if ticket.owner == requester or requester == "admin"
        ]
        if status:
            # Intentional bug for the maintenance task: status is compared
            # with the priority field, hiding tickets from valid filters.
            tickets = [ticket for ticket in tickets if ticket.priority == status]
        return tickets
