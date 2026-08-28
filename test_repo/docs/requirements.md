# TicketDesk Maintenance Requirements

## Context

TicketDesk is an internal support-ticket service used by a small operations
team. The current version supports creating tickets, viewing a ticket,
closing a ticket, and listing a user's inbox.

## Maintenance request

Please prepare the next release of TicketDesk:

1. Fix the ticket-list status filter. `list_tickets(requester, status=...)`
   must filter by `Ticket.status`, not by priority. Existing callers that omit
   `status` must keep the same behavior.
2. Add priority sorting and pagination to the inbox:
   `list_tickets(requester, status=None, sort_by="created_at", descending=False,
   page=1, page_size=20)`.
   - Valid sort fields are `created_at`, `priority`, and `title`.
   - Priority order is `urgent > high > medium > low`.
   - Invalid page values or sort fields should raise `ValueError`.
3. Add `reopen_ticket(ticket_id, requester)` and preserve the owner/admin
   authorization rule used by `get_ticket` and `close_ticket`.
4. Review the validation code in `service.py` and `cli.py`. Remove duplicated
   validation logic without changing the public helper functions.
5. Add regression and boundary tests for filtering, sorting, pagination,
   reopening, authorization, and invalid input.
6. Update `README.md` with the new inbox behavior and run the complete test
   suite.

## Acceptance criteria

- Existing tests continue to pass.
- A user can only list, view, close, or reopen their own tickets; `admin` can
  manage all tickets.
- No unrelated files are changed.
- The final test command exits successfully.
