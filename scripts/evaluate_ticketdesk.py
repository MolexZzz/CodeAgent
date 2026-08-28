"""External acceptance checks for the TicketDesk demo repository.

Run this from the parent project, passing a copied TicketDesk workspace:
    python scripts/evaluate_ticketdesk.py D:\test_repo_run

The checked repository does not contain these assertions, so an agent cannot
make the score green by weakening its own public tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate_ticketdesk.py <TicketDesk workspace>")
        return 2

    workspace = Path(sys.argv[1]).resolve()
    if not (workspace / "ticketdesk").is_dir():
        print(f"Error: invalid TicketDesk workspace: {workspace}")
        return 2

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(test.stdout, end="")
    if test.stderr:
        print(test.stderr, file=sys.stderr, end="")
    if test.returncode != 0:
        print("External acceptance failed: repository tests did not pass.")
        return 1

    sys.path.insert(0, str(workspace))
    from ticketdesk.service import TicketService

    service = TicketService()
    closed = service.create_ticket("Old printer", "Cannot print", "alice", "low")
    urgent = service.create_ticket("VPN", "无法连接", "alice", "urgent")
    closed.status = "closed"
    service.repository.save(closed)
    urgent.status = "in_progress"
    service.repository.save(urgent)

    assert [ticket.title for ticket in service.list_tickets("alice", status="closed")] == ["Old printer"]
    assert service.list_tickets("alice", sort_by="priority", descending=True, page_size=1)[0].title == "VPN"
    assert service.reopen_ticket(closed.id, "alice").status == "open"

    try:
        service.get_ticket(urgent.id, "bob")
    except PermissionError:
        pass
    else:
        raise AssertionError("non-owner should not view another user's ticket")

    print("External acceptance passed: filtering, sorting, pagination, reopen, and authorization work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
