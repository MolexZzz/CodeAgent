from pathlib import Path

import pytest

from memcodeagent.workspace import Workspace


def test_resolve_inside_accepts_workspace_file(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    assert workspace.resolve_inside("example.py") == tmp_path / "example.py"


def test_resolve_inside_rejects_parent_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    with pytest.raises(ValueError):
        workspace.resolve_inside("../outside.py")


def test_safe_command_blocks_destructive_commands(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    with pytest.raises(ValueError):
        workspace.ensure_safe_command("git reset --hard HEAD")


def test_workspace_protects_sensitive_and_runtime_write_paths(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    for path in (".env", ".memcode/session.json", ".git/config"):
        with pytest.raises(ValueError):
            workspace.ensure_writable_path(path)
