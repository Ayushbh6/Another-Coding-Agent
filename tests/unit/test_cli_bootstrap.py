from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from aca.cli import _build_james
from aca.console import AgentConsole
from aca.tools import build_registry
from aca.tools.registry import PermissionMode


def test_build_registry_registers_standard_tools() -> None:
    registry = build_registry()

    read_names = set(registry.list_names(PermissionMode.READ))
    edit_names = set(registry.list_names(PermissionMode.EDIT))
    full_names = set(registry.list_names(PermissionMode.FULL))

    assert {"read_file", "list_files", "search_repo", "get_file_outline", "search_memory"} <= read_names
    assert {"write_file", "update_file", "multi_update_file", "create_task_workspace", "write_task_file"} <= edit_names
    assert {"run_command", "run_tests"} <= full_names


def test_build_james_starts_with_registered_tools(tmp_path: Path) -> None:
    capture = StringIO()
    console = Console(file=capture, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    james = _build_james(
        session_id="session-123",
        repo_path=tmp_path,
        model="minimax/minimax-m2.7:nitro",
        thinking=False,
        db=None,
        agent_console=agent_console,
        repo_context="Repo path: test",
        user_name="Ayush",
    )

    read_names = set(james._registry.list_names(PermissionMode.READ))
    assert "read_file" in read_names
    assert "search_repo" in read_names
    assert "create_task_workspace" in set(james._registry.list_names(PermissionMode.EDIT))
    assert james._stream is True