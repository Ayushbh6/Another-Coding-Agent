from __future__ import annotations

import pytest

from aca.console import AgentConsole
from aca.tools.execution import _check_command_safety


def test_check_command_safety_allows_simple_read_only_git() -> None:
    _check_command_safety("git status --short")


def test_check_command_safety_blocks_destructive_git() -> None:
    with pytest.raises(ValueError, match="read-only git subcommands"):
        _check_command_safety("git reset --hard")


def test_check_command_safety_blocks_shell_chaining() -> None:
    with pytest.raises(ValueError, match="single direct command"):
        _check_command_safety("python -m pytest && git status")


def test_check_command_safety_blocks_unknown_executable() -> None:
    with pytest.raises(ValueError, match="safe allowlist"):
        _check_command_safety("bash script.sh")


def test_quiet_console_shows_run_command_text() -> None:
    from io import StringIO
    from rich.console import Console

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.tool_call("run_command", {"command": "git status --short"})
    agent_console.tool_result(
        "run_command",
        success=True,
        latency_ms=5,
        output={"command": "git status --short", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
    )

    rendered = buffer.getvalue()
    assert "Ran command: git status --short" in rendered
