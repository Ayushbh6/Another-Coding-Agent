from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from aca.cli import _build_james, _cmd_model, _cmd_new, _cmd_thinking
from aca.console import AgentConsole
from aca.db import open_db
from aca.llm.models import OpenAIModels, OpenRouterModels
from aca.llm.providers import ProviderName
from aca.tools import build_registry
from aca.tools.registry import PermissionMode


def test_build_registry_registers_standard_tools() -> None:
    registry = build_registry()

    read_names = set(registry.list_names(PermissionMode.READ))
    edit_names = set(registry.list_names(PermissionMode.EDIT))
    full_names = set(registry.list_names(PermissionMode.FULL))

    assert {"read_file", "read_files", "list_files", "search_repo", "get_file_outline", "search_memory"} <= read_names
    assert {"write_file", "edit_file", "update_file", "multi_update_file", "create_task_workspace", "write_task_file"} <= edit_names
    assert {"run_command", "run_tests"} <= full_names


def test_agent_visible_schemas_hide_legacy_read_and_exact_edit_tools() -> None:
    registry = build_registry()
    edit_schema_names = {schema["function"]["name"] for schema in registry.get_schemas(PermissionMode.EDIT)}

    assert "read_files" in edit_schema_names
    assert "edit_file" in edit_schema_names
    assert "read_file" not in edit_schema_names
    assert "update_file" not in edit_schema_names
    assert "multi_update_file" not in edit_schema_names


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
    assert "read_files" in read_names
    assert "search_repo" in read_names
    assert "create_task_workspace" in set(james._registry.list_names(PermissionMode.EDIT))
    assert james._stream is True


def test_cmd_model_switches_live_agent_without_resetting_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture = StringIO()
    console = Console(file=capture, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")
    db = open_db(tmp_path / "aca.db")
    db.execute(
        """
        INSERT INTO sessions (session_id, repo_path, started_at, model, permission_mode, user_name)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("session-123", str(tmp_path), 0, OpenRouterModels.minimax_2_7, "edit", "Ayush"),
    )
    db.commit()

    james = _build_james(
        session_id="session-123",
        repo_path=tmp_path,
        model=OpenRouterModels.minimax_2_7,
        thinking=False,
        db=db,
        agent_console=agent_console,
        repo_context="Repo path: test",
        user_name="Ayush",
    )
    james._history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    james._history_meta = [{"turn_id": "turn-1"}]
    state = {"model": OpenRouterModels.minimax_2_7, "thinking": False}

    monkeypatch.setattr("aca.cli.Prompt.ask", lambda *args, **kwargs: "6")

    _cmd_model(console, state, james, db, "session-123")

    assert state["model"] == OpenAIModels.GPT_5
    assert james._model == OpenAIModels.GPT_5
    assert james._provider == ProviderName.OPENAI
    assert james._history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    assert james._history_meta == [{"turn_id": "turn-1"}]
    row = db.execute(
        "SELECT model FROM sessions WHERE session_id = ?",
        ("session-123",),
    ).fetchone()
    assert row["model"] == OpenAIModels.GPT_5


def test_cmd_thinking_updates_state_and_live_agent(tmp_path: Path) -> None:
    capture = StringIO()
    console = Console(file=capture, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    james = _build_james(
        session_id="session-123",
        repo_path=tmp_path,
        model=OpenRouterModels.minimax_2_7,
        thinking=False,
        db=None,
        agent_console=agent_console,
        repo_context="Repo path: test",
        user_name="Ayush",
    )
    state = {"model": OpenRouterModels.minimax_2_7, "thinking": False}

    _cmd_thinking(console, state, james)
    assert state["thinking"] is True
    assert james._thinking is True

    _cmd_thinking(console, state, james)
    assert state["thinking"] is False
    assert james._thinking is False


def test_cmd_new_ends_current_session_and_requests_restart(tmp_path: Path) -> None:
    capture = StringIO()
    console = Console(file=capture, force_terminal=False, color_system=None, highlight=False)
    db = open_db(tmp_path / "aca.db")
    db.execute(
        """
        INSERT INTO sessions (session_id, repo_path, started_at, model, permission_mode)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("session-123", str(tmp_path), 0, OpenRouterModels.minimax_2_7, "edit"),
    )
    db.commit()

    restart = _cmd_new(console, db, "session-123")

    assert restart is True
    row = db.execute(
        "SELECT ended_at FROM sessions WHERE session_id = ?",
        ("session-123",),
    ).fetchone()
    assert row["ended_at"] is not None
