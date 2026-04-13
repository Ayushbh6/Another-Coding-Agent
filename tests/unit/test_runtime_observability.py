from __future__ import annotations

import json
from pathlib import Path

from aca.agents.base_agent import BaseAgent
from aca.cli import _TASK_TTL_MS, _archive_expired_tasks
from aca.llm.client import LLMResponse
from aca.db import open_db
from aca.tools import build_registry
from aca.tools.registry import PermissionMode


def _seed_db(db, repo_root: Path) -> tuple[str, str, str]:
    session_id = "session-1"
    turn_id = "turn-1"
    call_id = "llm-call-1"
    db.execute(
        """
        INSERT INTO sessions (session_id, repo_path, started_at, model, permission_mode)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, str(repo_root), 0, "test-model", "edit"),
    )
    db.execute(
        """
        INSERT INTO turns (turn_id, session_id, turn_index, user_message, started_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (turn_id, session_id, 0, "hello", 0),
    )
    db.execute(
        """
        INSERT INTO llm_calls (
            call_id, turn_id, session_id, agent, call_index, system_prompt,
            messages_json, response_text, stop_reason, input_tokens, output_tokens,
            latency_ms, model, started_at, attempt_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (call_id, turn_id, session_id, "james", 0, "", "[]", "", "tool_use", 0, 0, 0, "test-model", 0, 1),
    )
    db.commit()
    return session_id, turn_id, call_id


def test_dispatch_injects_runtime_context_and_logs_tool_call(tmp_path: Path) -> None:
    db = open_db(tmp_path / "aca.db")
    session_id, turn_id, call_id = _seed_db(db, tmp_path)
    registry = build_registry()

    tool_call = {
        "id": "tool-1",
        "type": "function",
        "function": {"name": "create_task_workspace", "arguments": json.dumps({"task_id": "task-001"})},
    }
    result = registry.dispatch(
        tool_call=tool_call,
        mode=PermissionMode.EDIT,
        injected_kwargs={
            "repo_root": str(tmp_path),
            "db": db,
            "session_id": session_id,
            "turn_id": turn_id,
        },
        db=db,
        llm_call_id=call_id,
        turn_id=turn_id,
        session_id=session_id,
        agent="james",
    )

    assert result.success is True
    task_row = db.execute("SELECT task_id, workspace_path FROM tasks").fetchone()
    assert tuple(task_row) == ("task-001", ".aca/active/task-001")

    tool_row = db.execute("SELECT tool_name, success FROM tool_calls").fetchone()
    assert tuple(tool_row) == ("create_task_workspace", 1)


class _DummyAgent(BaseAgent):
    def agent_name(self) -> str:
        return "dummy"

    def system_prompt(self) -> str:
        return "test"

    def _turn_metadata(self) -> dict[str, str]:
        return {"route": "analysis_simple", "task_id": "task-123"}


def test_turn_end_persists_route_and_task_id(tmp_path: Path, monkeypatch) -> None:
    db = open_db(tmp_path / "aca.db")
    db.execute(
        """
        INSERT INTO sessions (session_id, repo_path, started_at, model, permission_mode)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("session-1", str(tmp_path), 0, "test-model", "read"),
    )
    db.commit()

    agent = _DummyAgent(
        registry=build_registry(),
        session_id="session-1",
        db=db,
        repo_root=str(tmp_path),
        stream=False,
    )

    monkeypatch.setattr(
        "aca.agents.base_agent.call_llm",
        lambda **kwargs: LLMResponse(
            call_id="call-1",
            content="done",
            tool_calls=[],
            stop_reason="end_turn",
            thinking_blocks=[],
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            model="test-model",
            raw=None,
        ),
    )

    agent.run_turn("hello")
    row = db.execute("SELECT route, task_id FROM turns").fetchone()
    assert tuple(row) == ("analysis_simple", "task-123")


def test_archive_expired_tasks_moves_completed_workspace_to_users_root(tmp_path: Path, monkeypatch) -> None:
    db = open_db(tmp_path / "aca.db")
    session_id, turn_id, _call_id = _seed_db(db, tmp_path)

    task_dir = tmp_path / ".aca" / "active" / "task-001"
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("task_id: task-001\n", encoding="utf-8")

    db.execute(
        """
        INSERT INTO tasks (
            task_id, session_id, turn_id, task_type, delegation, title, status, created_at, completed_at, workspace_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-001",
            session_id,
            turn_id,
            "analysis",
            "simple",
            "Task",
            "completed",
            0,
            _TASK_TTL_MS - 1,
            ".aca/active/task-001",
        ),
    )
    db.commit()

    archive_base = tmp_path / ".users-root"
    monkeypatch.setattr("aca.cli._task_archive_base", lambda _repo_path: archive_base)
    monkeypatch.setattr("time.time", lambda: ((_TASK_TTL_MS * 2) / 1000))

    archived = _archive_expired_tasks(tmp_path, db)

    assert archived == ["task-001"]
    assert not task_dir.exists()
    assert (archive_base / "task-001" / "task.md").exists()
