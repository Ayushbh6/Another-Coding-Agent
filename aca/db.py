"""
Database initialisation and connection helpers for ACA.

Single source of truth for:
  - SQLite schema creation (sessions, turns, llm_calls, tool_calls, tasks,
    task_state_changes, context_compactions, FTS5 virtual tables, indexes)
  - WAL mode + foreign keys pragmas
  - open_db() factory — call once per session, pass the connection everywhere

Design rules (from DB_SCHEMA.md):
  - All PKs are UUIDs stored as TEXT.
  - All timestamps are Unix milliseconds stored as INTEGER.
  - JSON payloads stored as TEXT.
  - Nothing is ever deleted; archival is additive.
  - WAL mode for concurrent reads during active sessions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# ── Schema SQL ────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    repo_path       TEXT NOT NULL,
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,
    model           TEXT NOT NULL,
    permission_mode TEXT NOT NULL,
    user_name       TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id             TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id),
    turn_index          INTEGER NOT NULL,
    user_message        TEXT NOT NULL,
    final_response      TEXT,
    route               TEXT,
    task_id             TEXT,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    total_input_tokens  INTEGER,
    total_output_tokens INTEGER,
    total_latency_ms    INTEGER
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id         TEXT PRIMARY KEY,
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    agent           TEXT NOT NULL,
    call_index      INTEGER NOT NULL,
    system_prompt   TEXT NOT NULL,
    messages_json   TEXT NOT NULL,
    response_text   TEXT,
    stop_reason     TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    model           TEXT NOT NULL,
    temperature     REAL,
    started_at      INTEGER NOT NULL,
    error           TEXT,
    attempt_number  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id    TEXT PRIMARY KEY,
    llm_call_id     TEXT NOT NULL REFERENCES llm_calls(call_id),
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    agent           TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    input_json      TEXT NOT NULL,
    output_json     TEXT,
    success         INTEGER NOT NULL,
    error           TEXT,
    latency_ms      INTEGER,
    started_at      INTEGER NOT NULL,
    attempt_number  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    task_type       TEXT NOT NULL,
    delegation      TEXT NOT NULL,
    title           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      INTEGER NOT NULL,
    completed_at    INTEGER,
    archived_at     INTEGER,
    workspace_path  TEXT NOT NULL,
    archive_path    TEXT
);

CREATE TABLE IF NOT EXISTS task_state_changes (
    change_id       TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    from_state      TEXT NOT NULL,
    to_state        TEXT NOT NULL,
    agent           TEXT NOT NULL,
    changed_at      INTEGER NOT NULL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS context_compactions (
    compaction_id           TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id                 TEXT NOT NULL,
    agent                   TEXT NOT NULL,
    compacted_turn_id       TEXT NOT NULL REFERENCES turns(turn_id),
    context_tokens_before   INTEGER NOT NULL,
    context_tokens_after    INTEGER NOT NULL,
    compacted_at            INTEGER NOT NULL
);

-- FTS5 virtual tables for hybrid memory retrieval (search_memory)
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    turn_id UNINDEXED,
    user_message,
    final_response,
    content='turns',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS tool_calls_fts USING fts5(
    tool_call_id UNINDEXED,
    tool_name UNINDEXED,
    input_json,
    output_json,
    content='tool_calls',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS llm_calls_fts USING fts5(
    call_id UNINDEXED,
    agent UNINDEXED,
    response_text,
    content='llm_calls',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_turns_session           ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_turn          ON llm_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_agent         ON llm_calls(agent);
CREATE INDEX IF NOT EXISTS idx_tool_calls_turn         ON tool_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent        ON tool_calls(agent);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name    ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tasks_session           ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status            ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_task_state_changes_task ON task_state_changes(task_id);
CREATE INDEX IF NOT EXISTS idx_compactions_session     ON context_compactions(session_id);
"""


# ── Public API ────────────────────────────────────────────────────────────────

def open_db(db_path: str | Path) -> sqlite3.Connection:
    """
    Open (or create) the ACA SQLite database.

    Creates all tables, FTS virtual tables, and indexes on first run.
    Enables WAL mode and foreign-key enforcement.

    Returns a sqlite3.Connection with row_factory=sqlite3.Row so callers
    can access columns by name.

    Usage:
        db = open_db("data/aca.db")
        # pass db to BaseAgent, ToolRegistry.dispatch(), call_llm(), etc.
        db.close()
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Execute the full schema as a script; IF NOT EXISTS guards make it safe
    # to run on every open — new tables appear, existing ones are untouched.
    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    # Migration: add user_name column if it doesn't exist yet (existing DBs).
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_name TEXT")
        conn.commit()
    except Exception:  # noqa: BLE001
        pass  # column already exists

    return conn
