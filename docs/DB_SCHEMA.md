# DB SCHEMA

## 1. Purpose

This document defines the complete SQLite database schema for ACA's durable backend (`data/aca.db`).

The design goal is **full retraceability**: given the database alone, it must be possible to reconstruct every step of every session — every user message, every LLM call, every tool invocation, every success, every failure, every retry, every token, every latency measurement, every task state change, and every context compaction event.

There should be no black box anywhere.

---

## 2. Design Principles

- **Event-sourced**: every significant event is a row. Nothing is updated in place except stable entity status fields (e.g. task status, session ended_at).
- **Nothing is deleted**: rows are never removed from the database. Archival is additive.
- **Everything is timestamped**: every table has at least one Unix timestamp in milliseconds.
- **Cross-referenced**: every row carries enough foreign keys to trace it back to its session, turn, and agent.
- **Retry-aware**: all call and tool tables carry `attempt_number` so failed and retried attempts are distinguishable from successful ones.

---

## 3. General Conventions

- All primary keys are UUIDs stored as TEXT.
- All timestamps are Unix milliseconds stored as INTEGER.
- All JSON payloads (messages, inputs, outputs, results) are stored as TEXT.
- Booleans are stored as INTEGER: 1 = true, 0 = false.
- Nullable columns are explicitly noted.

---

## 4. Tables

---

### 4.1 `sessions`

One row per ACA CLI invocation.

```sql
CREATE TABLE sessions (
    session_id      TEXT PRIMARY KEY,
    repo_path       TEXT NOT NULL,
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,               -- NULL while session is active
    model           TEXT NOT NULL,         -- default model for this session
    permission_mode TEXT NOT NULL          -- 'read' | 'edit' | 'full'
);
```

---

### 4.2 `turns`

One row per user turn. A turn begins when the user sends a message and ends when James delivers the final response.

```sql
CREATE TABLE turns (
    turn_id             TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id),
    turn_index          INTEGER NOT NULL,  -- sequential within session, 0-based
    user_message        TEXT NOT NULL,
    final_response      TEXT,              -- NULL until James finishes
    route               TEXT,              -- 'chat' | 'analysis_simple' | 'analysis_delegated' | 'implement_simple' | 'implement_delegated' — NULL until routing resolves
    task_id             TEXT,              -- NULL for chat turns
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,           -- NULL until turn completes
    total_input_tokens  INTEGER,           -- sum across all llm_calls in this turn
    total_output_tokens INTEGER,           -- sum across all llm_calls in this turn
    total_latency_ms    INTEGER            -- wall time from user message to final response
);
```

---

### 4.3 `llm_calls`

One row per individual LLM API call. A single turn typically contains multiple LLM calls (one per agent reasoning step).

```sql
CREATE TABLE llm_calls (
    call_id         TEXT PRIMARY KEY,
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    agent           TEXT NOT NULL,         -- 'james' | 'worker' | 'challenger'
    call_index      INTEGER NOT NULL,      -- sequential within turn, 0-based
    system_prompt   TEXT NOT NULL,         -- full system prompt sent
    messages_json   TEXT NOT NULL,         -- full messages array as JSON
    response_text   TEXT,                  -- full raw LLM response text, NULL if call failed
    stop_reason     TEXT,                  -- 'end_turn' | 'tool_use' | 'max_tokens' | 'error' etc.
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    model           TEXT NOT NULL,         -- actual model used for this call
    temperature     REAL,
    started_at      INTEGER NOT NULL,
    error           TEXT,                  -- NULL if call succeeded
    attempt_number  INTEGER NOT NULL DEFAULT 1
);
```

---

### 4.4 `tool_calls`

One row per tool invocation. Each LLM call with `stop_reason = 'tool_use'` produces one or more tool call rows.

```sql
CREATE TABLE tool_calls (
    tool_call_id    TEXT PRIMARY KEY,
    llm_call_id     TEXT NOT NULL REFERENCES llm_calls(call_id),
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    agent           TEXT NOT NULL,         -- 'james' | 'worker' | 'challenger'
    tool_name       TEXT NOT NULL,         -- e.g. 'read_file', 'write_task_file', 'search_memory'
    input_json      TEXT NOT NULL,         -- full tool input as JSON
    output_json     TEXT,                  -- full tool output as JSON, NULL if call failed
    success         INTEGER NOT NULL,      -- 1 | 0
    error           TEXT,                  -- NULL if success
    latency_ms      INTEGER,
    started_at      INTEGER NOT NULL,
    attempt_number  INTEGER NOT NULL DEFAULT 1
);
```

---

### 4.5 `tasks`

One row per task created. A task row is inserted when James writes `task.md`.

```sql
CREATE TABLE tasks (
    task_id         TEXT PRIMARY KEY,      -- matches folder name under .aca/active/
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id         TEXT NOT NULL REFERENCES turns(turn_id),
    task_type       TEXT NOT NULL,         -- 'analysis' | 'implement'
    delegation      TEXT NOT NULL,         -- 'simple' | 'delegated'
    title           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',  -- 'active' | 'complete' | 'archived'
    created_at      INTEGER NOT NULL,
    completed_at    INTEGER,               -- NULL until complete
    archived_at     INTEGER,               -- NULL until archived
    workspace_path  TEXT NOT NULL,         -- full path to .aca/active/<task-id>/
    archive_path    TEXT                   -- NULL until archived
);
```

---

### 4.6 `task_state_changes`

One row per status transition on a task. Gives complete task lifecycle history.

```sql
CREATE TABLE task_state_changes (
    change_id       TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    from_state      TEXT NOT NULL,
    to_state        TEXT NOT NULL,
    agent           TEXT NOT NULL,         -- who triggered the change
    changed_at      INTEGER NOT NULL,
    notes           TEXT                   -- NULL or brief context
);
```

---

### 4.7 `context_compactions`

One row each time a past Q&A pair is evicted from active context.

```sql
CREATE TABLE context_compactions (
    compaction_id           TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL REFERENCES sessions(session_id),
    turn_id                 TEXT NOT NULL,  -- turn during which compaction happened
    agent                   TEXT NOT NULL,  -- 'james' | 'worker'
    compacted_turn_id       TEXT NOT NULL REFERENCES turns(turn_id),  -- the Q&A pair evicted
    context_tokens_before   INTEGER NOT NULL,
    context_tokens_after    INTEGER NOT NULL,
    compacted_at            INTEGER NOT NULL
);
```

---

## 5. FTS5 Virtual Tables for Hybrid Retrieval

These virtual tables power the keyword and BM25 search component of `search_memory`.

```sql
-- Full-text search over conversation Q&A pairs
CREATE VIRTUAL TABLE turns_fts USING fts5(
    turn_id UNINDEXED,
    user_message,
    final_response,
    content='turns',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Full-text search over tool call inputs and outputs
CREATE VIRTUAL TABLE tool_calls_fts USING fts5(
    tool_call_id UNINDEXED,
    tool_name UNINDEXED,
    input_json,
    output_json,
    content='tool_calls',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Full-text search over raw LLM responses
CREATE VIRTUAL TABLE llm_calls_fts USING fts5(
    call_id UNINDEXED,
    agent UNINDEXED,
    response_text,
    content='llm_calls',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
```

BM25 ranking is available natively via `bm25(turns_fts)`, `bm25(tool_calls_fts)`, and `bm25(llm_calls_fts)` in ORDER BY clauses.

---

## 6. ChromaDB Vector Layer

ChromaDB sits alongside SQLite in `data/chroma/` and stores vector embeddings for semantic search.

### Collections

| Collection | What is embedded | Document ID format |
|---|---|---|
| `turns` | `user_message + final_response` concatenated | `turn_id` |
| `tool_calls` | `input_json + output_json` concatenated | `tool_call_id` |

### Linking back to SQLite

Every Chroma document ID is the corresponding SQLite primary key (`turn_id` or `tool_call_id`). A vector search hit immediately gives you the key needed to fetch the full row from SQLite.

### How `search_memory` combines the layers

1. Run FTS5 BM25 query across `turns_fts` and `tool_calls_fts` → get ranked keyword hits
2. Run vector similarity query in ChromaDB → get ranked semantic hits
3. Merge results, deduplicate by ID, apply recency bias as a tiebreaker
4. Return top-N results with full content fetched from SQLite

---

## 7. Indexes

```sql
-- Fast turn lookups by session
CREATE INDEX idx_turns_session ON turns(session_id);

-- Fast LLM call lookups by turn and agent
CREATE INDEX idx_llm_calls_turn ON llm_calls(turn_id);
CREATE INDEX idx_llm_calls_agent ON llm_calls(agent);

-- Fast tool call lookups by turn, agent, and tool name
CREATE INDEX idx_tool_calls_turn ON tool_calls(turn_id);
CREATE INDEX idx_tool_calls_agent ON tool_calls(agent);
CREATE INDEX idx_tool_calls_tool_name ON tool_calls(tool_name);

-- Fast task lookups by session and status
CREATE INDEX idx_tasks_session ON tasks(session_id);
CREATE INDEX idx_tasks_status ON tasks(status);

-- Fast state change lookups by task
CREATE INDEX idx_task_state_changes_task ON task_state_changes(task_id);

-- Fast compaction lookups by session and agent
CREATE INDEX idx_compactions_session ON context_compactions(session_id);
```

---

## 8. WAL Mode

The database must be opened in WAL (Write-Ahead Logging) mode for concurrent read access during active sessions:

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
```

---

## 9. What this enables

Given only `aca.db`, it is possible to:

- Reconstruct the exact sequence of every turn in a session
- See every LLM call: full prompt, full response, token counts, latency, model, retries
- See every tool call: inputs, outputs, success or failure, retries
- Trace a task from creation through state changes to archival
- See which Q&A pairs were compacted out of context during which turns
- Query any past information using keyword, BM25, or vector search via `search_memory`
- Identify performance bottlenecks via latency fields on every call
- Audit every agent action by filtering on the `agent` column

There is no black box.
