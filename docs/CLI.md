# CLI

## 1. Purpose

This document defines the three architectural decisions that bridge the design docs and the actual build:

1. How the repo is detected and the session is bootstrapped
2. How James is initialized with context before the first user message
3. How the worker is invoked mechanically

It also defines the turn loop and the overall process model.

---

## 2. Repo Detection

On startup, ACA determines the working repo using the following algorithm:

1. Start from the current working directory (`cwd`)
2. Walk up the directory tree looking for a `.git` folder
3. If a `.git` folder is found, use that directory as `repo_path`
4. If no `.git` folder is found anywhere in the tree, use `cwd` as `repo_path`

A `--repo <path>` flag can override this and set `repo_path` explicitly.

`repo_path` is stored in the `sessions` table at session creation and never changes for the lifetime of that session.

---

## 3. Session Startup Sequence

When the CLI is invoked, the following steps happen before the first user message is processed:

1. **Repo detection** — determine `repo_path` as described above
2. **Session creation** — generate a `session_id` (UUID), insert a row into `sessions` with `started_at`, `repo_path`, `model`, and `permission_mode`
3. **Repo bootstrap scan** — perform a lightweight read-only inspection of the repo:
   - list top-level directory structure
   - read `README.md` if present
   - read `.gitignore` if present
   - check for common project indicators (`package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, etc.) to identify the tech stack
4. **System prompt assembly** — build James's system prompt from:
   - core James identity and behavior instructions
   - repo context from the bootstrap scan (repo name, tech stack, top-level structure, README summary)
   - active permission mode
   - workspace rules (`.aca/` path, artifact constraints)
   - memory rules (context compaction behavior, hybrid retriever availability)
5. **Recent history load (optional)** — query SQLite for the last N turns from prior sessions on the same `repo_path`. If found, prepend them to the conversation history as Q&A pairs. This gives James cross-session continuity without replaying raw tool traces. N is configurable; default is 10 turns.
6. **REPL starts** — the CLI enters the interactive turn loop and waits for the first user message

The bootstrap scan is always read-only and uses no more than a small fixed number of tool calls. It never creates tasks or writes files.

---

## 4. Turn Loop

The turn loop is a synchronous REPL: read one user message, run James to completion, print the response, repeat.

### 4.1 Per-turn sequence

1. Read user input from stdin
2. Generate `turn_id` (UUID), insert row into `turns` with `session_id`, `turn_index`, `user_message`, `started_at`
3. Assemble the messages array:
   - system prompt (constant across all turns in the session)
   - conversation history: past turns as Q&A pairs only (no tool traces)
   - current user message
4. Run the **James agent loop** (see section 5)
5. When James produces a final response:
   - update `turns` row: `final_response`, `route`, `task_id` (if any), `ended_at`, token totals, total latency
   - print James's response to stdout
6. Update conversation history with the new Q&A pair
7. Loop back to step 1

### 4.2 Conversation history passed across turns

Only Q&A pairs cross turn boundaries — no tool traces, no intermediate reasoning. Each past turn contributes exactly two entries to the messages array:

```
{ role: "user",      content: <user_message> }
{ role: "assistant", content: <final_response> }
```

This is the Category B content that is eligible for soft removal under context pressure.

### 4.3 Context pressure handling

Before assembling the messages array for a new turn, check whether the estimated token count of the full history exceeds the configured compaction threshold. If it does:

1. Identify the oldest Q&A pair(s) in the history
2. Insert a row into `context_compactions` for each evicted pair
3. Remove those pairs from the in-memory history list
4. Proceed with the reduced history

This happens at the turn boundary, not mid-turn. Current-turn content is never touched.

---

## 5. James Agent Loop

The James agent loop is a standard LLM tool-use loop running in-process.

```
while True:
    make LLM call → record in llm_calls
    if stop_reason == "end_turn":
        final_response = response_text
        break
    if stop_reason == "tool_use":
        for each tool call in response:
            execute tool → record in tool_calls
            collect tool result
        append assistant message + tool results to messages
        continue
```

Each LLM call in this loop gets its own `llm_call_id` and `call_index`. Each tool invocation gets its own `tool_call_id`. All are linked to the same `turn_id`.

A maximum iteration limit (e.g. 20 LLM calls per turn) should be enforced to prevent runaway loops. If the limit is hit, James is forced to `JAMES_FINALIZE_RESPONSE` with whatever it has.

---

## 6. Worker Invocation Model

The worker is **a second in-process LLM call sequence** running under a different system prompt.

There are no subprocesses, threads, queues, or IPC mechanisms. The worker is invoked as a function call within the same turn.

### 6.1 How delegation happens

When James reaches `JAMES_WAIT_FOR_WORKER`:

1. James has already written `task.md`, `plan.md`, and `todo.md` to `.aca/active/<task-id>/`
2. The runtime calls `run_worker(task_id)` — a plain function
3. `run_worker` assembles a fresh worker context:
   - worker system prompt (different from James's — no user-facing role, scoped execution focus)
   - task package: contents of `task.md`, `plan.md`, `todo.md` injected directly into the initial messages
   - no James turn history, no James tool traces
4. The worker agent loop runs (same structure as James's loop in section 5) under the worker system prompt
5. When the worker reaches `WORKER_DONE`, `run_worker` returns
6. Control returns to James, which transitions to `JAMES_READ_WORKER_RESULT`

All LLM calls made during the worker run are recorded in `llm_calls` with `agent = 'worker'` and the same `turn_id` as the delegating James turn. This means the full turn — James setup, worker execution, James response — is one logical unit in the database.

### 6.2 The "completion signal"

There is no signal in the IPC sense. `run_worker` is a blocking function call. When it returns, the worker is done. James reads the result files and continues.

### 6.3 Worker context isolation

The worker receives only:
- its own system prompt
- the task package (task.md, plan.md, todo.md contents)
- its own tool call history within its own loop

The worker does not receive:
- James's system prompt
- James's conversation history
- James's tool traces

This matches the separation principle in `ARCHITECTURE.md` sections 10.4 and the memory separation rule in `STATE-MACHINE.md` section 5.4.

---

## 7. Challenger Invocation Model

Challenger follows the same pattern as the worker — a separate in-process LLM call with its own system prompt.

When James reaches `JAMES_CHALLENGER_REVIEW`:

1. The runtime calls `run_challenger(task_id)` — a plain function
2. Challenger receives:
   - challenger system prompt (bounded critic role)
   - current `plan.md` contents only
   - no James history, no worker history, no prior context
3. Challenger runs a single LLM call (no tool loop needed — it only reads and critiques)
4. Challenger returns its critique as text
5. James incorporates the critique and revises `plan.md` and/or `todo.md` before delegating

All challenger LLM calls are recorded in `llm_calls` with `agent = 'challenger'`.

The challenger receives only `plan.md` — a clean, fresh document with no surrounding context. This ensures the critique is uncontaminated by James's reasoning process.

---

## 8. Process Model Summary

```
aca (CLI process)
│
├── session bootstrap (read-only repo scan)
│
└── turn loop (REPL)
    │
    ├── James agent loop (LLM tool-use loop, in-process)
    │   │
    │   ├── [if delegated] run_challenger(task_id)  ← blocking, in-process
    │   │                  single LLM call, returns critique text
    │   │
    │   └── [if delegated] run_worker(task_id)      ← blocking, in-process
    │                      LLM tool-use loop, returns when WORKER_DONE
    │
    └── James reads results, produces final response
```

Everything runs in a single process. There is no concurrency in v1. One turn completes fully before the next begins.

---

## 9. Permission Mode

The permission mode is set at session start and applies for the entire session.

| Mode | Flag | Write tools | Execution tools |
|---|---|---|---|
| Read | `--read` | blocked | blocked |
| Edit | (default) | allowed with confirmation | blocked |
| Full | `--full` | allowed | allowed with confirmation |

The default mode is **Edit** with confirmation on writes. The permission mode is stored in `sessions.permission_mode`.
