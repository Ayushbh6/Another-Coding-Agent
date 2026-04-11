

# TOOLS AND PERMISSIONS

## 1. Purpose

This document defines:
- what tools ACA (Another Coding Agent) has access to
- how those tools are used by James and workers
- what permissions are required
- what safety constraints must be enforced

This is a **critical file** because ACA is a coding agent that can read, write, and execute actions on a user's machine.

The goal is to make the system:
- safe by default
- transparent in behavior
- controllable by the user
- auditable at all times

---

## 2. Core Philosophy

### 2.1 Least privilege by default
The agent should start with minimal permissions.

### 2.2 Explicit over implicit
Dangerous actions must always be explicit and visible.

### 2.3 Read is cheap, write is controlled
Reading files is low risk. Writing files and executing commands must be controlled.

### 2.4 Repo-bounded behavior
The agent should operate only within the target repo and its workspace.

### 2.5 Full auditability
Every tool call must be logged in SQLite.

### 2.6 No silent destructive actions
Destructive operations must always require confirmation.

---

## 3. Tool Categories

ACA tools are grouped into the following categories:

### 3.1 Read Tools
Used to inspect the repo.

Examples:
- read_file
- list_files
- search_repo
- get_repo_summary

### 3.2 Write Tools
Used to modify files.

Examples:
- write_file
- update_file
- create_file
- apply_patch

### 3.3 Execution Tools (Future / Restricted)
Used to run commands.

Examples:
- run_command
- run_tests

### 3.4 Memory Tools
Used for context management and on-demand retrieval.

Examples:
- compact_context
- search_memory

### 3.5 Task / Workspace Tools
Used to manage `.aca/` files.

Examples:
- create_task_workspace
- write_task_file
- move_task_to_archive

---

## 4. Permission Modes

ACA should support different permission modes.

### 4.1 Read Mode
- can only use read tools
- no file writes
- no command execution

### 4.2 Edit Mode
- can read files
- can write files
- cannot execute arbitrary shell commands

### 4.3 Full Mode (Restricted)
- can read files
- can write files
- can execute approved commands

Default mode should be:
→ **Edit Mode with confirmation for writes**

---

## 5. Read Tool Rules

### Allowed
- reading any file inside repo
- listing directories
- semantic search via RAG

### Restricted
- reading system files outside repo
- reading sensitive directories

### Safety
- block access to paths like:
  - ~/.ssh
  - ~/.aws
  - system root directories

---

## 6. Write Tool Rules

### Allowed
- create new files inside repo
- update existing repo files
- modify `.aca/` workspace files

### Restricted
- writing outside repo root
- modifying system-level files

### Required behavior
Before applying changes:
- show diff (recommended)
- confirm action (unless auto-approve is enabled)

### Path safety
- block path traversal (`../`)
- enforce repo-root boundary

---

## 7. Execution Tool Rules (Strict)

Execution tools are high-risk and must be tightly controlled.

### Allowed (with approval)
- running tests
- running safe build commands

### Restricted
- destructive commands
- system-level commands

### Blocked by default
- rm -rf
- sudo commands
- chmod on sensitive files
- network exfiltration commands

### Recommendation
Execution tools should be:
- disabled in v1 OR
- strictly allowlisted

---

## 8. Memory Tool Rules

These tools control context and must be used carefully.

### 8.1 compact_context
- soft-removes the oldest past Q&A pairs from active context
- must only target past Q&A pairs (Category B)
- must never touch current-turn tool calls or results (Category A)
- does not delete data from the SQLite backend — all compacted pairs remain fully retrievable
- eviction is oldest-first; no agent judgment about importance is required

### 8.2 search_memory
- the unified hybrid retrieval tool covering all backend history
- combines keyword search (SQLite FTS5), BM25 relevance ranking, and vector similarity search (ChromaDB embeddings)
- accepts a natural language query and returns ranked results
- applies a recency bias when BM25 and vector scores are close, preferring more recent results
- covers past Q&A pairs, raw tool traces from prior turns, completed task workspaces, and all other durable history
- agents must use this tool whenever they need information that is no longer in active context, rather than guessing

### Critical rule
Compaction is always safe because:
1. only past Q&A pairs are ever removed from active context
2. those pairs are already persisted in SQLite before the turn ends
3. the hybrid retriever can recover any of them on demand

No pinning, checkpointing, or fact extraction step is required before compaction.

---

## 9. Workspace Tool Rules

These tools manage the `.aca/` workspace. The rules in this section are strict path and file-type guardrails that must be enforced by the tool layer, not left to agent discretion.

---

### 9.1 Folder structure and path rules

The only valid path pattern for task artifact files is:

```
.aca/active/<task-id>/<artifact-file>.md
```

Rules:
- `<task-id>` is a unique identifier assigned at task creation (e.g. `task-001`, `task-abc123`)
- The `<task-id>` subfolder must be created by `create_task_workspace` before any artifact can be written into it
- Agents may not write files directly into `.aca/` or `.aca/active/` — only into a named task subfolder
- Path traversal out of the task subfolder is blocked (`../` is never permitted)
- Writing into `.aca/summaries/` or `.aca/logs/` is reserved for system-level operations only, not agent tool calls

---

### 9.2 Permitted file types

Only markdown files (`.md`) may be written inside a task subfolder.

No other file types are permitted inside `.aca/active/<task-id>/`. This means:
- no `.py`, `.json`, `.yaml`, `.txt`, or any non-markdown file
- no binary files
- no hidden files (`.dotfile`)

---

### 9.3 Permitted artifact filenames

Only the following filenames are valid inside a task subfolder:

| Filename | Purpose |
|---|---|
| `task.md` | Task definition — required for all non-trivial tasks |
| `plan.md` | Execution plan — required for delegated tasks |
| `todo.md` | Step-by-step todo list |
| `findings.md` | Worker output for analysis tasks |
| `output.md` | Worker output for implementation tasks |

Any attempt to write a file with a name not on this list inside a task subfolder must be blocked.

---

### 9.4 Tool routing rules for `.aca/` writes

The general write tools (`write_file`, `create_file`, `update_file`, `apply_patch`) are **blocked from writing into `.aca/` entirely**.

All writes into `.aca/active/<task-id>/` must go through the dedicated workspace tool:
- `write_task_file(task_id, filename, content)` — the only permitted way to write or update an artifact file inside a task subfolder

This separation ensures that task artifact writes are always validated against the permitted filename list and path rules before being executed.

---

### 9.5 Task folder creation rule

A task subfolder may only be created by `create_task_workspace(task_id)`.

Agents may not manually create folders inside `.aca/` using general file or shell tools.

---

### 9.6 Lifecycle

- Task subfolders are created at task init
- They remain in `.aca/active/` for the duration of the task and for approximately 12 hours after completion
- After TTL expiry, `move_task_to_archive(task_id)` moves the folder to user-level archival storage outside the repo
- The repo `.aca/` folder should never accumulate stale task subfolders

---

## 10. Secret Protection

The agent must never expose or misuse secrets.

### Sensitive files
- .env
- API keys
- tokens
- credentials

### Rules
- detect sensitive patterns
- require explicit confirmation before reading
- never print secrets in output unless user explicitly asks

---

## 11. Logging and Audit

Every tool call must be logged in SQLite.

### Log fields
- tool name
- inputs
- outputs
- success/failure
- timestamp
- agent (James / worker)

This enables:
- debugging
- replay
- trust

---

## 12. Git Safety (Recommended)

### Before major changes
- show diff
- suggest commit

### Optional features
- auto-commit after task
- rollback support

---

## 13. Approval System

The CLI should support user approval flows.

### Modes
- auto-approve (trusted usage)
- confirm-on-write
- confirm-on-command

### UX
- clear prompts
- minimal friction
- no silent actions

---

## 14. Agent Behavior Constraints

James and workers must follow:

### 14.1 Do not overuse tools
- unnecessary tool calls are wasteful

### 14.2 Respect tool budgets
- especially in chat mode

### 14.3 Prefer read before write
- inspect before modifying

### 14.4 Keep operations scoped
- do not wander outside task boundaries

---

## 15. Summary

ACA tools must be:
- safe
- controlled
- transparent
- auditable

Core rules:
- read freely within repo
- write carefully with boundaries
- write task artifacts only into `.aca/active/<task-id>/`, only as `.md` files, only with permitted filenames, only via `write_task_file`
- general write tools are blocked from writing into `.aca/`
- execute rarely and safely
- compact only past Q&A pairs, oldest-first, never current-turn state
- retrieve anything out of context via the hybrid memory tool
- log everything

This ensures ACA is powerful but never reckless.