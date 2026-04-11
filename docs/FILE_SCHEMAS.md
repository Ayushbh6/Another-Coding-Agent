# FILE SCHEMAS

## 1. Purpose

This document defines the exact format and required fields for every task artifact file that ACA writes into `.aca/active/<task-id>/`.

These files serve two purposes:
- **Machine-readable**: the header fields must be parseable by the system to extract routing information, task state, and cross-references to the database
- **Human-readable**: the body content is free-form markdown so the user can inspect what the agent is doing at any point

---

## 2. General Rules

- Every artifact file is a markdown file with a structured header block followed by free-form markdown sections
- The header block uses simple `key: value` lines — no YAML frontmatter delimiters, no special syntax
- Header fields must appear at the top of the file, before any markdown headings
- The `task_id` field must appear in every artifact file so any file can be independently cross-referenced to the `tasks` table in SQLite
- Files are written by agents using `write_task_file(task_id, filename, content)` only
- Only the filenames defined in this document are permitted inside a task subfolder

---

## 3. `task.md`

**Written by**: James, during `JAMES_TASK_INIT`
**Required for**: all non-trivial tasks (analysis simple, analysis delegated, implement simple, implement delegated)

This is the routing declaration file. The `type` and `delegation` fields are what the system reads to determine the task route. James writes these as a natural consequence of deciding how to handle the request — routing is internalized by the LLM, not enforced by an external classifier.

### Format

```
task_id: <task-id>
type: analysis | implement
delegation: simple | delegated
status: active
created_at: <ISO 8601 timestamp>
session_id: <uuid>
turn_id: <uuid>

# Task: <short title>

## Request

<The original user request, copied verbatim>

## Context

<James's brief summary of what repo context is relevant — what files were inspected, what was found>

## Scope

<What is in scope for this task. What is explicitly out of scope.>
```

### Field rules

| Field | Required | Values |
|---|---|---|
| `task_id` | yes | matches `.aca/active/` subfolder name |
| `type` | yes | `analysis` or `implement` |
| `delegation` | yes | `simple` or `delegated` |
| `status` | yes | `active` at creation |
| `created_at` | yes | ISO 8601 |
| `session_id` | yes | UUID |
| `turn_id` | yes | UUID |

---

## 4. `plan.md`

**Written by**: James, during `JAMES_ANALYSIS_DELEGATION_PREP` or `JAMES_IMPLEMENT_DELEGATION_PREP`
**Required for**: delegated tasks only

This is the execution plan handed off to the worker. The worker reads this file as part of `WORKER_LOAD_TASK`. It must be self-contained — the worker should not need to ask James questions after reading it.

### Format

```
task_id: <task-id>

# Plan: <task title>

## Approach

<High-level description of what will be done and in what order. Should be clear enough that the worker can execute without ambiguity.>

## Steps

1. <step one>
2. <step two>
3. <step three>
...

## Known risks or unknowns

<Anything James identified as uncertain, ambiguous, or potentially problematic before handoff. Empty section if none.>
```

### Field rules

| Field | Required | Values |
|---|---|---|
| `task_id` | yes | must match the task's `task_id` |

---

## 5. `todo.md`

**Written by**: James, during task init or delegation prep
**Updated by**: worker, as steps are completed
**Required for**: all non-trivial tasks

This is the live progress artifact. The worker updates checkboxes as it works through the steps. James and the user can inspect this file at any point to see exactly where execution is.

### Format

```
task_id: <task-id>

# Todo: <task title>

## Items

- [ ] <step one>
- [ ] <step two>
- [ ] <step three>
...

## Current step

<The item currently being worked on. Updated by the worker as it progresses.>
```

### Field rules

| Field | Required | Values |
|---|---|---|
| `task_id` | yes | must match the task's `task_id` |

### Update rules

- Completed items are marked `[x]`
- The `## Current step` section is updated each time the worker moves to a new item
- Workers must update `todo.md` incrementally as they work, not only at the end

---

## 6. `findings.md`

**Written by**: worker, during `WORKER_WRITE_RESULT` for analysis tasks
**Required for**: delegated analysis tasks

This is the worker's structured output for analysis work. James reads this file in `JAMES_READ_WORKER_RESULT` to produce the final user-facing response. The worker writes this and stops — it does not reply to the user directly.

### Format

```
task_id: <task-id>
status: complete
completed_at: <ISO 8601 timestamp>

# Findings: <task title>

## Summary

<One to three paragraph summary of what was found. This is what James will draw from most heavily when writing the user response.>

## Detailed findings

<Full analysis content. Can be as long as needed. Use subheadings to organize if the findings are complex.>

## Limitations

<What was out of scope, could not be determined, or would require additional investigation. Empty section if none.>
```

### Field rules

| Field | Required | Values |
|---|---|---|
| `task_id` | yes | must match the task's `task_id` |
| `status` | yes | `complete` |
| `completed_at` | yes | ISO 8601 |

---

## 7. `output.md`

**Written by**: worker, during `WORKER_WRITE_RESULT` for implementation tasks
**Required for**: delegated implementation tasks

This is the worker's structured output for implementation work. James reads this file in `JAMES_READ_WORKER_RESULT` to produce the final user-facing summary. The worker writes this and stops.

### Format

```
task_id: <task-id>
status: complete
completed_at: <ISO 8601 timestamp>

# Output: <task title>

## Summary of changes

<Concise description of what was changed and why.>

## Files modified

- `<path/to/file>` — <what changed in this file>
- `<path/to/other>` — <what changed in this file>

## What remains

<Any follow-up work that was not completed, is out of scope, or should be addressed separately. Empty section if everything is complete.>
```

### Field rules

| Field | Required | Values |
|---|---|---|
| `task_id` | yes | must match the task's `task_id` |
| `status` | yes | `complete` |
| `completed_at` | yes | ISO 8601 |

---

## 8. Permitted filenames summary

| Filename | Written by | Required for |
|---|---|---|
| `task.md` | James | all non-trivial tasks |
| `plan.md` | James | delegated tasks only |
| `todo.md` | James (created), worker (updated) | all non-trivial tasks |
| `findings.md` | worker | delegated analysis tasks |
| `output.md` | worker | delegated implementation tasks |

No other filenames are permitted inside a task subfolder. See `TOOLS_AND_PERMISSIONS.md` section 9 for enforcement rules.

---

## 9. How routing is internalized

There is no external routing classifier. The route emerges entirely from what James writes:

- If James answers the user directly without creating a task workspace → **Chat**
- If James creates a task and writes `task.md` with `type: analysis` and `delegation: simple` → **Analysis Simple**
- If James creates a task and writes `task.md` with `type: analysis` and `delegation: delegated` → **Analysis Delegated**
- If James creates a task and writes `task.md` with `type: implement` and `delegation: simple` → **Implement Simple**
- If James creates a task and writes `task.md` with `type: implement` and `delegation: delegated` → **Implement Delegated**

The `type` and `delegation` fields in `task.md` are the routing declaration. The system reads them to update the `route` field in the `turns` table and to drive state machine transitions. James determines these fields as a natural part of deciding how to handle the request — it is a judgment call, not a classifier output.
