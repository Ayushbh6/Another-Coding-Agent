# ARCHITECTURE

## 1. Vision

Another Coding Agent (ACA) is a CLI coding agent built around one core idea:

**make agent work visible, structured, and resumable instead of hidden and chatty.**

The primary agent, **James**, should feel like a capable coding partner that can inspect a repository, reason about work, create structured task artifacts, delegate when needed, and maintain a clean working memory.

ACA is not designed as a brittle workflow engine with hardcoded branches for every case. Instead, it is an agentic system with:

* strong prompting and routing rules
* explicit task states
* controlled tool budgets
* lightweight file-based workspaces
* durable backend logs and memory
* clean separation between active context and archived trace data

The product goal is to combine the flexibility of modern coding agents with the traceability and structure of a disciplined engineering workflow.

---

## 2. Core Product Principles

### 2.1 Visible work

James and any worker agents write task artifacts into repo-local workspace folders so the user can inspect what is happening.

### 2.2 Clean context

Only useful information should remain in active context. Raw tool noise should be preserved in storage, not replayed by default.

### 2.3 Controlled escalation

Simple questions should stay simple. More complex work should become explicit tasks.

### 2.4 Separate concerns

James, workers, and future spawned agents operate with separate working contexts.

### 2.5 Durable backend truth

All events, tool calls, task state changes, failures, and outcomes are stored in SQLite for debugging, auditability, and future intelligence.

### 2.6 Repo respect

The repo should not become cluttered. Active workspace files are temporary and should age out automatically.

---

## 3. Main Entities

## 3.1 James

James is the primary user-facing coding agent.

Responsibilities:

* talk to the user
* inspect repo context
* decide routing mode
* answer simple requests directly
* create task artifacts for non-trivial work
* delegate to a worker when needed
* read worker findings and respond back to the user
* manage context boundaries and memory rules

James is the orchestrator and final narrator.

---

## 3.2 Worker

The worker is a delegated execution agent used in v1 for longer or more involved tasks.

Responsibilities:

* receive a scoped task from James
* follow a defined plan and todo list
* perform sequential work
* write findings into workspace files
* stop after producing findings

The worker does not speak directly to the user.

---

## 3.3 Challenger

Challenger mode is an optional critique pass used before larger planning or implementation steps.

Responsibilities:

* challenge James's plan
* identify flaws or blind spots
* suggest missing edge cases
* push toward simpler or safer approaches

The challenger is not meant to become a second full planner. It is a bounded critic.

---

## 3.4 Future Spawned Sub-Agents

In v2, ACA may support multiple domain-bounded spawned agents.

Examples:

* frontend agent
* backend agent
* auth/security agent
* database agent
* core agent logic owner

Each spawned sub-agent will have:

* file ownership boundaries
* dedicated workspace
* dedicated task files
* a `CONTRACTS.md` file describing key interfaces and obligations

---

## 4. System Scope by Version

## 4.1 V1 Scope

V1 includes:

* James
* one optional worker
* optional challenger mode
* task workspace in `.aca/`
* durable storage in `data/`
* SQLite logs and trace history
* ChromaDB for RAG
* past Q&A context compaction with on-demand hybrid memory retrieval
* structured routing between chat, analysis, and implementation

V1 does **not** include:

* multi-agent spawn orchestration
* many parallel workers
* cross-agent contract reconciliation
* recursive delegation trees

---

## 4.2 V2 Scope

V2 extends the foundation with:

* `/spawn` mode
* `.spawn/` folder structure
* multiple sub-agents with owned file boundaries
* `CONTRACTS.md` per sub-agent
* contract-aware coordination across owned domains
* stronger multi-agent repo planning

The v2 system should only be built after the v1 state machine, memory system, and task flow are stable.

---

## 5. High-Level Routing Model

Every user request is classified into one of three high-level routes:

### 5.1 Chat

Used when the user request can be answered directly with minimal inspection.

Characteristics:

* few read-only tool calls
* no durable task workflow needed
* James answers directly in chat

### 5.2 Analysis

Used when the user wants understanding, diagnosis, exploration, explanation, or recommendations.

Can be:

* analysis simple
* analysis delegated

### 5.3 Implement

Used when the user wants file changes, code changes, edits, patches, commands, or structured repo work.

Can be:

* implement simple
* implement delegated

---

## 6. Simple vs Delegated Work

## 6.1 Simple

A simple task is one James can finish directly without handing off to a worker.

Typical signs:

* limited repo inspection needed
* low ambiguity
* small scope
* no long sequential execution loop required

For simple tasks, James writes the necessary task artifact(s), executes the task itself, and returns the result.

---

## 6.2 Delegated

A delegated task is one James should hand off to a worker.

Typical signs:

* long or multi-step work
* broader file inspection required
* higher uncertainty
* more reasoning or execution than should sit in James's main turn loop

For delegated tasks, James prepares the task and plan, hands it off, then later reads findings and replies to the user.

---

## 7. High-Level Task Lifecycle

A non-trivial task follows this general lifecycle:

1. user makes a request
2. James inspects repo context using limited read-only tools
3. James classifies the route
4. if task-based, James creates task artifacts in `.aca/`
5. James either executes directly or delegates to a worker
6. work proceeds with structured files and logged tool actions
7. findings are written
8. James returns a final user-facing response
9. task workspace remains temporarily available
10. after TTL expiry, workspace is moved out of repo-local active area into user-level storage/archive

---

## 8. Workspace and Storage Layout

## 8.1 Repo-Local Workspace: `.aca/`

Each repo that ACA works in contains a `.aca/` folder.

Purpose:

* current task workspaces
* inspectable task artifacts
* short-lived agent working files

Illustrative structure:

```text
.aca/
  active/
    task-001/
      task.md
      plan.md
      todo.md
      findings.md
  summaries/
  logs/
```

The exact subfolder structure may evolve, but `.aca/` is the repo-local active workspace area.

Key constraint: agents may only write into named task subfolders under `.aca/active/`. They may not write directly into `.aca/` or `.aca/active/` itself. Only markdown files with permitted artifact filenames are allowed inside task subfolders. The full path and file-type guardrails are defined in `TOOLS_AND_PERMISSIONS.md` section 9.

---

## 8.2 Backend Data Folder: `data/`

ACA maintains a backend data area outside the lightweight task files.

Core contents:

* `aca.db` for SQLite — the single source of truth for all events, tool calls, conversation history, and trace data
* ChromaDB files for vector embeddings used by the hybrid memory retriever

Illustrative structure:

```text
data/
  aca.db
  chroma/
```

This is the durable system memory and trace layer. Everything is stored here permanently. Active context is a selective window over this complete history.

---

## 8.3 User-Level Base Archive

Completed or expired repo-local task workspaces are moved after TTL expiry to a user-level base directory outside the active repo.

Purpose:

* keep repos clean
* preserve work history
* support later debugging or retrieval

The intended TTL for repo-local task workspace retention is **12 hours** after task completion.

---

## 8.4 Future Spawn Workspace: `.spawn/`

In v2, spawned sub-agent structures may live in a dedicated `.spawn/` folder.

Illustrative direction:

```text
.spawn/
  backend-agent/
    CONTRACTS.md
    todo.md
    findings.md
  frontend-agent/
    CONTRACTS.md
    todo.md
    findings.md
```

This is not part of v1.

---

## 9. Durable Logging and Traceability

SQLite is the system of record for all fine-grained operational history.

ACA should log:

* user turns
* James responses
* worker runs
* challenger runs
* tool calls
* tool success/failure
* state changes
* task creation
* task completion
* context compaction events
* archive references
* errors and retries

This allows full reconstruction of what happened in any task or turn.

Important design rule:
**the backend stores everything, but the active model context should stay selective.**

---

## 10. Turn and Context Rules

## 10.1 Within-Turn Rule

Within the current active turn, the agent should see the tool interaction history relevant to completing that turn.

Reason:

* the current turn depends on the immediate chain of reads, calls, and results

---

## 10.2 Across-Turn Rule

Across past turns, James should only receive:

* the user message
* the final James response

By default, James should **not** receive previous turns' raw tool traces as chat history.

Reason:

* saves tokens
* reduces noise
* keeps historical context readable
* avoids replaying unnecessary tool garbage

---

## 10.3 Fetch-on-Demand Principle

If James needs detailed information from older tool traces, those details should be fetched from backend storage through dedicated retrieval mechanisms rather than replayed automatically in the conversational history.

---

## 10.4 Separate Agent Memory Principle

James and workers do not share full internal working histories by default.

Rules:

* James does not automatically see the worker's full tool-call trace
* worker does not automatically see James's full tool-call trace
* each agent operates in its own active memory context
* handoff should happen through structured artifacts, not raw trace replay

This same principle will apply to future spawned agents in v2.

---

## 11. Context Management Philosophy

ACA treats context as a managed working set, not an ever-growing transcript.

Key idea:

* active context is like RAM
* the SQLite backend is like disk
* everything is always retrievable on demand

The model should not be forced to carry all prior conversation history forever.

There are two strictly separated categories of content in active context:

### Category A — Current turn (locked)

Everything from the current active turn is untouchable:

* current user query
* all tool calls and results from this turn
* agent reasoning in progress

Current-turn content is **never soft-removed**. The agent always has full fidelity of everything that has happened in the current execution run.

### Category B — Past turns (soft-removable)

Past turns are carried as Q&A pairs only:

* previous user message
* previous final James or worker response

These pairs are the only thing eligible for soft removal under context pressure. They are evicted oldest-first, like an LRU cache. No metacognitive judgment is required from the model — eviction is purely mechanical based on recency.

Because past turns already strip raw tool traces by design (see section 10.2), soft removal of Q&A pairs is safe. There is no risk of losing in-flight tool state, because in-flight state is always in Category A and is never touched.

---

## 12. Memory Management Components

V1 memory management is built around three core actions:

### 12.1 Context compaction

When active context approaches its token threshold, the agent soft-removes the oldest past Q&A pairs from active memory. These pairs are not deleted — they remain fully preserved in the SQLite backend. Only Category B content (past Q&A pairs) is ever compacted. Category A content (current-turn tool calls and results) is never touched.

### 12.2 Hybrid memory retrieval

When the agent needs information that has been compacted out of active context, it calls the hybrid retriever tool. This tool queries the SQLite backend using a combination of:

* **Keyword search** — exact word and phrase matching via SQLite FTS5
* **BM25 ranking** — relevance-scored full-text search, natively supported by SQLite FTS5
* **Vector search** — semantic similarity search using embeddings stored in ChromaDB

The retriever accepts a natural language query and returns ranked results from the complete backend history. The agent does not need to know in advance what category a piece of information falls into — the hybrid search finds it regardless of whether it is best matched by keyword, relevance score, or semantic meaning.

A recency bias is applied when BM25 and vector scores are close: more recent results are preferred to avoid surfacing stale superseded context.

### 12.3 Archive search

The same hybrid retriever also covers the broader archive — completed task workspaces, older runs, and any other durable history stored in the backend. There is no separate archive tool. One retriever covers everything.

This system enables long-running tasks without uncontrolled context growth, and without requiring the agent to make upfront predictions about what information it will need later.

---

## 13. Compaction Rule

When active context grows too large, ACA soft-removes past Q&A pairs in a controlled order:

1. identify the oldest past Q&A pairs in active context (Category B only)
2. ensure those pairs are already persisted in the SQLite backend (they always are, by design)
3. remove the oldest pair(s) from active context
4. continue work — the current turn's full tool trace remains intact and untouched

Important clarification:
this is a **soft removal from active memory only**, not permanent deletion from durable storage. The compacted Q&A pairs remain fully retrievable via the hybrid memory retriever at any time.

Critical constraint:
**current-turn tool calls and results (Category A) are never compacted.** Only past Q&A pairs are eligible for removal.

---

## 14. Retrieval Rule

Whenever the agent needs information that is no longer in active context, it must use the hybrid retriever tool rather than guessing or hallucinating.

The hybrid retriever covers:

* past Q&A pairs that were compacted out of active context
* raw tool traces from past turns stored in the backend
* completed task workspace contents
* any other durable history in SQLite or ChromaDB

The agent should treat the hybrid retriever as its extended memory — anything that was ever seen is still findable.

---

## 15. Tool Budget Philosophy

James should operate under controlled tool budgets, especially for simple chat-like requests.

The purpose of tool budgets is to:

* prevent runaway loops
* force routing decisions early
* encourage task creation for non-trivial work
* keep simple interactions fast

Detailed state-by-state tool allowances belong in `STATE_MACHINE.md` and `TOOLS_AND_PERMISSIONS.md`.

---

## 16. File-Based Task Artifacts

For non-trivial work, ACA writes structured task artifacts into the active workspace.

Likely artifact types include:

* `task.md`
* `plan.md`
* `todo.md`
* `findings.md`
* `output.md`

These files act as shared structure for the active task and as visible evidence of work.

Exact schemas should be defined separately in `FILE_SCHEMAS.md`.

---

## 17. Safety and Permission Philosophy

ACA is a coding agent with read/write capability, so safety must be built in from the start.

Core principles:

* least privilege by default
* read-only by default where possible
* explicit approval for risky actions
* repo-bounded file access
* traceable edits and commands
* secret-aware behavior
* auditability of all actions

The detailed permission model belongs in `TOOLS_AND_PERMISSIONS.md`.

---

## 18. Memory Retrieval and Repo Understanding

ACA uses a hybrid retriever as the single unified memory access tool.

The retriever combines:

* **Keyword / FTS search** — SQLite FTS5 for exact word and phrase matching
* **BM25 ranking** — relevance-scored full-text search, built into SQLite FTS5
* **Vector search** — semantic similarity via embeddings stored in ChromaDB

Primary use cases:

* retrieving past Q&A pairs compacted out of active context
* retrieving raw tool traces from prior turns when needed
* repo semantic search during analysis tasks
* archive lookup for completed task history

The retriever is the single tool that covers all of these cases. It does not replace direct file inspection tools for current-turn work, but it is the canonical way to access anything that has left active context.

---

## 19. Challenger Mode Philosophy

Challenger mode exists to improve plan quality before costly execution.

It should be used selectively, especially:

* before larger delegated work
* before major implementation plans
* before future spawn orchestration

Challenger mode should remain bounded and focused on critique, not become an uncontrolled second planner.

---

## 20. Future Spawn Architecture

In v2, `/spawn` introduces a stronger multi-agent architecture.

Proposed characteristics:

* James evaluates the repo or project goal
* James identifies meaningful ownership boundaries
* James creates a limited set of spawned sub-agents
* each spawned sub-agent owns a file/domain boundary
* each spawned sub-agent maintains its own `CONTRACTS.md`
* cross-domain work is coordinated through plans and contract consistency

This enables larger repo work while keeping domain responsibilities clean.

---

## 21. Why `CONTRACTS.md` Matters in V2

The `CONTRACTS.md` concept is the backbone of future multi-agent coordination.

Each spawned agent's contract file should describe:

* owned files
* public interfaces
* key functions
* input/output expectations
* routes or schemas
* critical dependencies
* invariants that must remain valid

This prevents multi-agent work from degenerating into uncontrolled overlapping edits.

---

## 22. Design Boundaries

To keep the system practical, ACA should avoid these pitfalls:

* overusing task files for trivial questions
* replaying raw old tool traces by default
* adding too many agent types in v1
* building spawn orchestration before the single-worker model is stable
* using context as a garbage dump instead of a managed resource
* letting sub-agents share raw traces without structure

---

## 23. Summary

ACA is a structured CLI coding agent system centered on James.

In v1, it provides:

* a clear routing model
* one main agent and one worker
* optional challenger mode
* repo-local visible task workspaces
* durable SQLite and Chroma-backed memory
* past Q&A context compaction with oldest-first eviction
* hybrid memory retrieval (keyword + BM25 + vector) over the complete SQLite backend
* strong separation between current-turn tool state (locked) and past Q&A history (compactable)

In v2, it grows into a contract-aware spawned multi-agent system.

The core architectural idea is simple:

**keep active context lean, keep backend memory complete, and make work visible, structured, and recoverable.**

The memory model in one sentence: **current turn is sacred, past Q&A compresses under pressure, and everything is always retrievable.**
