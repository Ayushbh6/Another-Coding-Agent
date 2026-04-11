# FLOWS

## 1. Purpose

This document defines the exact expected execution flows for Another Coding Agent (ACA).

The goal is to make James's behavior predictable without making the system brittle. These flows describe the **ideal operating pattern** for v1 and the intended future pattern for v2.

They are the reference examples for how James should think, route, write task files, delegate, and return results.

---

## 2. Core Flow Rules

These rules apply across all flows unless explicitly stated otherwise.

### 2.1 James is always the user-facing agent

James receives the user request, decides the route, and gives the final response to the user.

### 2.2 Simple things stay simple

If a request can be answered with a small amount of inspection, James should answer directly without creating unnecessary task files.

### 2.3 Non-trivial work becomes a task

If the request needs more than a lightweight inspection or structured execution, James must convert it into a task workflow.

### 2.4 Delegation is structured

When delegating, James does not dump raw thinking onto the worker. James hands off through structured files.

### 2.5 Workers do the work and stop

Workers do not speak to the user. They write findings and stop.

### 2.6 Old tool traces are not default chat history

Tool traces belong to the current turn only. Across turns, only the user message and the final James response are carried by default.

### 2.7 James and workers have separate active memory

James does not automatically see the worker's raw tool trace. Workers do not automatically see James's raw tool trace.

---

## 3. Route Types

Every incoming user request is routed into one of these modes:

* **Chat**
* **Analysis Simple**
* **Analysis Delegated**
* **Implement Simple**
* **Implement Delegated**

Future mode:

* **Spawn Mode** (v2)

---

## 4. Dream Flow 1: Simple Chat

### 4.1 Use case

The user asks something small that can be answered with minimal repo inspection.

Example:

* "What is the name of this repo?"
* "Is this a Python project?"
* "What framework is this using?"

### 4.2 Expected behavior

1. user sends a simple question
2. James reads the question
3. James decides this is a **Chat** route
4. James may use up to a small number of read-only tool calls
5. James uses those calls to inspect repo context
6. James answers directly in chat
7. no task workflow is created
8. the turn ends

### 4.3 Important context rule

Inside this turn, James sees the current tool-call trace.

On the next user turn, the history passed back to James contains only:

* the user's message
* James's final answer

It does **not** contain the raw tool traces from this completed turn.

### 4.4 Dream example

User:

> What is the name of this repo?

James:

1. routes to Chat
2. calls repo inspection / summary tools
3. reads result
4. replies directly:

> The repo is called `another-coding-agent`.

No task files are created.

---

## 5. Dream Flow 2: Analysis Simple

### 5.1 Use case

The user wants explanation or diagnosis, but the work is still contained enough that James can do it directly.

Example:

* "Explain how the agent structure works in this repo."
* "What does the main CLI entrypoint do?"
* "Can you summarize how config loading works here?"

### 5.2 Expected behavior

1. user sends an analysis request
2. James inspects the request
3. James decides this is **Analysis Simple**
4. James creates a task workspace in `.aca/`
5. James writes `task.md`
6. James writes `todo.md`
7. James performs the analysis itself
8. James updates task artifacts as needed
9. James produces the final explanation to the user
10. task workspace remains temporarily available
11. after TTL, workspace is moved to user-level archive

### 5.3 Why task files are used here

Even though James does the work itself, the task is still non-trivial enough to benefit from visible structure and recoverability.

### 5.4 Dream example

User:

> Analyze the repo and tell me how the agent structure works.

James:

1. decides it is too large for a pure chat reply
2. creates `.aca/active/task-xyz/`
3. writes `task.md`
4. writes `todo.md`
5. inspects repo structure
6. reads core files
7. produces a clear analysis
8. replies to the user with the explanation

Files created:

* `task.md`
* `todo.md`

---

## 6. Dream Flow 3: Analysis Delegated

### 6.1 Use case

The user wants analysis that is broad, multi-step, or too large for James's direct working loop.

Example:

* "Analyze this whole repo and explain the architecture in detail."
* "Find design weaknesses in the current agent memory system."
* "Review all auth flows and tell me where the security risks are."

### 6.2 Expected behavior

1. user sends a larger analysis request
2. James inspects the request and repo context
3. James decides this is **Analysis Delegated**
4. James creates a task workspace in `.aca/`
5. James writes `task.md`
6. James writes `plan.md`
7. James writes `todo.md`
8. James hands off to a worker
9. worker reads scoped task artifacts
10. worker executes the analysis sequentially
11. worker writes `findings.md`
12. worker stops
13. James wakes back up
14. James reads `findings.md`
15. James produces the final user-facing answer
16. task workspace remains for TTL period
17. workspace is later archived out of repo-local active area

### 6.3 Critical handoff rule

The worker is handed:

* task definition
* plan
* todo list
* any necessary scoped context

The worker is **not** handed James's full raw internal trace.

### 6.4 Critical response rule

The worker does **not** reply to the user.
The worker only writes `findings.md` and stops.

### 6.5 Dream example

User:

> Analyze the whole repo and tell me how the agent memory architecture currently works, where it is weak, and what should be improved.

James:

1. routes to Analysis Delegated
2. writes `task.md`
3. writes `plan.md`
4. writes `todo.md`
5. delegates to worker

Worker:

1. reads the files
2. performs file inspection and analysis
3. writes `findings.md`
4. stops

James:

1. reads `findings.md`
2. writes a polished final answer to the user

Files created:

* `task.md`
* `plan.md`
* `todo.md`
* `findings.md`

---

## 7. Dream Flow 4: Implement Simple

### 7.1 Use case

The user wants a contained code change that James can safely perform itself.

Example:

* "Rename this function."
* "Add a missing type hint here."
* "Fix this small import issue."
* "Create a basic README section for setup."

### 7.2 Expected behavior

1. user sends an implementation request
2. James inspects request and local repo context
3. James decides this is **Implement Simple**
4. James creates task workspace
5. James writes `task.md`
6. James writes `todo.md`
7. James performs the edits itself
8. James verifies the result if needed
9. James replies with what changed
10. workspace remains until TTL expiry

### 7.3 Dream example

User:

> Add type hints to the main config loader.

James:

1. routes to Implement Simple
2. writes `task.md`
3. writes `todo.md`
4. inspects the file
5. edits the file
6. checks for consistency
7. reports back clearly to the user

---

## 8. Dream Flow 5: Implement Delegated

### 8.1 Use case

The user wants a larger implementation that should be handed to a worker.

Example:

* "Refactor the memory manager."
* "Add SQLite logging for all tool calls."
* "Implement the checkpoint and pin-fact system."
* "Add CLI approval modes and permission boundaries."

### 8.2 Expected behavior

1. user sends a larger implementation request
2. James inspects request and repo context
3. James decides this is **Implement Delegated**
4. James creates task workspace
5. James writes `task.md`
6. James writes `plan.md`
7. James writes `todo.md`
8. James optionally invokes challenger before finalizing the plan
9. James delegates the task to a worker
10. worker reads scoped files
11. worker implements step by step
12. worker may update internal task artifacts during execution
13. worker writes `output.md`
14. worker stops
15. James reads `findings.md`
16. James gives final user-facing summary
17. workspace is later archived after TTL

### 8.3 Dream example

User:

> Implement a context compaction system with checkpointing and pinned facts.

James:

1. routes to Implement Delegated
2. writes `task.md`
3. writes `plan.md`
4. writes `todo.md`
5. maybe invokes challenger
6. delegates to worker

Worker:

1. reads the scoped files
2. modifies relevant code
3. writes `output.md`
4. stops

James:

1. reads `output.md`
2. explains what was implemented and what remains

---

## 9. Dream Flow 6: Challenger Mode

### 9.1 Use case

A challenge pass is useful before larger analysis or implementation work.

Example:

* before a major refactor
* before a delegated architecture review
* before future spawn planning

### 9.2 Expected behavior

1. James forms an initial plan
2. challenger mode is invoked
3. challenger reviews the plan
4. challenger tries to:

   * identify flaws
   * identify missed edge cases
   * suggest simplifications
5. challenger returns critique
6. James updates the plan if needed
7. James continues with simple execution or delegation

### 9.3 Boundaries

Challenger is a bounded critic.
It is not meant to become a second full worker or a second James.

### 9.4 Dream example

User:

> Refactor the entire memory system.

James:

1. drafts a plan
2. calls challenger
3. receives critique like:

   * missing migration plan
   * unclear archive retrieval path
   * too much scope in one patch
4. improves plan
5. proceeds to delegation

---

## 10. Dream Flow 7: Current-Turn Context Behavior

This is a core rule and must remain true across all flows.

### 10.1 Within one parent turn

Inside one user turn, James may make multiple sub-turns with tool calls.

In that same turn, the running model context includes:

* current user request
* relevant system instructions
* relevant task state
* current turn tool calls and tool results

That is correct and expected.

### 10.2 After the turn ends

Once James gives the final answer, that turn is considered complete.

When the next user turn begins, the previous turn is carried forward only as:

* previous user message
* previous final James reply

The old tool-call trace is not replayed into the new turn by default.

---

## 11. Dream Flow 8: Separate James and Worker Memory

This is another core rule.

### 11.1 James memory

James sees:

* user request
* James task files
* James current-turn tool trace
* final structured outputs from workers

### 11.2 Worker memory

Worker sees:

* scoped handoff files
* worker current-turn tool trace
* any explicitly provided task context

### 11.3 Separation rule

* worker does not automatically inherit James's full trace
* James does not automatically inherit worker's full trace
* handoff must happen via structured artifacts

This keeps contexts smaller, cleaner, and more modular.

---

## 12. Dream Flow 9: Context Compaction During Long Worker Runs

### 12.1 Use case

A worker is executing a long delegated task and active context is getting too full.

### 12.2 Expected behavior

1. worker notices token budget threshold is being approached
2. worker enters `WORKER_COMPACT_CONTEXT`
3. worker identifies the oldest past Q&A pairs in active context
4. worker confirms those pairs are already persisted in the SQLite backend
5. worker removes the oldest pair(s) from active context
6. worker resumes execution — current-turn tool calls and results remain fully intact

### 12.3 Important rule

Compaction is a **soft removal from active memory only**, not a permanent deletion from backend history. All compacted pairs remain retrievable at any time via the hybrid memory retriever.

### 12.4 What must never be compacted

Current-turn tool calls and results are never removed during compaction. Only past Q&A pairs (Category B) are eligible. This means the worker always retains full fidelity of everything that happened in the current execution run.

### 12.5 If the worker needs compacted information later

The worker calls `search_memory` with a natural language query. The hybrid retriever searches the SQLite backend using keyword search, BM25 ranking, and vector similarity, and returns ranked results. No pinning or pre-compaction decisions are required — the retriever handles recall on demand.

---

## 13. Dream Flow 10: Task Workspace Lifecycle

### 13.1 Creation

When a non-trivial task begins, a task workspace is created in `.aca/`.

### 13.2 Active use

James and/or workers write task artifacts into that workspace while the task is active.

### 13.3 Completion

After the task completes, the workspace remains temporarily available in the repo.

### 13.4 TTL expiry

After approximately 12 hours, the completed task workspace is removed from repo-local active storage and moved to user-level archival storage.

### 13.5 Backend durability

Even after workspace movement, the SQLite backend still contains the full trace and state history.

---

## 14. Dream Flow 11: Future Spawn Mode (V2)

This is not part of v1 implementation, but it is part of the long-term dream flow.

### 14.1 Use case

The user wants a larger system or feature set built across multiple domains.

Example:

* build a chatbot app
* scaffold a full-stack product
* implement a large feature touching backend, frontend, auth, and DB

### 14.2 Expected behavior

1. user makes a larger build or architecture request
2. James reviews the repo state or empty project state
3. James forms a high-level architecture understanding
4. challenger is automatically invoked before spawn planning
5. James identifies domain boundaries
6. James creates a spawn plan
7. James creates a `.spawn/` workspace
8. James spawns bounded sub-agents
9. each sub-agent is assigned a file/domain boundary
10. each sub-agent receives its own `CONTRACTS.md`
11. each sub-agent executes within its boundary
12. cross-agent coordination respects interface contracts
13. James supervises overall progress
14. James delivers the final integrated summary to the user

### 14.3 Example sub-agent ownership

Possible agent ownerships:

* frontend UI
* backend API
* auth/security
* database layer
* core agent orchestration
* tools layer

### 14.4 Why `CONTRACTS.md` matters

Each spawned agent must know:

* what files it owns
* what interfaces it exposes
* what input/output shapes must remain valid
* what other agents depend on it

This is what keeps spawn mode coherent.

---

## 15. Exact Dream Summary

The exact dream behavior is:

1. James stays light for small questions
2. James creates visible tasks for non-trivial work
3. James delegates only when it should
4. workers execute and write findings, then stop
5. active context stays clean
6. backend trace stays complete
7. previous turns do not replay raw tool noise
8. James and workers keep separate active working memory
9. long tasks compact context by evicting oldest past Q&A pairs, with no checkpointing or fact pinning required — the hybrid retriever handles recall on demand
10. future spawned sub-agents work inside bounded ownership with explicit contracts

That is the target operating model for ACA.
