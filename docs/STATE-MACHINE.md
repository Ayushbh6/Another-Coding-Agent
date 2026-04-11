

# STATE MACHINE

## 1. Purpose

This document defines the operational state machine for Another Coding Agent (ACA).

The purpose of the state machine is to make James predictable without making the system rigid. It defines:
- what state James or a worker is currently in
- what actions are allowed in that state
- what files must be written
- when a state must transition
- what the valid next states are

This state machine is the control backbone of v1.

---

## 2. Core Entities

### 2.1 James
James is the primary user-facing agent.

James is responsible for:
- receiving the user request
- routing the request
- deciding whether the work is chat, analysis, or implementation
- deciding whether the task is simple or delegated
- writing task setup artifacts
- optionally invoking challenger mode
- delegating to a worker when needed
- reading worker outputs
- replying to the user

### 2.2 Worker
The worker is a delegated execution agent.

The worker is responsible for:
- reading scoped task artifacts
- executing the assigned work
- managing its own active context
- writing structured task outputs
- stopping after its output is complete

### 2.3 Challenger
The challenger is an optional bounded critic.

The challenger is responsible for:
- challenging James's plan
- identifying flaws or blind spots
- suggesting safer or simpler alternatives

The challenger does not become a second James and does not directly respond to the user.

---

## 3. High-Level State Groups

ACA v1 has two major state groups:

### 3.1 James states
These govern the user-facing orchestration flow.

### 3.2 Worker states
These govern delegated execution.

Challenger is modeled as a temporary bounded state invoked by James.

---

## 4. State List

### James states
- `JAMES_IDLE`
- `JAMES_ROUTE`
- `JAMES_CHAT_EXECUTE`
- `JAMES_TASK_INIT`
- `JAMES_ANALYSIS_SIMPLE_EXECUTE`
- `JAMES_ANALYSIS_DELEGATION_PREP`
- `JAMES_IMPLEMENT_SIMPLE_EXECUTE`
- `JAMES_IMPLEMENT_DELEGATION_PREP`
- `JAMES_CHALLENGER_REVIEW`
- `JAMES_WAIT_FOR_WORKER`
- `JAMES_READ_WORKER_RESULT`
- `JAMES_FINALIZE_RESPONSE`
- `JAMES_DONE`

### Worker states
- `WORKER_IDLE`
- `WORKER_LOAD_TASK`
- `WORKER_ANALYSIS_EXECUTE`
- `WORKER_IMPLEMENT_EXECUTE`
- `WORKER_COMPACT_CONTEXT`
- `WORKER_WRITE_RESULT`
- `WORKER_DONE`

---

## 5. Global Rules

### 5.1 James always owns the user conversation
Only James responds to the user.

### 5.2 Workers never directly answer the user
Workers only write structured artifacts and stop.

### 5.3 Tool traces are current-turn only
Within one active turn, the agent can see relevant tool traces.
Across past turns, only the user message and final James response are replayed by default.

### 5.4 James and worker active memory are separate
James does not automatically see the worker's raw tool traces.
Workers do not automatically see James's raw tool traces.

### 5.5 Non-trivial work becomes a task
If work requires structured execution, James must create task artifacts.

### 5.6 Context compaction is scoped to past Q&A pairs only
When active context approaches its token threshold, the agent soft-removes the oldest past Q&A pairs from active memory. These pairs are always already persisted in the SQLite backend and are retrievable at any time.

Current-turn tool calls and results are **never compacted**. They remain in active context for the full duration of the current turn.

### 5.7 Compaction never requires fact preservation decisions
Because only past Q&A pairs are compacted, and because all history is fully retrievable via the hybrid memory retriever, the agent does not need to make any judgment about what facts to pin before compaction. Eviction is oldest-first and purely mechanical.

---

## 6. James State Definitions

## 6.1 `JAMES_IDLE`

### Meaning
James is waiting for a new user request.

### Entry condition
- no active user turn is being processed

### Allowed actions
- receive user input

### Exit condition
- a new user request arrives

### Valid next state
- `JAMES_ROUTE`

---

## 6.2 `JAMES_ROUTE`

### Meaning
James classifies the user request and decides the route.

### Main decision dimensions
- chat vs analysis vs implement
- simple vs delegated
- whether challenger should be invoked before delegation or major execution

### Allowed actions
- inspect the user request
- inspect lightweight repo context
- use limited read-only tools
- consult repo summary mechanisms
- decide routing mode

### Required decision output
James must choose exactly one of the following paths:
- chat
- analysis simple
- analysis delegated
- implement simple
- implement delegated

### Exit conditions and valid next states
- if Chat: `JAMES_CHAT_EXECUTE`
- if Analysis Simple: `JAMES_TASK_INIT`
- if Analysis Delegated: `JAMES_TASK_INIT`
- if Implement Simple: `JAMES_TASK_INIT`
- if Implement Delegated: `JAMES_TASK_INIT`

---

## 6.3 `JAMES_CHAT_EXECUTE`

### Meaning
James handles a simple request directly without turning it into a task workflow.

### Allowed actions
- use a small number of read-only tool calls
- inspect repo context
- compose a direct reply

### Forbidden actions
- no unnecessary task creation
- no worker delegation

### Exit condition
- James has enough information to answer the user

### Valid next state
- `JAMES_FINALIZE_RESPONSE`

---

## 6.4 `JAMES_TASK_INIT`

### Meaning
James creates the task workspace and writes the minimum required task setup artifacts.

### Allowed actions
- create task workspace in `.aca/`
- assign task identity
- write `task.md`
- determine whether the task is analysis or implementation
- determine whether the task is simple or delegated

### Required outputs
At minimum:
- `task.md`

Additional files depend on the route.

### Exit conditions and valid next states
- if Analysis Simple: `JAMES_ANALYSIS_SIMPLE_EXECUTE`
- if Analysis Delegated: `JAMES_ANALYSIS_DELEGATION_PREP`
- if Implement Simple: `JAMES_IMPLEMENT_SIMPLE_EXECUTE`
- if Implement Delegated: `JAMES_IMPLEMENT_DELEGATION_PREP`

---

## 6.5 `JAMES_ANALYSIS_SIMPLE_EXECUTE`

### Meaning
James performs a non-trivial analysis task directly.

### Allowed actions
- write `todo.md`
- inspect relevant files
- use analysis-oriented tools

### Required artifacts
- `task.md`
- `todo.md`

### Exit condition
- James has completed the analysis and has enough material to answer the user

### Valid next state
- `JAMES_FINALIZE_RESPONSE`

---

## 6.6 `JAMES_ANALYSIS_DELEGATION_PREP`

### Meaning
James prepares a delegated analysis task for a worker.

### Allowed actions
- write `plan.md`
- write `todo.md`
- optionally invoke challenger mode
- scope the handoff for the worker

### Required artifacts
- `task.md`
- `plan.md`
- `todo.md`

### Conditional transition
- if challenger is needed: `JAMES_CHALLENGER_REVIEW`
- otherwise: `JAMES_WAIT_FOR_WORKER`

### Exit condition
- the delegated analysis package is ready for worker execution

---

## 6.7 `JAMES_IMPLEMENT_SIMPLE_EXECUTE`

### Meaning
James performs a contained implementation task directly.

### Allowed actions
- write `todo.md`
- inspect relevant files
- perform edits
- run lightweight verification if available
- document what changed

### Required artifacts
- `task.md`
- `todo.md`

Optional:
- `output.md`

### Exit condition
- James has completed the implementation and can summarize the changes

### Valid next state
- `JAMES_FINALIZE_RESPONSE`

---

## 6.8 `JAMES_IMPLEMENT_DELEGATION_PREP`

### Meaning
James prepares a delegated implementation task for a worker.

### Allowed actions
- write `plan.md`
- write `todo.md`
- optionally invoke challenger mode
- scope the implementation handoff

### Required artifacts
- `task.md`
- `plan.md`
- `todo.md`

### Conditional transition
- if challenger is needed: `JAMES_CHALLENGER_REVIEW`
- otherwise: `JAMES_WAIT_FOR_WORKER`

### Exit condition
- the delegated implementation package is ready for worker execution

---

## 6.9 `JAMES_CHALLENGER_REVIEW`

### Meaning
James pauses before delegation or major execution so the challenger can critique the plan.

### Allowed actions
- submit the current plan to challenger
- receive critique
- revise `plan.md`
- revise `todo.md`

### Challenger goals
- identify flaws
- identify blind spots
- suggest simpler or safer alternatives

### Exit conditions and valid next states
- if originating state was analysis delegation prep: `JAMES_WAIT_FOR_WORKER`
- if originating state was implement delegation prep: `JAMES_WAIT_FOR_WORKER`

---

## 6.10 `JAMES_WAIT_FOR_WORKER`

### Meaning
James has handed off the task and is waiting for the worker to complete.

### Allowed actions
- remain inactive with respect to user-facing generation
- wait for worker completion signal

### Forbidden actions
- James should not continue doing the worker's job in parallel

### Exit conditions and valid next states
- when worker completes analysis output: `JAMES_READ_WORKER_RESULT`
- when worker completes implementation output: `JAMES_READ_WORKER_RESULT`

---

## 6.11 `JAMES_READ_WORKER_RESULT`

### Meaning
James reads the worker's structured result artifact.

### Allowed actions
- read `findings.md` for analysis tasks
- read `output.md` for implementation tasks
- inspect any worker-written supporting artifacts if needed

### Important rule
James reads the worker's structured result, not the worker's full raw trace by default.

### Exit condition
- James has enough information to produce the final user-facing response

### Valid next state
- `JAMES_FINALIZE_RESPONSE`

---

## 6.12 `JAMES_FINALIZE_RESPONSE`

### Meaning
James produces the final answer to the user.

### Allowed actions
- summarize the result clearly
- explain what was analyzed or changed
- mention remaining limitations if relevant
- conclude the current turn

### Exit condition
- final response has been delivered to the user

### Valid next state
- `JAMES_DONE`

---

## 6.13 `JAMES_DONE`

### Meaning
The turn is complete.

### Allowed actions
- mark the task complete
- leave workspace active until TTL expiry
- prepare for next user turn

### Important history rule
When the next user turn begins, the previous turn is replayed as:
- user message
- final James response

Raw tool traces from the completed turn are not replayed by default.

### Valid next state
- `JAMES_IDLE`

---

## 7. Worker State Definitions

## 7.1 `WORKER_IDLE`

### Meaning
The worker is inactive and waiting for delegated work.

### Entry condition
- no delegated task is currently assigned

### Exit condition
- a delegated task is assigned

### Valid next state
- `WORKER_LOAD_TASK`

---

## 7.2 `WORKER_LOAD_TASK`

### Meaning
The worker loads the delegated task package.

### Allowed actions
- read `task.md`
- read `plan.md`
- read `todo.md`
- load any explicitly provided scoped context

### Forbidden actions
- do not ingest James's full raw trace by default

### Exit conditions and valid next states
- if task type is analysis: `WORKER_ANALYSIS_EXECUTE`
- if task type is implement: `WORKER_IMPLEMENT_EXECUTE`

---

## 7.3 `WORKER_ANALYSIS_EXECUTE`

### Meaning
The worker executes a delegated analysis task.

### Allowed actions
- inspect repo files
- use approved analysis tools
- follow `todo.md`
- update internal notes if needed
- monitor context budget
- call hybrid memory retriever when needed information is no longer in active context

### Required final result artifact
- `findings.md`

### Conditional transitions
- if active context is approaching threshold: `WORKER_COMPACT_CONTEXT`
- if task analysis is complete: `WORKER_WRITE_RESULT`

---

## 7.4 `WORKER_IMPLEMENT_EXECUTE`

### Meaning
The worker executes a delegated implementation task.

### Allowed actions
- inspect relevant files
- perform code changes
- follow `todo.md`
- run allowed verification steps
- monitor context budget
- call hybrid memory retriever when needed information is no longer in active context

### Required final result artifact
- `output.md`

### Conditional transitions
- if active context is approaching threshold: `WORKER_COMPACT_CONTEXT`
- if implementation is complete: `WORKER_WRITE_RESULT`

---

## 7.5 `WORKER_COMPACT_CONTEXT`

### Meaning
The worker soft-removes the oldest past Q&A pairs from active context to free working memory.

### Allowed actions
- identify the oldest past Q&A pairs in active context
- confirm they are already persisted in the SQLite backend (they always are)
- remove the oldest pair(s) from active context

### What must NOT be touched
- current-turn tool calls and results are never removed
- only past Q&A pairs (Category B) are eligible for eviction

### Important rule
This is soft removal from active memory only, not permanent deletion from backend storage. All compacted pairs remain fully retrievable via the hybrid memory retriever.

No fact preservation or pinning step is required. The retriever handles recall on demand.

### Exit conditions and valid next states
- if original task type is analysis: `WORKER_ANALYSIS_EXECUTE`
- if original task type is implement: `WORKER_IMPLEMENT_EXECUTE`

---

## 7.6 `WORKER_WRITE_RESULT`

### Meaning
The worker writes the final structured result artifact and stops.

### Allowed actions
For analysis tasks:
- write `findings.md`

For implementation tasks:
- write `output.md`

Optional:
- update supporting notes if needed

### Exit condition
- required result artifact is complete

### Valid next state
- `WORKER_DONE`

---

## 7.7 `WORKER_DONE`

### Meaning
The worker has completed its task and stops.

### Allowed actions
- emit completion signal to James
- stop execution

### Valid next state
- `WORKER_IDLE`

---

## 8. Challenger State Model

Challenger does not need a large standalone state graph in v1.

It operates as a bounded subroutine inside `JAMES_CHALLENGER_REVIEW`.

Its internal sub-steps are:
1. receive current plan
2. critique plan
3. identify flaws, blind spots, or simplifications
4. return critique to James
5. stop

---

## 9. Main Transition Flows

## 9.1 Chat flow
`JAMES_IDLE`
→ `JAMES_ROUTE`
→ `JAMES_CHAT_EXECUTE`
→ `JAMES_FINALIZE_RESPONSE`
→ `JAMES_DONE`
→ `JAMES_IDLE`

---

## 9.2 Analysis simple flow
`JAMES_IDLE`
→ `JAMES_ROUTE`
→ `JAMES_TASK_INIT`
→ `JAMES_ANALYSIS_SIMPLE_EXECUTE`
→ `JAMES_FINALIZE_RESPONSE`
→ `JAMES_DONE`
→ `JAMES_IDLE`

---

## 9.3 Analysis delegated flow
`JAMES_IDLE`
→ `JAMES_ROUTE`
→ `JAMES_TASK_INIT`
→ `JAMES_ANALYSIS_DELEGATION_PREP`
→ optional `JAMES_CHALLENGER_REVIEW`
→ `JAMES_WAIT_FOR_WORKER`

Worker side:
`WORKER_IDLE`
→ `WORKER_LOAD_TASK`
→ `WORKER_ANALYSIS_EXECUTE`
→ optional `WORKER_COMPACT_CONTEXT`
→ `WORKER_WRITE_RESULT`
→ `WORKER_DONE`

Back to James:
`JAMES_READ_WORKER_RESULT`
→ `JAMES_FINALIZE_RESPONSE`
→ `JAMES_DONE`
→ `JAMES_IDLE`

---

## 9.4 Implement simple flow
`JAMES_IDLE`
→ `JAMES_ROUTE`
→ `JAMES_TASK_INIT`
→ `JAMES_IMPLEMENT_SIMPLE_EXECUTE`
→ `JAMES_FINALIZE_RESPONSE`
→ `JAMES_DONE`
→ `JAMES_IDLE`

---

## 9.5 Implement delegated flow
`JAMES_IDLE`
→ `JAMES_ROUTE`
→ `JAMES_TASK_INIT`
→ `JAMES_IMPLEMENT_DELEGATION_PREP`
→ optional `JAMES_CHALLENGER_REVIEW`
→ `JAMES_WAIT_FOR_WORKER`

Worker side:
`WORKER_IDLE`
→ `WORKER_LOAD_TASK`
→ `WORKER_IMPLEMENT_EXECUTE`
→ optional `WORKER_COMPACT_CONTEXT`
→ `WORKER_WRITE_RESULT`
→ `WORKER_DONE`

Back to James:
`JAMES_READ_WORKER_RESULT`
→ `JAMES_FINALIZE_RESPONSE`
→ `JAMES_DONE`
→ `JAMES_IDLE`

---

## 10. File Requirements by State

| State | Required files |
|---|---|
| `JAMES_TASK_INIT` | `task.md` |
| `JAMES_ANALYSIS_SIMPLE_EXECUTE` | `task.md`, `todo.md` |
| `JAMES_ANALYSIS_DELEGATION_PREP` | `task.md`, `plan.md`, `todo.md` |
| `JAMES_IMPLEMENT_SIMPLE_EXECUTE` | `task.md`, `todo.md` |
| `JAMES_IMPLEMENT_DELEGATION_PREP` | `task.md`, `plan.md`, `todo.md` |
| `WORKER_ANALYSIS_EXECUTE` | delegated task package loaded |
| `WORKER_IMPLEMENT_EXECUTE` | delegated task package loaded |
| `WORKER_WRITE_RESULT` (analysis) | `findings.md` |
| `WORKER_WRITE_RESULT` (implement) | `output.md` |

---

## 11. Design Intent

The purpose of this state machine is not to force James into robotic behavior.

The purpose is to ensure that:
- routing is explicit
- delegation is structured
- workers are bounded
- context stays manageable
- task artifacts are predictable
- future growth to spawn mode is possible

This is the v1 control system.

Future spawn mode in v2 will extend this state machine with sub-agent orchestration states.