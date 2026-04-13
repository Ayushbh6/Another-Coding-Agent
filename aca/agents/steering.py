"""
Steering junctions for ACA agents.

All LLM-facing injection strings and user-facing terminal messages live here
so the flow is visible in one place.

Injection policy: use role=\"user\" for mid-turn steering (OpenRouter / chat
completions models handle this reliably).

Junction keys are used by AgentConsole.steering_junction() for icons/styling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from aca.tools.registry import PermissionMode


class SteeringPhase(str, Enum):
    """High-level phase label for logging / debugging."""

    ORIENT = "orient"
    PLAN = "plan"
    EXECUTE = "execute"
    DELEGATE = "delegate"
    WORKER_EXECUTE = "worker_execute"


@dataclass(frozen=True)
class SteeringJunction:
    """One steering injection event."""

    key: str
    agent_msg: str
    user_terminal_msg: str
    restrict_to_mode: PermissionMode | None = None
    tool_choice: str | None = None  # e.g. "none" for forced final text


# ── Junction templates (agent-facing) ─────────────────────────────────────────

# ROUTE: after N read tools in orient phase — must create task workspace or reply in chat
ROUTE_AGENT = (
    "[STEERING — ROUTE] You have used {n} read/search tool calls in the orient phase. "
    "No more read_file, list_files, search_repo, or get_file_outline until you commit.\n\n"
    "Pick exactly ONE path now:\n"
    "  A) CHAT — Answer the user directly in your next message (plain text, no tools). "
    "Use this only if the request is trivial and you already have enough context.\n"
    "  B) TASK — Call `create_task_workspace()`, then use `write_task_file` "
    "to write `task.md` describing the work. Use this for anything non-trivial.\n\n"
    "You may use workspace + write_task_file tools now. Choose A or B immediately."
)

ROUTE_USER = (
    "[James] Magnifying glass time is over — "
    "pick a lane: quick chat answer, or spin up a real task. "
    "No more repo-snooping until you commit."
)

# WRITE_ARTIFACTS: after task workspace exists, ≤5 post-task tools, todo not done
WRITE_ARTIFACTS_AGENT = (
    "[STEERING — ARTIFACTS] You created a task workspace but `todo.md` is still missing "
    "after {n} post-setup tool calls.\n\n"
    "You MUST write `todo.md` via `write_task_file` now (checklist of concrete steps).\n"
    "If this task is **delegated** (large / multi-step / needs a Worker): "
    "you MUST also have `plan.md` written **before** `todo.md` is considered complete. "
    "If `plan.md` is missing for a delegated task, write `plan.md` first, then `todo.md`.\n\n"
    "No exploratory reads — only workspace writes until `todo.md` exists"
    "{deleg_hint}."
)

WRITE_ARTIFACTS_USER = (
    "[James] Stop wandering — the task folder needs a proper todo list. "
    "Clipboard energy: lock in `todo.md` (and `plan.md` if this is a big one)."
)

INVALID_ARTIFACT_AGENT = (
    "[STEERING — FIX ARTIFACT] `{filename}` does not match the required format.\n\n"
    "Problem detected: {reason}\n\n"
    "Read the matching example file from `~/.aca/example_guidelines/` now, then rewrite `{filename}` "
    "so the required header fields and section headings match the expected format. "
    "Do not continue to other work until `{filename}` is valid."
)

INVALID_ARTIFACT_USER = (
    "[James] The paperwork is sloppy — reread the template and rewrite the artifact before moving on."
)

# EXECUTE_SIMPLE: James does the work himself
EXECUTE_SIMPLE_AGENT = (
    "[STEERING — EXECUTE] Task artifacts are ready (`task.md` + `todo.md`). "
    "You are in **work mode**.\n\n"
    "Follow this exact sequence for every todo item:\n"
    "  1. Call `get_next_todo(task_id)` — this marks the next pending item [>] and tells you what to work on.\n"
    "  2. Work ONLY that item using repo edit tools (write_file, update_file, etc.).\n"
    "  3. When done, call `advance_todo(task_id, item_index, action='complete')`.\n"
    "  4. Repeat from step 1 until `all_done=true`.\n\n"
    "To skip an item: only if you are 100% certain it was already handled as a side-effect "
    "of previous work. Call `advance_todo(..., action='skip', skip_reason='<specific reason>')`. "
    "State exactly what was done, where, and when. Vague reasons are rejected by the tool. "
    "Do NOT skip items to avoid work.\n\n"
    "When `all_done=true`, reply to the user with a clear summary of what changed — no delegation."
)

EXECUTE_SIMPLE_USER = (
    "[James] Gloves on — working the checklist solo. "
    "Wrench mode engaged."
)

# DELEGATE: hand off to Worker
DELEGATE_AGENT = (
    "[STEERING — DELEGATE] `task.md`, `plan.md`, and `todo.md` are all written. "
    "This is a **delegated** task.\n\n"
    "Tell the user briefly that you are handing off to the Worker. "
    "Do NOT try to execute the todo yourself — the Worker will read the artifacts, "
    "complete the work, and write `findings.md` (analysis) or `output.md` (implementation). "
    "Keep your message short; the system will run the Worker next."
)

DELEGATE_USER = (
    "[James] Passing the baton — Worker picks up the heavy lifting. "
    "Right arrow of doom activated."
)

# FORCE_REPLY: global tool budget exhausted (Option B soft-stop)
FORCE_REPLY_AGENT = (
    "[STEERING — LIMIT] You have hit the tool-call ceiling for this turn ({n} calls). "
    "No more tools — synthesize your **final answer to the user** now from everything "
    "you already know. If work is unfinished, say exactly what remains."
)

FORCE_REPLY_USER = (
    "[James] Tool budget tapped out — time to land the plane. "
    "Checkered flag: give the user the straight answer."
)


def junction_route(n_reads: int) -> SteeringJunction:
    return SteeringJunction(
        key="route",
        agent_msg=ROUTE_AGENT.format(n=n_reads),
        user_terminal_msg=ROUTE_USER,
        restrict_to_mode=PermissionMode.EDIT,
    )


def junction_write_artifacts(n_post_task: int, needs_plan: bool) -> SteeringJunction:
    deleg_hint = " (and `plan.md` for delegated tasks)" if needs_plan else ""
    return SteeringJunction(
        key="write_artifacts",
        agent_msg=WRITE_ARTIFACTS_AGENT.format(n=n_post_task, deleg_hint=deleg_hint),
        user_terminal_msg=WRITE_ARTIFACTS_USER,
        restrict_to_mode=PermissionMode.EDIT,
    )


def junction_invalid_artifact(filename: str, reason: str) -> SteeringJunction:
    return SteeringJunction(
        key="invalid_artifact",
        agent_msg=INVALID_ARTIFACT_AGENT.format(filename=filename, reason=reason),
        user_terminal_msg=INVALID_ARTIFACT_USER,
        restrict_to_mode=PermissionMode.EDIT,
    )


def junction_execute_simple() -> SteeringJunction:
    return SteeringJunction(
        key="execute_simple",
        agent_msg=EXECUTE_SIMPLE_AGENT,
        user_terminal_msg=EXECUTE_SIMPLE_USER,
        restrict_to_mode=None,
    )


def junction_delegate() -> SteeringJunction:
    # tool_choice="none" forces James to emit a plain handoff text — no risk of
    # him doing the delegated work himself instead of passing to Worker.
    return SteeringJunction(
        key="delegate",
        agent_msg=DELEGATE_AGENT,
        user_terminal_msg=DELEGATE_USER,
        restrict_to_mode=None,
        tool_choice="none",
    )


def junction_force_reply(n_tools: int) -> SteeringJunction:
    return SteeringJunction(
        key="force_reply",
        agent_msg=FORCE_REPLY_AGENT.format(n=n_tools),
        user_terminal_msg=FORCE_REPLY_USER,
        restrict_to_mode=None,
        tool_choice="none",
    )


# ── COMPACT_CONTEXT junction ───────────────────────────────────────────────────

# Injected when resp.input_tokens >= context_token_threshold.
# The agent is shown the numbered inventory of Category B Q&A pairs and must
# call compact_context with the turn_ids it wants to evict.
COMPACT_CONTEXT_AGENT = (
    "[STEERING — COMPACT CONTEXT]\n\n"
    "Your active context has reached {current_tokens:,} / {threshold:,} tokens "
    "({pct:.0f} %).  You must reduce it before continuing.\n\n"
    "── What you must do ──────────────────────────────────────────────────────\n"
    "Call `compact_context` with a list of `compacted_turn_ids` chosen from the "
    "inventory below.  Choose conversations whose content you no longer need.  "
    "You must evict enough pairs to bring context back to ≤40 % of the threshold "
    "({target:,} tokens).  Whole pairs only — both the user message and the "
    "assistant response are removed together.\n\n"
    "── Rules ─────────────────────────────────────────────────────────────────\n"
    "• Only Category B pairs (past turns listed below) may be evicted.\n"
    "• The current turn (turn_id={current_turn_id!r}) CANNOT be touched.\n"
    "• You MUST call compact_context — it is the only tool available right now.\n\n"
    "── Available Category B pairs (past Q&A — safe to evict) ────────────────\n"
    "{pairs_inventory}\n\n"
    "Call `compact_context` now."
)

COMPACT_CONTEXT_USER = (
    "[Agent] Context ceiling hit — entering compaction phase. "
    "Agent is choosing which past Q&A pairs to evict."
)


def junction_compact_context(
    pairs_inventory: str,
    current_tokens: int,
    threshold: int,
    current_turn_id: str,
) -> SteeringJunction:
    pct = current_tokens / threshold * 100 if threshold else 0.0
    target = int(threshold * 0.40)
    return SteeringJunction(
        key="compact_context",
        agent_msg=COMPACT_CONTEXT_AGENT.format(
            current_tokens=current_tokens,
            threshold=threshold,
            pct=pct,
            target=target,
            current_turn_id=current_turn_id,
            pairs_inventory=pairs_inventory,
        ),
        user_terminal_msg=COMPACT_CONTEXT_USER,
        restrict_to_mode=None,
        tool_choice=None,  # enforced at call-site via tool_choice="required"
    )


def worker_started_user_msg(task_id: str) -> str:
    return (
        f"[Worker] Fresh coffee, clean context — cracking `.aca/active/{task_id}/`. "
        "Let's ship this thing."
    )


def worker_finished_user_msg(task_id: str) -> str:
    return (
        f"[Worker] Dusting hands off — done with `.aca/active/{task_id}/`. "
        "James, your move."
    )


def james_wake_user_msg(task_id: str) -> str:
    return (
        f"[James] Back from the delegation nap — reading what Worker left in "
        f"`.aca/active/{task_id}/`."
    )
