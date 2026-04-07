from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from aca.core_types import TurnIntent
from aca.llm.types import Message
from aca.services.chat import ConversationSummary


ARCHIVE_DELAY = timedelta(hours=12)
MAX_PRETASK_READS = 2
MAX_ORIENTATION_READS = 2

ACTOR_NEON = "neon"
ACTOR_ANALYZE_WORKER = "analyze_worker"
ACTOR_IMPLEMENT_WORKER = "implement_worker"

ANALYZE_SIMPLE_ROUTE = "analyze_simple"
ANALYZE_DELEGATED_ROUTE = "analyze_delegated"
IMPLEMENT_ROUTE = "implement"
ANALYZE_ROUTES = {ANALYZE_SIMPLE_ROUTE, ANALYZE_DELEGATED_ROUTE}

PHASE_PRETASK = "pretask"
PHASE_TASK_CREATED = "task_created"
PHASE_ORIENTATION = "orientation"
PHASE_TODO_READY = "todo_ready"
PHASE_DELEGATED_WAIT = "delegated_wait"
PHASE_SYNTHESIZE = "synthesize"

TASK_ARTIFACT_NAMES = ("task.md", "plan.md", "todo.md", "findings.md", "output.md", "completion.json")

STEERING_PRETASK_LIMIT_REACHED = "pretask_limit_reached"
STEERING_TASK_REQUIRED_NOW = "task_required_now"
STEERING_ORIENTATION_SIMPLE = "orientation_complete_simple"
STEERING_ORIENTATION_DELEGATED = "orientation_complete_delegated"
STEERING_PLAN_REQUIRED = "plan_required"
STEERING_TODO_REQUIRED = "todo_required"
STEERING_TODO_ITEM_REQUIRED = "todo_item_required"
STEERING_TODO_REVIEW = "todo_review"
STEERING_SPAWN_WORKER_REQUIRED = "spawn_worker_required"
STEERING_READ_FINDINGS = "read_findings"
STEERING_READ_OUTPUT = "read_output"
STEERING_WORKER_CONSOLIDATING = "worker_consolidating_findings"
STEERING_IMPLEMENT_WORKER_CONSOLIDATING = "worker_consolidating_output"
STEERING_WORKER_WRITE_FINDINGS_NOW = "worker_write_findings_now"
STEERING_WORKER_WRITE_OUTPUT_NOW = "worker_write_output_now"
STEERING_ROUTE_RECOVERY = "route_recovery"


class NeonGuardrailError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


@dataclass(slots=True)
class OrchestratedStreamEvent:
    type: str
    agent: str | None = None
    phase: str | None = None
    text: str | None = None
    thinking_text: str | None = None
    message: str | None = None
    final_answer: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None
    intent: TurnIntent | None = None
    summary: ConversationSummary | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NeonRunState:
    conversation_id: str
    user_input: str
    user_name: str | None
    planned_task_id: str
    model: str
    thinking_enabled: bool
    carryover_messages: list[Message]
    phase: str = PHASE_PRETASK
    pretask_read_calls: int = 0
    orientation_read_calls: int = 0
    task_id: str | None = None
    route: str | None = None
    intent: TurnIntent | None = None
    task_title: str = ""
    task_statement: str = ""
    task_written: bool = False
    plan_written: bool = False
    todo_written: bool = False
    todo_items: list[dict[str, Any]] = field(default_factory=list)
    current_todo_id: str | None = None
    worker_requested: bool = False
    worker_spawned: bool = False
    worker_summary: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    repo_summary_used: bool = False
    # Real API token usage across all Neon + worker LLM calls in this turn.
    # input_tokens: peak context window (max across calls — each call already includes prior messages).
    # output_tokens: cumulative generation (sum across calls).
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0


@dataclass(slots=True)
class AnalyzeWorkerState:
    task_id: str
    phase: str = "delegated_execute"
    orientation_read_calls: int = 0
    todo_items: list[dict[str, Any]] = field(default_factory=list)
    current_todo_id: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    findings_written: bool = False
    findings_path: str | None = None


@dataclass(slots=True)
class ImplementWorkerState:
    task_id: str
    phase: str = "delegated_execute"
    orientation_read_calls: int = 0
    todo_items: list[dict[str, Any]] = field(default_factory=list)
    current_todo_id: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    output_written: bool = False
    output_path: str | None = None
