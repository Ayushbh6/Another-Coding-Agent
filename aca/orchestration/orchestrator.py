from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from aca.agent import AgentRunResult, create_agent
from aca.approval import ApprovalPolicy
from aca.core_types import TurnIntent
from aca.llm.types import HistoryMode, Message, ToolCall
from aca.llm.providers.base import LLMProvider
from aca.orchestration.prompts import ANALYZE_WORKER_PROMPT, NEON_PROMPT
from aca.orchestration.repo_summary import RepoSummaryService
from aca.orchestration.state import (
    ACTOR_ANALYZE_WORKER,
    ACTOR_NEON,
    ANALYZE_DELEGATED_ROUTE,
    ANALYZE_SIMPLE_ROUTE,
    ARCHIVE_DELAY,
    PHASE_DELEGATED_WAIT,
    PHASE_ORIENTATION,
    PHASE_SYNTHESIZE,
    PHASE_TASK_CREATED,
    PHASE_TODO_READY,
    STEERING_IMPLEMENT_DISABLED,
    STEERING_ORIENTATION_DELEGATED,
    STEERING_ORIENTATION_SIMPLE,
    STEERING_PLAN_REQUIRED,
    STEERING_PRETASK_LIMIT_REACHED,
    STEERING_READ_FINDINGS,
    STEERING_SPAWN_WORKER_REQUIRED,
    STEERING_TASK_REQUIRED_NOW,
    STEERING_TODO_ITEM_REQUIRED,
    STEERING_TODO_REQUIRED,
    STEERING_TODO_REVIEW,
    STEERING_WORKER_CONSOLIDATING,
    AnalyzeWorkerState,
    NeonGuardrailError,
    NeonRunState,
    OrchestratedStreamEvent,
)
from aca.orchestration.steering import steering_message
from aca.orchestration.tools import OrchestrationToolRegistry
from aca.orchestration.todo import todo_in_progress, todo_is_complete
from aca.services.chat import ConversationSummary
from aca.storage.models import (
    Action,
    AgentMessage,
    Checkpoint,
    Conversation,
    ConversationMessage,
    MemoryEntry,
    Task,
    User,
)
from aca.task_workspace import TaskWorkspaceManager, parse_task_markdown, parse_todo_markdown
from aca.token_utils import estimate_json_tokens, estimate_text_tokens
from aca.workspace_tools import ToolPermissionError, WorkspaceToolContext, WorkspaceToolRegistry


class NeonOrchestrator:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        provider: LLMProvider,
        approval_policy: ApprovalPolicy | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
        archive_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._approval_policy = approval_policy
        self._workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self._archive_root = Path(archive_root or (Path.home() / ".aca" / "task-archives")).expanduser().resolve()
        self._workspace_manager = TaskWorkspaceManager(self._workspace_root, self._archive_root)
        self._repo_summary_service = RepoSummaryService(self._workspace_root)

    def sweep_archives(self) -> None:
        now = datetime.utcnow()
        with self._session_factory.begin() as session:
            tasks = session.scalars(select(Task).where(Task.completed_at.is_not(None))).all()
            for task in tasks:
                metadata = dict(task.metadata_json or {})
                if metadata.get("archived"):
                    continue
                active_task_dir = metadata.get("active_task_dir")
                archive_due_at_raw = metadata.get("archive_due_at")
                if not active_task_dir or not archive_due_at_raw:
                    continue
                try:
                    archive_due_at = datetime.fromisoformat(str(archive_due_at_raw))
                except ValueError:
                    continue
                if archive_due_at > now:
                    continue

                try:
                    _, destination = self._workspace_manager.archive_task(task.id)
                except FileNotFoundError:
                    metadata["archived"] = True
                    metadata["archive_status"] = "missing_source"
                    task.metadata_json = metadata
                    continue

                metadata["archived"] = True
                metadata["archive_status"] = "archived"
                metadata["archived_task_dir"] = str(destination)
                metadata["archived_at"] = now.isoformat()
                task.metadata_json = metadata
                self._log_action(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    action_type="task_archived",
                    target=str(destination),
                    input_json={"active_task_dir": active_task_dir},
                    result_json={"archived_task_dir": str(destination)},
                    status="success",
                )
                self._write_checkpoint(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    phase="ARCHIVE",
                    checkpoint_kind="archived",
                    summary="Task workspace archived out of the repo.",
                    state_json={"archived_task_dir": str(destination)},
                )

    def stream_turn(
        self,
        *,
        conversation_id: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
    ) -> Iterator[OrchestratedStreamEvent]:
        self.sweep_archives()

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            user = session.get(User, conversation.user_id) if conversation.user_id else None

            full_history = self._load_full_history(session, conversation.id)
            carryover_messages = full_history
            visible_task_count = session.scalar(
                select(func.count()).select_from(Task).where(Task.conversation_id == conversation.id)
            )
            if conversation.message_count == 0 and not visible_task_count:
                conversation.title = user_input.strip()[:40] or "New chat"
            conversation.active_model = model
            conversation.thinking_enabled = thinking_enabled
            self._insert_conversation_message(
                session,
                conversation=conversation,
                role="user",
                message_kind="user",
                content_text=user_input,
                model_name=model,
            )

        state = NeonRunState(
            conversation_id=conversation_id,
            user_input=user_input,
            user_name=user.name if user is not None else None,
            planned_task_id=self._new_id("task"),
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
        )

        yield OrchestratedStreamEvent(
            type="phase.started",
            agent="neon",
            phase="route",
            conversation_id=conversation_id,
            message="Routing the request.",
        )

        neon_run = yield from self._stream_neon_path(
            conversation_id=conversation_id,
            state=state,
            model=model,
            thinking_enabled=thinking_enabled,
        )
        final_answer = self._final_text_payload(neon_run)
        final_intent = state.intent or TurnIntent.CHAT

        if state.task_id is not None:
            self._write_completion_artifact(
                state,
                status="completed",
                final_answer=final_answer,
            )

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared during Neon execution.")
            self._insert_conversation_message(
                session,
                conversation=conversation,
                role="assistant",
                message_kind="assistant_final",
                intent=final_intent,
                content_text=final_answer,
                model_name=model,
                input_tokens=state.turn_input_tokens or None,
                output_tokens=state.turn_output_tokens or None,
                total_tokens=state.turn_total_tokens or None,
                metadata_json={"orchestration": "neon"},
            )

            if state.task_id is not None:
                task = session.get(Task, state.task_id)
                if task is not None:
                    task.status = "completed"
                    task.phase = "COMPLETE"
                    task.completed_at = datetime.utcnow()
                    metadata = dict(task.metadata_json or {})
                    metadata["archive_due_at"] = (task.completed_at + ARCHIVE_DELAY).isoformat()
                    metadata.setdefault("artifact_paths", {}).update(state.artifact_paths)
                    task.metadata_json = metadata
                    task.spec_json = {**task.spec_json, "route": state.route, "final_answer": final_answer}
                    conversation.current_task_id = None
                    self._write_checkpoint(
                        session,
                        task_id=task.id,
                        agent_id="master",
                        phase="COMPLETE",
                        checkpoint_kind="task_completed",
                        summary="Neon orchestration completed.",
                        state_json={"route": state.route, "intent": final_intent.value},
                    )
                    session.add(
                        MemoryEntry(
                            id=self._new_id("mem"),
                            memory_type="episodic_summary",
                            scope_type="task",
                            scope_id=task.id,
                            title=task.title,
                            content=final_answer,
                            source_kind="task_completion",
                            source_id=task.id,
                            importance=2,
                            created_by_agent_id="master",
                            metadata_json={"intent": task.intent, "route": state.route},
                        )
                    )

            summary = self._build_summary(session, conversation)

        yield OrchestratedStreamEvent(
            type="completed",
            agent="neon",
            phase="complete",
            final_answer=final_answer,
            conversation_id=conversation_id,
            task_id=state.task_id,
            intent=final_intent,
            summary=summary,
        )

    def _stream_neon_path(
        self,
        *,
        conversation_id: str,
        state: NeonRunState,
        model: str,
        thinking_enabled: bool,
    ) -> Iterator[AgentRunResult | OrchestratedStreamEvent]:
        carryover_messages = state.carryover_messages
        correction_message = self._run_instructions(state)
        guardrail_failures = 0
        for _cycle in range(12):
            context = WorkspaceToolContext(
                root=self._workspace_root,
                session_factory=self._session_factory,
                task_id=state.task_id,
                agent_id="master",
                approval_policy=None,
            )
            read_registry = WorkspaceToolRegistry(context)
            tool_registry = OrchestrationToolRegistry(
                orchestrator=self,
                actor=ACTOR_NEON,
                route=state.route,
                state=state,
                workspace_manager=self._workspace_manager,
                read_registry=read_registry,
                context=context,
            )
            agent = create_agent(
                name="Neon",
                model=model,
                instructions=NEON_PROMPT,
                provider=self._provider,
                tools=tool_registry.schemas(),
                tool_registry=tool_registry.handlers(),
                max_turns=24,
                reasoning_enabled=thinking_enabled,
                history_mode=HistoryMode.DIALOGUE_ONLY,
            )

            final_result: AgentRunResult | None = None
            handoff_requested = False
            boundary_restart_requested = False
            pending_text_chunks: list[str] = []
            cycle_history_messages: list[Message] = []
            pending_text_phase = "route"
            stream = agent.stream(
                state.user_input,
                carryover_messages=carryover_messages,
                extra_instructions=correction_message,
            )

            def flush_pending_text() -> Iterator[OrchestratedStreamEvent]:
                nonlocal pending_text_chunks
                if not pending_text_chunks:
                    return
                for chunk in pending_text_chunks:
                    yield OrchestratedStreamEvent(
                        type="text.delta",
                        agent="neon",
                        phase=pending_text_phase,
                        conversation_id=conversation_id,
                        task_id=state.task_id,
                        text=chunk,
                    )
                pending_text_chunks = []

            for event in stream:
                if event.type == "reasoning.delta":
                    yield OrchestratedStreamEvent(
                        type="reasoning.delta",
                        agent="neon",
                        phase="route",
                        conversation_id=conversation_id,
                        task_id=state.task_id,
                        thinking_text=event.delta,
                    )
                    continue

                if event.type == "usage.update":
                    state.turn_input_tokens += event.metadata.get("input_tokens", 0)
                    state.turn_output_tokens += event.metadata.get("output_tokens", 0)
                    state.turn_total_tokens += event.metadata.get("total_tokens", 0)
                    continue

                if event.type == "text.delta":
                    if self._should_stream_live_text(state):
                        yield OrchestratedStreamEvent(
                            type="text.delta",
                            agent="neon",
                            phase=pending_text_phase,
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            text=event.delta or "",
                        )
                    else:
                        pending_text_chunks.append(event.delta or "")
                    continue

                if event.type == "runtime.tool_result":
                    yield from flush_pending_text()
                    tool_name = event.tool_call.name if event.tool_call is not None else "unknown"
                    steering_code = self._steering_code_from_result(event.delta)
                    if event.tool_call is not None:
                        cycle_history_messages.append(
                            Message(
                                role="assistant",
                                content=None,
                                tool_calls=[event.tool_call],
                            )
                        )
                    cycle_history_messages.append(
                        Message(
                            role="tool",
                            content=event.delta,
                            tool_call_id=event.tool_call.id if event.tool_call is not None else None,
                            metadata={"tool_name": tool_name},
                        )
                    )
                    yield OrchestratedStreamEvent(
                        type="worker.status",
                        agent="neon",
                        phase="route",
                        conversation_id=conversation_id,
                        task_id=state.task_id,
                        message=self._summarize_tool_result(tool_name, event.delta),
                    )
                    if tool_name == "spawn_analyze_worker" and state.worker_requested and not state.worker_spawned:
                        handoff_requested = True
                        break
                    if tool_name == "write_task_artifact" and steering_code in {
                        STEERING_TODO_ITEM_REQUIRED,
                        STEERING_SPAWN_WORKER_REQUIRED,
                    }:
                        boundary_restart_requested = True
                        break
                    continue

                if event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                    final_result = event.result

            if handoff_requested:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                worker_payload = yield from self._stream_delegated_analyze_worker(
                    conversation_id=conversation_id,
                    state=state,
                )
                state.worker_requested = False
                state.worker_spawned = worker_payload.get("status") == "completed"
                state.phase = PHASE_SYNTHESIZE
                state.worker_summary = str(worker_payload.get("summary", ""))
                correction_message = (
                    f"{self._run_instructions(state)}\n\n"
                    "The delegated worker phase has finished. Read findings.md and completion.json now and synthesize the final answer."
                )
                # Use the actual tool-call history accumulated during this cycle.
                # _delegated_progress_messages produces synthetic role="tool" messages with no
                # tool_call_id and no matching assistant tool_calls, which strict providers like
                # MiniMax reject with "tool result's tool id not found". The cycle_history_messages
                # already contains properly-paired assistant+tool messages from the planning phase.
                carryover_messages = [*carryover_messages, *cycle_history_messages]
                continue

            if boundary_restart_requested:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                carryover_messages = [*carryover_messages, *cycle_history_messages]
                correction_message = self._run_instructions(state)
                continue

            if final_result is None:
                raise RuntimeError("Neon did not complete the run.")

            feedback_code = self._guardrail_feedback_code(state)
            if feedback_code is None:
                yield from flush_pending_text()
                return final_result

            pending_text_chunks = []
            yield OrchestratedStreamEvent(
                type="worker.status",
                agent="neon",
                phase="guardrail",
                conversation_id=conversation_id,
                task_id=state.task_id,
                message=steering_message(feedback_code),
            )
            carryover_messages = final_result.working_history
            guardrail_failures += 1
            correction_message = self._guardrail_recovery_instruction(state, feedback_code, guardrail_failures)
            if guardrail_failures >= 3:
                break

        raise NeonGuardrailError(self._guardrail_feedback_code(state) or STEERING_TASK_REQUIRED_NOW)

    def _should_stream_live_text(self, state: NeonRunState) -> bool:
        if state.task_id is None and state.pretask_read_calls == 0:
            return True
        if state.phase == PHASE_SYNTHESIZE:
            return True
        if state.route == ANALYZE_SIMPLE_ROUTE and state.todo_written and todo_is_complete(state.todo_items):
            return True
        return False

    def _steering_code_from_result(self, raw_result: str | None) -> str | None:
        if not raw_result:
            return None
        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        steering_code = payload.get("steering_code")
        return str(steering_code) if steering_code else None

    def _should_restart_after_tool_result(self, state: NeonRunState, tool_name: str, steering_code: str | None) -> bool:
        if tool_name in {"list_files", "read_file", "search_code"}:
            return steering_code in {
                STEERING_PRETASK_LIMIT_REACHED,
                STEERING_ORIENTATION_SIMPLE,
                STEERING_ORIENTATION_DELEGATED,
            }
        if tool_name == "write_task_artifact":
            return steering_code in {
                STEERING_TODO_ITEM_REQUIRED,
                STEERING_SPAWN_WORKER_REQUIRED,
            }
        return False

    def _phase_restart_instruction(self, state: NeonRunState, steering_code: str | None) -> str:
        base = self._run_instructions(state)
        if steering_code == STEERING_PRETASK_LIMIT_REACHED:
            return (
                f"{base}\n\n"
                "The scout pass is complete. Do not call list_files, read_file, or search_code now. "
                "Your next step must be either a direct chat answer or write_task_artifact(task.md)."
            )
        if steering_code == STEERING_ORIENTATION_SIMPLE:
            return (
                f"{base}\n\n"
                "Orientation is complete. Do not call repo-read tools now. "
                "Your next step must be write_task_artifact(todo.md)."
            )
        if steering_code == STEERING_ORIENTATION_DELEGATED:
            return (
                f"{base}\n\n"
                "Orientation is complete. Do not call repo-read tools now. "
                "Your next step must be write_task_artifact(plan.md), then write_task_artifact(todo.md)."
            )
        if steering_code == STEERING_PLAN_REQUIRED:
            return f"{base}\n\nYour next step must be write_task_artifact(plan.md)."
        if steering_code == STEERING_TODO_REQUIRED:
            return f"{base}\n\nYour next step must be write_task_artifact(todo.md)."
        if steering_code == STEERING_TODO_ITEM_REQUIRED:
            return f"{base}\n\nYour next step must be read_todo_state or start_todo_item for the first pending item."
        if steering_code == STEERING_SPAWN_WORKER_REQUIRED:
            return f"{base}\n\nYour next step must be spawn_analyze_worker."
        if steering_code == STEERING_READ_FINDINGS:
            return f"{base}\n\nYour next step must be read_task_artifact(findings.md) and read_task_artifact(completion.json)."
        return base

    def _preferred_analysis_route(self, state: NeonRunState) -> str:
        lowered = state.user_input.lower()
        broad_markers = ("scan", "codebase", "architecture", "repo", "repository", "system", "how it works")
        return ANALYZE_DELEGATED_ROUTE if any(marker in lowered for marker in broad_markers) else ANALYZE_SIMPLE_ROUTE

    def _guardrail_recovery_instruction(self, state: NeonRunState, feedback_code: str, failure_count: int) -> str:
        base = self._run_instructions(state)
        if feedback_code in {STEERING_PRETASK_LIMIT_REACHED, STEERING_TASK_REQUIRED_NOW}:
            preferred_route = self._preferred_analysis_route(state)
            if failure_count <= 1:
                return (
                    f"{base}\n\n"
                    "Do not answer yet. The scout pass is done. "
                    "Your next action must be write_task_artifact(task.md)."
                )
            return (
                f"{base}\n\n"
                "You must call write_task_artifact next. Do not answer in plain text and do not call repo-read tools.\n"
                "Use this task.md frontmatter shape exactly:\n"
                f"---\n"
                f"task_id: {state.planned_task_id}\n"
                "intent: analyze\n"
                f"route: {preferred_route}\n"
                "title: <short task title>\n"
                "---\n"
                "<normalized task statement>"
            )
        if feedback_code == STEERING_PLAN_REQUIRED:
            return f"{base}\n\nYour next action must be write_task_artifact(plan.md). Do not read artifacts or answer yet."
        if feedback_code == STEERING_TODO_REQUIRED:
            return f"{base}\n\nYour next action must be write_task_artifact(todo.md). Do not answer yet."
        if feedback_code == STEERING_TODO_ITEM_REQUIRED:
            return f"{base}\n\nYour next action must be start_todo_item for the first pending item."
        if feedback_code == STEERING_SPAWN_WORKER_REQUIRED:
            return f"{base}\n\nYour next action must be spawn_analyze_worker. Do not answer yet."
        if feedback_code == STEERING_READ_FINDINGS:
            return f"{base}\n\nYour next actions must be read_task_artifact(findings.md) and read_task_artifact(completion.json), then synthesize."
        return f"{base}\n\nDo not abandon the task. Correct the route or next action now using the allowed tools."

    def _ensure_task_created(self, state: NeonRunState, parsed: Any) -> str:
        if state.task_id is not None:
            return state.task_id

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, state.conversation_id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared while creating a task.")
            task = Task(
                id=state.planned_task_id,
                conversation_id=conversation.id,
                parent_task_id=None,
                intent=TurnIntent.ANALYZE.value,
                title=parsed.title,
                description=parsed.normalized_task,
                status="in_progress",
                phase="EXECUTE",
                priority="normal",
                assigned_to_agent_id="master",
                created_by="master",
                spec_json={"route": parsed.route},
                scope_json={},
                metadata_json={
                    "active_task_dir": str((self._workspace_root / ".aca" / "tasks" / state.planned_task_id).resolve()),
                    "artifact_paths": {},
                    "route": parsed.route,
                    "phase": PHASE_TASK_CREATED,
                    "orientation_read_calls": 0,
                    "todo_state": {"items": [], "current_todo_id": None},
                    "archived": False,
                },
            )
            session.add(task)
            session.flush()
            conversation.current_task_id = task.id
            self._log_action(
                session,
                task_id=task.id,
                agent_id="master",
                action_type="task_created",
                target=task.id,
                input_json={"route": parsed.route, "intent": TurnIntent.ANALYZE.value},
                result_json={"status": task.status, "phase": task.phase},
                status="success",
            )
            self._write_checkpoint(
                session,
                task_id=task.id,
                agent_id="master",
                phase="NEW_TASK",
                checkpoint_kind="task_created",
                summary="Task workspace created from task.md.",
                state_json={"route": parsed.route, "intent": TurnIntent.ANALYZE.value},
            )

        state.task_id = state.planned_task_id
        self._workspace_manager.ensure_task_dir(state.task_id)
        return state.task_id

    def _stream_delegated_analyze_worker(
        self,
        *,
        conversation_id: str,
        state: NeonRunState,
    ) -> Iterator[dict[str, Any] | OrchestratedStreamEvent]:
        if state.task_id is None:
            raise RuntimeError("Delegated analyze worker requires an active task.")

        worker_state = AnalyzeWorkerState(task_id=state.task_id)
        with self._session_factory() as session:
            task = session.get(Task, state.task_id)
            if task is not None:
                metadata = dict(task.metadata_json or {})
                worker_state.orientation_read_calls = int(metadata.get("orientation_read_calls", 0) or 0)
                artifact_paths = metadata.get("artifact_paths", {})
                if isinstance(artifact_paths, dict):
                    worker_state.artifact_paths = {
                        str(name): str(path)
                        for name, path in artifact_paths.items()
                        if isinstance(name, str) and isinstance(path, str)
                    }
                todo_state = dict(metadata.get("todo_state", {}))
                raw_items = todo_state.get("items", [])
                if isinstance(raw_items, list):
                    worker_state.todo_items = [dict(item) for item in raw_items if isinstance(item, dict)]
                current_todo_id = todo_state.get("current_todo_id")
                if isinstance(current_todo_id, str) and current_todo_id:
                    worker_state.current_todo_id = current_todo_id

        context = WorkspaceToolContext(
            root=self._workspace_root,
            session_factory=self._session_factory,
            task_id=state.task_id,
            agent_id="worker",
            approval_policy=None,
        )
        read_registry = WorkspaceToolRegistry(context)
        registry = OrchestrationToolRegistry(
            orchestrator=self,
            actor=ACTOR_ANALYZE_WORKER,
            route=ANALYZE_DELEGATED_ROUTE,
            state=worker_state,
            workspace_manager=self._workspace_manager,
            read_registry=read_registry,
        )
        worker = create_agent(
            name="AnalyzeWorker",
            model=state.model,
            instructions=ANALYZE_WORKER_PROMPT,
            provider=self._provider,
            tools=registry.schemas(),
            tool_registry=registry.handlers(),
            max_turns=20,
            reasoning_enabled=state.thinking_enabled,
            history_mode=HistoryMode.DIALOGUE_ONLY,
        )
        yield OrchestratedStreamEvent(
            type="phase.started",
            agent="worker",
            phase="delegated_analyze",
            conversation_id=conversation_id,
            task_id=state.task_id,
            message="Delegated analyze worker started.",
        )

        carryover_messages: list[Message] = []
        extra_instructions = (
            f"{self._user_identity_instruction(state)}\n"
            "Read task.md, plan.md, and todo.md first. "
            "Execute the todo one item at a time. "
            "Finish by writing findings.md."
        )
        result: AgentRunResult | None = None
        for _attempt in range(3):
            try:
                final_result: AgentRunResult | None = None
                pending_text_chunks: list[str] = []
                for event in worker.stream(
                    "Read the delegated analysis task artifacts, inspect the repository, write findings.md, and stop.",
                    carryover_messages=carryover_messages,
                    extra_instructions=extra_instructions,
                ):
                    if event.type == "text.delta":
                        pending_text_chunks.append(event.delta or "")
                        continue

                    if event.type == "reasoning.delta":
                        yield OrchestratedStreamEvent(
                            type="reasoning.delta",
                            agent="worker",
                            phase="delegated_analyze",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            thinking_text=event.delta,
                        )
                        continue

                    if event.type == "usage.update":
                        state.turn_input_tokens += event.metadata.get("input_tokens", 0)
                        state.turn_output_tokens += event.metadata.get("output_tokens", 0)
                        state.turn_total_tokens += event.metadata.get("total_tokens", 0)
                        continue

                    if event.type == "runtime.tool_result":
                        if pending_text_chunks:
                            for chunk in pending_text_chunks:
                                yield OrchestratedStreamEvent(
                                    type="text.delta",
                                    agent="worker",
                                    phase="delegated_analyze",
                                    conversation_id=conversation_id,
                                    task_id=state.task_id,
                                    text=chunk,
                                )
                            pending_text_chunks = []
                        tool_name = event.tool_call.name if event.tool_call is not None else "unknown"
                        yield OrchestratedStreamEvent(
                            type="worker.status",
                            agent="worker",
                            phase="delegated_analyze",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            message=self._summarize_tool_result(tool_name, event.delta),
                        )
                        continue

                    if event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                        final_result = event.result

                if final_result is None:
                    raise RuntimeError("Analyze worker stream did not complete.")
                if worker_state.findings_written and pending_text_chunks:
                    for chunk in pending_text_chunks:
                        yield OrchestratedStreamEvent(
                            type="text.delta",
                            agent="worker",
                            phase="delegated_analyze",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            text=chunk,
                        )
                result = final_result
            except ToolPermissionError as exc:
                payload = {"status": "failed", "summary": str(exc), "findings_path": None}
                self._write_completion_artifact(state, status="failed", worker_status="failed", worker_summary=str(exc))
                yield OrchestratedStreamEvent(
                    type="phase.completed",
                    agent="worker",
                    phase="delegated_analyze",
                    conversation_id=conversation_id,
                    task_id=state.task_id,
                    message=str(exc),
                    payload=payload,
                )
                return payload

            if worker_state.findings_written:
                break
            if result is None:
                break
            carryover_messages = result.working_history
            extra_instructions = (
                f"{self._user_identity_instruction(state)}\n"
                "Use what you already gathered. "
                "Do not continue broad exploration. "
                "Consolidate the evidence into findings.md now."
            )
            yield OrchestratedStreamEvent(
                type="worker.status",
                agent="worker",
                phase="delegated_analyze",
                conversation_id=conversation_id,
                task_id=state.task_id,
                message=steering_message(STEERING_WORKER_CONSOLIDATING),
            )

        if result is None or not worker_state.findings_written:
            summary = "Analyze worker finished without writing findings.md."
            self._write_completion_artifact(state, status="failed", worker_status="failed", worker_summary=summary)
            payload = {"status": "failed", "summary": summary, "findings_path": None}
            yield OrchestratedStreamEvent(
                type="phase.completed",
                agent="worker",
                phase="delegated_analyze",
                conversation_id=conversation_id,
                task_id=state.task_id,
                message=steering_message(STEERING_WORKER_CONSOLIDATING),
                payload=payload,
            )
            return payload

        self._persist_agent_message(
            state.task_id,
            "master",
            "worker",
            "assignment",
            "delegated_analyze",
            "Complete the delegated repo analysis and write findings.md.",
        )
        self._persist_agent_message(
            state.task_id,
            "worker",
            "master",
            "completion",
            "delegated_findings",
            worker_state.findings_path or "findings.md",
        )
        self._write_completion_artifact(
            state,
            status="worker_completed",
            worker_status="completed",
            worker_summary=self._final_text_payload(result),
        )
        with self._session_factory.begin() as session:
            task = session.get(Task, state.task_id)
            if task is not None:
                task.phase = "REVIEW"
                self._write_checkpoint(
                    session,
                    task_id=task.id,
                    agent_id="worker",
                    phase="REVIEW",
                    checkpoint_kind="worker_completed",
                    summary="Delegated analyze worker completed.",
                    state_json={"findings_path": worker_state.findings_path},
                )

        payload = {
            "status": "completed",
            "summary": "Analyze worker completed.",
            "findings_path": worker_state.findings_path,
        }
        yield OrchestratedStreamEvent(
            type="phase.completed",
            agent="worker",
            phase="delegated_analyze",
            conversation_id=conversation_id,
            task_id=state.task_id,
            message="Analyze worker completed.",
            payload=payload,
        )
        return payload

    def _write_completion_artifact(
        self,
        state: NeonRunState,
        *,
        status: str,
        final_answer: str | None = None,
        worker_status: str | None = None,
        worker_summary: str | None = None,
    ) -> None:
        if state.task_id is None:
            return

        payload: dict[str, Any] = {
            "task_id": state.task_id,
            "intent": (state.intent or TurnIntent.ANALYZE).value,
            "route": state.route,
            "phase": state.phase,
            "status": status,
            "artifact_paths": state.artifact_paths,
            "updated_at": datetime.utcnow().isoformat(),
            "archived": False,
        }
        if final_answer is not None:
            payload["final_answer"] = final_answer
        if worker_status is not None:
            payload["worker_status"] = worker_status
        if worker_summary is not None:
            payload["worker_summary"] = worker_summary

        path = self._workspace_manager.write_completion(state.task_id, payload)
        relative_path = path.relative_to(self._workspace_root).as_posix()
        state.artifact_paths["completion.json"] = relative_path
        self._record_runtime_action(
            task_id=state.task_id,
            agent_id="master",
            action_type="completion_artifact_written",
            target=relative_path,
            input_json={"status": status},
            result_json={"path": relative_path},
        )

        with self._session_factory.begin() as session:
            task = session.get(Task, state.task_id)
            if task is not None:
                metadata = dict(task.metadata_json or {})
                artifact_paths = dict(metadata.get("artifact_paths", {}))
                artifact_paths.update(state.artifact_paths)
                metadata["artifact_paths"] = artifact_paths
                task.metadata_json = metadata

    def _initialize_todo_state(self, state: NeonRunState, raw_content: str) -> list[dict[str, Any]]:
        parsed_items = parse_todo_markdown(raw_content)
        todo_items = [
            {"todo_id": self._new_id("todo"), "title": str(item["title"]).strip(), "status": "pending"}
            for item in parsed_items
        ]
        self._persist_todo_state(
            state,
            action_type="todo_initialized",
            todo_id=None,
            note=None,
            todo_items=todo_items,
        )
        return todo_items

    def _persist_todo_state(
        self,
        state: NeonRunState | AnalyzeWorkerState,
        *,
        agent_id: str = "master",
        action_type: str,
        todo_id: str | None,
        note: str | None,
        todo_items: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if state.task_id is None:
            return
        items = [dict(item) for item in (todo_items if todo_items is not None else state.todo_items)]
        state.todo_items = items
        todo_path = self._workspace_manager.write_todo_state(state.task_id, items)
        relative_path = todo_path.relative_to(self._workspace_root).as_posix()
        state.artifact_paths["todo.md"] = relative_path

        with self._session_factory.begin() as session:
            task = session.get(Task, state.task_id)
            if task is not None:
                metadata = dict(task.metadata_json or {})
                artifact_paths = dict(metadata.get("artifact_paths", {}))
                artifact_paths["todo.md"] = relative_path
                metadata["artifact_paths"] = artifact_paths
                metadata["phase"] = state.phase
                metadata["orientation_read_calls"] = state.orientation_read_calls
                metadata["todo_state"] = {"items": items, "current_todo_id": state.current_todo_id}
                task.metadata_json = metadata
                self._log_action(
                    session,
                    task_id=state.task_id,
                    agent_id=agent_id,
                    action_type=action_type,
                    target=todo_id or relative_path,
                    input_json={"todo_id": todo_id, "note": note or "", **(extra or {})},
                    result_json={"path": relative_path, "count": len(items), "current_todo_id": state.current_todo_id},
                    status="success",
                )
                checkpoint_kind = {
                    "todo_initialized": "todo_ready",
                    "todo_item_started": "todo_execution_started",
                    "todo_item_completed": "todo_item_completed",
                    "todo_item_skipped": "todo_item_skipped",
                    "todo_revised": "todo_revised",
                }.get(action_type, "todo_updated")
                self._write_checkpoint(
                    session,
                    task_id=state.task_id,
                    agent_id=agent_id,
                    phase=state.phase.upper(),
                    checkpoint_kind=checkpoint_kind,
                    summary=note or action_type.replace("_", " "),
                    state_json={"todo_id": todo_id, "current_todo_id": state.current_todo_id, "items": items, **(extra or {})},
                )

    def _steering_instructions(self, state: NeonRunState) -> str:
        base = (
            f"If you take a non-chat route, use task_id `{state.planned_task_id}` in task.md frontmatter exactly. "
            "Do not invent another task id."
        )
        if not state.task_written:
            return (
                f"{base}\n"
                "You may use get_repo_summary and up to 2 pre-task repo reads before you must either answer as chat or write task.md."
            )
        if state.route == ANALYZE_SIMPLE_ROUTE:
            if not state.todo_written:
                return f"{base}\nTask created. You may now do up to 2 orientation reads, then write todo.md."
            if todo_is_complete(state.todo_items):
                return f"{base}\nTodo execution is complete. Synthesize the final answer now."
            current = todo_in_progress(state.todo_items)
            if current is None:
                return f"{base}\nTodo is ready. Start one todo item before continuing the analysis."
            return f"{base}\nCurrent todo item in progress: {current['title']}. Continue or finish it before moving on."
        if state.route == ANALYZE_DELEGATED_ROUTE:
            if not state.plan_written:
                return f"{base}\nTask created. You may now do up to 2 orientation reads, then write plan.md."
            if not state.todo_written:
                return f"{base}\nPlan is ready. Write todo.md next."
            if not state.worker_spawned:
                return f"{base}\nTodo is ready. Your next action must be spawn_analyze_worker."
            return f"{base}\nThe worker has finished. Read findings.md and completion.json, then synthesize the final answer."
        return f"{base}\nIf the user asked for code changes, implementation mode is disabled. Answer directly without creating a task."

    def _user_identity_instruction(self, state: NeonRunState) -> str:
        if state.user_name:
            return f"You are currently assisting the user named {state.user_name}."
        return "You are currently assisting the active CLI user for this conversation."

    def _run_instructions(self, state: NeonRunState) -> str:
        return f"{self._user_identity_instruction(state)}\n\n{self._steering_instructions(state)}"

    def _guardrail_feedback_code(self, state: NeonRunState) -> str | None:
        if state.task_id is None:
            if state.pretask_read_calls >= 2:
                return STEERING_PRETASK_LIMIT_REACHED
            if state.pretask_read_calls > 0:
                return STEERING_TASK_REQUIRED_NOW
            return None

        if state.route == ANALYZE_SIMPLE_ROUTE:
            if not state.todo_written:
                return STEERING_TODO_REQUIRED
            if todo_in_progress(state.todo_items) is None and not todo_is_complete(state.todo_items):
                return STEERING_TODO_ITEM_REQUIRED
            if not todo_is_complete(state.todo_items):
                return STEERING_TODO_REVIEW
            return None

        if state.route == ANALYZE_DELEGATED_ROUTE:
            if not state.plan_written:
                return STEERING_PLAN_REQUIRED
            if not state.todo_written:
                return STEERING_TODO_REQUIRED
            if not state.worker_spawned:
                return STEERING_SPAWN_WORKER_REQUIRED
            return None

        return STEERING_IMPLEMENT_DISABLED

    def _delegated_progress_messages(self, state: NeonRunState) -> list[Message]:
        messages: list[Message] = []
        for artifact_name in ("task.md", "plan.md", "todo.md"):
            if artifact_name in state.artifact_paths:
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps({"artifact_name": artifact_name, "path": state.artifact_paths[artifact_name]}),
                        metadata={"tool_name": "write_task_artifact"},
                    )
                )
        messages.append(
            Message(
                role="tool",
                content=json.dumps({"status": "scheduled", "summary": "Analyze worker scheduled."}),
                metadata={"tool_name": "spawn_analyze_worker"},
            )
        )
        if "findings.md" in state.artifact_paths:
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps({"artifact_name": "findings.md", "path": state.artifact_paths["findings.md"]}),
                    metadata={"tool_name": "read_task_artifact"},
                )
            )
        if "completion.json" in state.artifact_paths:
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps({"artifact_name": "completion.json", "path": state.artifact_paths["completion.json"]}),
                    metadata={"tool_name": "read_task_artifact"},
                )
            )
        return messages

    def _record_artifact_write(self, *, task_id: str, agent_id: str, artifact_name: str, relative_path: str) -> None:
        self._record_runtime_action(
            task_id=task_id,
            agent_id=agent_id,
            action_type="task_artifact_written",
            target=relative_path,
            input_json={"artifact_name": artifact_name},
            result_json={"path": relative_path},
        )
        with self._session_factory.begin() as session:
            task = session.get(Task, task_id)
            if task is not None:
                metadata = dict(task.metadata_json or {})
                artifact_paths = dict(metadata.get("artifact_paths", {}))
                artifact_paths[artifact_name] = relative_path
                metadata["artifact_paths"] = artifact_paths
                task.metadata_json = metadata

    def _update_task_metadata_after_task_rewrite(self, state: NeonRunState, parsed: Any) -> None:
        if state.task_id is None:
            return
        with self._session_factory.begin() as session:
            task = session.get(Task, state.task_id)
            if task is None:
                return
            task.title = parsed.title
            task.description = parsed.normalized_task
            task.spec_json = {**task.spec_json, "route": parsed.route}
            metadata = dict(task.metadata_json or {})
            metadata["route"] = parsed.route
            metadata["phase"] = state.phase
            metadata["todo_state"] = {"items": [], "current_todo_id": None}
            task.metadata_json = metadata

    def _record_runtime_action(
        self,
        *,
        task_id: str | None,
        agent_id: str,
        action_type: str,
        target: str | None,
        input_json: dict[str, Any],
        result_json: dict[str, Any],
        status: str = "success",
        error_text: str | None = None,
    ) -> None:
        with self._session_factory.begin() as session:
            self._log_action(
                session,
                task_id=task_id,
                agent_id=agent_id,
                action_type=action_type,
                target=target,
                input_json=input_json,
                result_json=result_json,
                status=status,
                error_text=error_text,
            )

    def _web_search(self, *, query: str, limit: int) -> dict[str, Any]:
        encoded = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
        url = f"https://api.duckduckgo.com/?{encoded}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"query": query, "results": [], "count": 0, "error": str(exc)}

        results: list[dict[str, Any]] = []
        abstract = str(payload.get("AbstractText", "")).strip()
        if abstract:
            results.append({"title": str(payload.get("Heading", "Result")).strip() or "Result", "snippet": abstract})
        for topic in payload.get("RelatedTopics", []):
            if not isinstance(topic, dict):
                continue
            if "Topics" in topic:
                for nested in topic.get("Topics", []):
                    if isinstance(nested, dict) and nested.get("Text"):
                        results.append({"title": str(nested.get("FirstURL", "")), "snippet": str(nested["Text"])})
            elif topic.get("Text"):
                results.append({"title": str(topic.get("FirstURL", "")), "snippet": str(topic["Text"])})
            if len(results) >= max(1, limit):
                break
        return {"query": query, "results": results[: max(1, limit)], "count": len(results[: max(1, limit)])}

    def _summarize_tool_result(self, tool_name: str, raw_result: str | None) -> str:
        payload: dict[str, Any] | None = None
        if raw_result:
            try:
                parsed = json.loads(raw_result)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = None

        if payload is not None and payload.get("error"):
            error_text = str(payload["error"]).strip()
            return f"{tool_name} hit a guardrail: {error_text}"

        steering_suffix = ""
        if payload is not None and payload.get("steering_code"):
            steering_suffix = f" | {steering_message(str(payload['steering_code']))}"

        if tool_name == "list_files" and payload is not None:
            count = payload.get("count", 0)
            files = payload.get("files", [])
            scope = payload.get("path", ".")
            sample = ", ".join(str(item) for item in files[:3]) if isinstance(files, list) else ""
            return f"listed {count} files under {scope}" + (f": {sample}" if sample else "") + steering_suffix

        if tool_name == "read_file" and payload is not None:
            path = payload.get("path", "file")
            start_line = payload.get("start_line")
            end_line = payload.get("end_line")
            truncated = payload.get("truncated", False)
            if isinstance(start_line, int) and isinstance(end_line, int):
                slice_label = f"lines {start_line}-{end_line}"
            else:
                content = str(payload.get("content", ""))
                slice_label = f"{len(content.splitlines())} lines"
            return f"read {path} {slice_label}" + (" (truncated)" if truncated else "") + steering_suffix

        if tool_name == "search_code" and payload is not None:
            count = payload.get("count", 0)
            matches = payload.get("matches", [])
            sample = ""
            if isinstance(matches, list) and matches:
                first = matches[0]
                if isinstance(first, dict) and first.get("path"):
                    sample = str(first["path"])
                    if first.get("line"):
                        sample += f":{first['line']}"
            return f"found {count} matches" + (f" starting at {sample}" if sample else "") + steering_suffix

        if tool_name == "get_repo_summary" and payload is not None:
            stack = ", ".join(str(item) for item in payload.get("likely_stack", [])[:3])
            return "repo summary ready" + (f": {stack}" if stack else "")

        if tool_name == "write_task_artifact" and payload is not None:
            artifact_name = payload.get("artifact_name", "artifact")
            summary = str(payload.get("summary") or f"wrote {artifact_name}")
            return summary + steering_suffix

        if tool_name == "read_task_artifact" and payload is not None:
            return f"read {payload.get('artifact_name', 'artifact')}"

        if tool_name == "read_todo_state" and payload is not None:
            return f"read todo state ({len(payload.get('items', []))} items)"

        if tool_name in {"start_todo_item", "complete_todo_item", "skip_todo_item", "revise_todo"} and payload is not None:
            return str(payload.get("summary", tool_name))

        if tool_name == "spawn_analyze_worker" and payload is not None:
            return str(payload.get("summary", "Analyze worker scheduled."))

        if tool_name == "write_repo_doc" and payload is not None:
            return f"updated {payload.get('doc_name', 'repo doc')}"

        if tool_name == "web_search" and payload is not None:
            return f"web search returned {payload.get('count', 0)} results"

        compact = (raw_result or "").strip()
        if len(compact) > 160:
            compact = compact[:157] + "..."
        return f"{tool_name}: {compact}" if compact else tool_name

    def _final_text_payload(self, run_result: AgentRunResult) -> str:
        if run_result.provider_runs:
            last = run_result.provider_runs[-1]
            if last.text:
                return last.text
        return run_result.final_answer or ""

    @staticmethod
    def _accumulate_usage(state: NeonRunState, run_result: AgentRunResult) -> None:
        """Sum real API token usage from all provider runs into the turn-level accumulators."""
        for pr in run_result.provider_runs:
            u = pr.usage
            state.turn_input_tokens += u.input_tokens or 0
            state.turn_output_tokens += u.output_tokens or 0
            state.turn_total_tokens += u.total_tokens or 0

    def _build_summary(self, session: Session, conversation: Conversation) -> ConversationSummary:
        message_aggregates = session.execute(
            select(
                func.count(ConversationMessage.id),
                func.coalesce(func.sum(ConversationMessage.input_tokens), 0),
                func.coalesce(func.sum(ConversationMessage.output_tokens), 0),
                func.coalesce(func.sum(ConversationMessage.total_tokens), 0),
            ).where(ConversationMessage.conversation_id == conversation.id)
        ).one()
        return ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            visible_message_count=int(message_aggregates[0] or 0),
            active_model=conversation.active_model,
            thinking_enabled=conversation.thinking_enabled,
            total_input_tokens=int(message_aggregates[1] or 0),
            total_output_tokens=int(message_aggregates[2] or 0),
            total_tokens=int(message_aggregates[3] or 0),
            updated_at=conversation.updated_at,
        )

    def _load_full_history(self, session: Session, conversation_id: str) -> list[Message]:
        rows = session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.sequence_no)
        ).all()
        messages: list[Message] = []
        for row in rows:
            if row.role == "assistant" and row.message_kind == "assistant_tool_request":
                tool_calls = []
                payload = row.content_json or {}
                for raw_tool_call in payload.get("tool_calls", []):
                    tool_calls.append(
                        ToolCall(
                            id=str(raw_tool_call.get("id", "")),
                            name=str(raw_tool_call.get("name", "")),
                            arguments=str(raw_tool_call.get("arguments", "")),
                        )
                    )
                messages.append(
                    Message(
                        role="assistant",
                        content=row.content_text,
                        tool_calls=tool_calls,
                        reasoning=payload.get("reasoning"),
                        reasoning_details=payload.get("reasoning_details", []),
                    )
                )
                continue

            if row.role == "tool":
                messages.append(
                    Message(
                        role="tool",
                        content=row.content_text,
                        tool_call_id=row.tool_call_id,
                        metadata={"tool_name": row.tool_name} if row.tool_name else {},
                    )
                )
                continue

            payload = row.content_json or {}
            messages.append(
                Message(
                    role=row.role,
                    content=row.content_text,
                    reasoning=payload.get("reasoning") if row.role == "assistant" else None,
                    reasoning_details=payload.get("reasoning_details", []) if row.role == "assistant" else [],
                )
            )
        return messages

    def _insert_conversation_message(
        self,
        session: Session,
        *,
        conversation: Conversation,
        role: str,
        message_kind: str,
        intent: TurnIntent | None = None,
        content_text: str | None = None,
        content_json: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        response_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        latency_ms: float | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        sequence_no = self._next_message_sequence(session, conversation.id)
        text_token_estimate = estimate_text_tokens(content_text, model=model_name)
        if content_json:
            text_token_estimate += estimate_json_tokens(content_json, model=model_name)

        resolved_input = input_tokens if input_tokens is not None else (text_token_estimate if role in {"user", "tool"} else 0)
        resolved_output = output_tokens if output_tokens is not None else (text_token_estimate if role == "assistant" else 0)
        resolved_total = total_tokens if total_tokens is not None else resolved_input + resolved_output
        resolved_reasoning = reasoning_tokens if reasoning_tokens is not None else 0

        row = ConversationMessage(
            id=self._new_id("msg"),
            conversation_id=conversation.id,
            sequence_no=sequence_no,
            role=role,
            message_kind=message_kind,
            intent=intent.value if intent is not None else None,
            content_text=content_text,
            content_json=content_json,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            provider_name=provider_name,
            model_name=model_name,
            response_id=response_id,
            text_token_estimate=text_token_estimate,
            input_tokens=resolved_input,
            output_tokens=resolved_output,
            total_tokens=resolved_total,
            reasoning_tokens=resolved_reasoning,
            latency_ms=latency_ms,
            metadata_json=metadata_json or {},
        )
        session.add(row)

        now = datetime.utcnow()
        conversation.message_count += 1
        conversation.total_input_tokens += resolved_input
        conversation.total_output_tokens += resolved_output
        conversation.total_tokens += resolved_total
        conversation.last_message_at = now
        conversation.updated_at = now
        return row

    def _log_action(
        self,
        session: Session,
        *,
        task_id: str | None,
        agent_id: str,
        action_type: str,
        target: str | None,
        input_json: dict[str, Any],
        result_json: dict[str, Any],
        status: str,
        error_text: str | None = None,
    ) -> None:
        session.add(
            Action(
                id=self._new_id("act"),
                task_id=task_id,
                agent_id=agent_id,
                action_type=action_type,
                target=target,
                input_json=input_json,
                result_json=result_json,
                status=status,
                completed_at=datetime.utcnow(),
                error_text=error_text,
                metadata_json={},
            )
        )

    def _persist_agent_message(
        self,
        task_id: str,
        from_agent_id: str,
        to_agent_id: str,
        channel: str,
        message_type: str,
        content: str,
    ) -> None:
        with self._session_factory.begin() as session:
            self._log_agent_message(
                session,
                task_id=task_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                channel=channel,
                message_type=message_type,
                content=content,
            )

    def _log_agent_message(
        self,
        session: Session,
        *,
        task_id: str | None,
        from_agent_id: str,
        to_agent_id: str,
        channel: str,
        message_type: str,
        content: str,
    ) -> None:
        session.add(
            AgentMessage(
                id=self._new_id("amsg"),
                task_id=task_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                channel=channel,
                message_type=message_type,
                content=content,
                round_no=1,
                status="sent",
                metadata_json={},
            )
        )

    def _write_checkpoint(
        self,
        session: Session,
        *,
        task_id: str | None,
        agent_id: str,
        phase: str,
        checkpoint_kind: str,
        summary: str,
        state_json: dict[str, Any],
        resume_cursor: str | None = None,
    ) -> None:
        session.add(
            Checkpoint(
                id=self._new_id("ckpt"),
                task_id=task_id,
                agent_id=agent_id,
                phase=phase,
                checkpoint_kind=checkpoint_kind,
                summary=summary,
                state_json=state_json,
                resume_cursor=resume_cursor,
                metadata_json={},
            )
        )

    def _next_message_sequence(self, session: Session, conversation_id: str) -> int:
        current = session.scalar(
            select(func.max(ConversationMessage.sequence_no)).where(
                ConversationMessage.conversation_id == conversation_id
            )
        )
        return int(current or 0) + 1

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"
