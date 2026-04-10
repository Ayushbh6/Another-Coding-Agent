from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from aca.agent import AgentRunResult, create_agent
from aca.approval import ApprovalPolicy
from aca.core_types import TurnIntent
from aca.llm.types import HistoryMode, Message
from aca.llm.providers.base import LLMProvider
from aca.orchestration.prompts import ANALYZE_WORKER_PROMPT, IMPLEMENT_WORKER_PROMPT, NEON_PROMPT
from aca.orchestration.guardrails import GuardrailMixin
from aca.orchestration.history import HistoryMixin
from aca.orchestration.persistence import PersistenceMixin
from aca.orchestration.repo_summary import RepoSummaryService
from aca.orchestration.state import (
    ACTOR_ANALYZE_WORKER,
    ACTOR_IMPLEMENT_WORKER,
    ACTOR_NEON,
    ANALYZE_DELEGATED_ROUTE,
    ANALYZE_SIMPLE_ROUTE,
    ARCHIVE_DELAY,
    IMPLEMENT_ROUTE,
    PHASE_DELEGATED_WAIT,
    PHASE_ORIENTATION,
    PHASE_SYNTHESIZE,
    PHASE_TASK_CREATED,
    PHASE_TODO_READY,
    STEERING_ORIENTATION_DELEGATED,
    STEERING_ORIENTATION_SIMPLE,
    STEERING_IMPLEMENT_WORKER_CONSOLIDATING,
    STEERING_PRETASK_LIMIT_REACHED,
    STEERING_SPAWN_WORKER_REQUIRED,
    STEERING_TASK_REQUIRED_NOW,
    STEERING_TODO_ITEM_REQUIRED,
    STEERING_WORKER_CONSOLIDATING,
    STEERING_WORKER_WRITE_FINDINGS_NOW,
    STEERING_WORKER_WRITE_OUTPUT_NOW,
    AnalyzeWorkerState,
    ImplementWorkerState,
    NeonGuardrailError,
    NeonRunState,
    OrchestratedStreamEvent,
)
from aca.orchestration.steering import steering_message
from aca.orchestration.tools import OrchestrationToolRegistry
from aca.orchestration.todo import todo_is_complete
from aca.storage.models import (
    Conversation,
    MemoryEntry,
    Task,
    User,
)
from aca.task_workspace import TaskWorkspaceManager, parse_task_markdown, parse_todo_markdown
from aca.tools import PathAccessManager, ToolPermissionError, WorkspaceToolContext, WorkspaceToolRegistry


class NeonOrchestrator(PersistenceMixin, HistoryMixin, GuardrailMixin):
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        provider: LLMProvider,
        approval_policy: ApprovalPolicy | None = None,
        path_access_manager: PathAccessManager | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
        archive_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._approval_policy = approval_policy
        self._path_access_manager = path_access_manager
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
                total_tokens=(state.turn_input_tokens + state.turn_output_tokens) or None,
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
                approval_policy=self._approval_policy,
                path_access_manager=self._path_access_manager,
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
            boundary_steering_code: str | None = None
            restart_with_phase_instruction = False
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
                    state.turn_input_tokens = max(state.turn_input_tokens, event.metadata.get("input_tokens", 0))
                    state.turn_output_tokens += event.metadata.get("output_tokens", 0)
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
                    tool_ok = event.metadata.get("ok", True)
                    if not tool_ok:
                        self._record_runtime_action(
                            task_id=state.task_id,
                            agent_id="master",
                            action_type=f"tool:{tool_name}",
                            target=None,
                            input_json=event.metadata.get("arguments", {}),
                            result_json={"raw": (event.delta or "")[:500]},
                            status="failed",
                            error_text=(event.delta or "")[:1000],
                        )
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
                    if tool_name in {"spawn_analyze_worker", "spawn_implement_worker"} and state.worker_requested and not state.worker_spawned:
                        handoff_requested = True
                        break
                    if self._should_restart_neon_cycle(
                        tool_name=tool_name,
                        steering_code=steering_code,
                        state=state,
                    ):
                        boundary_restart_requested = True
                        boundary_steering_code = steering_code
                        restart_with_phase_instruction = steering_code in {
                            STEERING_PRETASK_LIMIT_REACHED,
                            STEERING_ORIENTATION_SIMPLE,
                            STEERING_ORIENTATION_DELEGATED,
                        }
                        break
                    continue

                if event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                    final_result = event.result

            if handoff_requested:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                if state.route == IMPLEMENT_ROUTE:
                    worker_payload = yield from self._stream_delegated_implement_worker(
                        conversation_id=conversation_id,
                        state=state,
                    )
                else:
                    worker_payload = yield from self._stream_delegated_analyze_worker(
                        conversation_id=conversation_id,
                        state=state,
                    )
                state.worker_requested = False
                state.worker_spawned = worker_payload.get("status") == "completed"
                state.phase = PHASE_SYNTHESIZE
                state.worker_summary = str(worker_payload.get("summary", ""))
                artifact_name = "output.md" if state.route == IMPLEMENT_ROUTE else "findings.md"
                correction_message = (
                    f"{self._run_instructions(state)}\n\n"
                    f"The delegated worker phase has finished. Read {artifact_name} and completion.json now and synthesize the final answer.\n\n"
                    "IMPORTANT: Your synthesis must be concise — at most ~200 lines of markdown for analysis, ~100 for implementation. "
                    f"Do NOT reproduce the full contents of {artifact_name}. Summarize the key findings, patterns, and conclusions. "
                    "The user can ask follow-up questions if they need more detail."
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
                if restart_with_phase_instruction and boundary_steering_code:
                    correction_message = self._phase_restart_instruction(state, boundary_steering_code)
                else:
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

    @staticmethod
    def _should_restart_neon_cycle(
        *,
        tool_name: str,
        steering_code: str | None,
        state: NeonRunState,
    ) -> bool:
        if steering_code in {
            STEERING_PRETASK_LIMIT_REACHED,
            STEERING_ORIENTATION_SIMPLE,
            STEERING_ORIENTATION_DELEGATED,
        }:
            return True
        if tool_name == "write_task_artifact":
            return True
        if tool_name in {"complete_todo_item", "skip_todo_item"} and state.route == ANALYZE_SIMPLE_ROUTE:
            return todo_is_complete(state.todo_items)
        return False

    def _ensure_task_created(self, state: NeonRunState, parsed: Any) -> str:
        if state.task_id is not None:
            return state.task_id
        intent = state.intent or TurnIntent.ANALYZE

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, state.conversation_id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared while creating a task.")
            task = Task(
                id=state.planned_task_id,
                conversation_id=conversation.id,
                parent_task_id=None,
                intent=intent.value,
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
                input_json={"route": parsed.route, "intent": intent.value},
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
                state_json={"route": parsed.route, "intent": intent.value},
            )

        state.task_id = state.planned_task_id
        self._workspace_manager.ensure_task_dir(state.task_id)
        return state.task_id

    def _hydrate_worker_state(self, worker_state: AnalyzeWorkerState | ImplementWorkerState) -> None:
        with self._session_factory() as session:
            task = session.get(Task, worker_state.task_id)
            if task is None:
                return
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

    def _stream_delegated_analyze_worker(
        self,
        *,
        conversation_id: str,
        state: NeonRunState,
    ) -> Iterator[dict[str, Any] | OrchestratedStreamEvent]:
        if state.task_id is None:
            raise RuntimeError("Delegated analyze worker requires an active task.")

        worker_state = AnalyzeWorkerState(task_id=state.task_id)
        self._hydrate_worker_state(worker_state)

        context = WorkspaceToolContext(
            root=self._workspace_root,
            session_factory=self._session_factory,
            task_id=state.task_id,
            agent_id="worker",
            approval_policy=self._approval_policy,
            path_access_manager=self._path_access_manager,
        )
        read_registry = WorkspaceToolRegistry(context)
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
            f"{self._user_identity_instruction(state)}\n\n"
            f"## TASK TITLE\n{state.task_title}\n\n"
            f"## TASK STATEMENT\n{state.task_statement}\n\n"
            "Anchor on the delegated task artifacts and produce a dense, factual findings.md."
        )
        result: AgentRunResult | None = None
        for _attempt in range(3):
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
            try:
                final_result: AgentRunResult | None = None
                boundary_restart_requested = False
                pending_text_chunks: list[str] = []
                cycle_history_messages: list[Message] = []
                stream = worker.stream(
                    "Read task.md, plan.md, and todo.md. "
                    "Follow todo.md sequentially. "
                    "Write a dense, factual findings.md, then stop.",
                    carryover_messages=carryover_messages,
                    extra_instructions=extra_instructions,
                )
                for event in stream:
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
                        state.turn_input_tokens = max(state.turn_input_tokens, event.metadata.get("input_tokens", 0))
                        state.turn_output_tokens += event.metadata.get("output_tokens", 0)
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
                        steering_code = self._steering_code_from_result(event.delta)
                        tool_ok = event.metadata.get("ok", True)
                        if not tool_ok:
                            self._record_runtime_action(
                                task_id=state.task_id,
                                agent_id="worker",
                                action_type=f"tool:{tool_name}",
                                target=None,
                                input_json=event.metadata.get("arguments", {}),
                                result_json={"raw": (event.delta or "")[:500]},
                                status="failed",
                                error_text=(event.delta or "")[:1000],
                            )
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
                            agent="worker",
                            phase="delegated_analyze",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            message=self._summarize_tool_result(tool_name, event.delta),
                        )
                        if steering_code == STEERING_WORKER_WRITE_FINDINGS_NOW:
                            boundary_restart_requested = True
                            break
                        if tool_name == "write_task_artifact" and worker_state.findings_written:
                            break
                        continue

                    if event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                        final_result = event.result

                if final_result is None and worker_state.findings_written:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if boundary_restart_requested:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                    carryover_messages = [*carryover_messages, *cycle_history_messages]
                    extra_instructions = (
                        f"{self._user_identity_instruction(state)}\n"
                        "Use what you already gathered. "
                        "Do not continue broad exploration. "
                        "Write findings.md now."
                    )
                    yield OrchestratedStreamEvent(
                        type="worker.status",
                        agent="worker",
                        phase="delegated_analyze",
                        conversation_id=conversation_id,
                        task_id=state.task_id,
                        message=steering_message(STEERING_WORKER_CONSOLIDATING),
                    )
                    continue
                if final_result is None and not worker_state.findings_written:
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

        if not worker_state.findings_written:
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
            worker_summary=self._final_text_payload(result) if result is not None else "",
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

    def _stream_delegated_implement_worker(
        self,
        *,
        conversation_id: str,
        state: NeonRunState,
    ) -> Iterator[dict[str, Any] | OrchestratedStreamEvent]:
        if state.task_id is None:
            raise RuntimeError("Delegated implement worker requires an active task.")

        worker_state = ImplementWorkerState(task_id=state.task_id)
        self._hydrate_worker_state(worker_state)

        context = WorkspaceToolContext(
            root=self._workspace_root,
            session_factory=self._session_factory,
            task_id=state.task_id,
            agent_id="worker",
            approval_policy=self._approval_policy,
            path_access_manager=self._path_access_manager,
            auto_approve_tools=frozenset({"write_file", "edit_file", "file_ops"}),
        )
        read_registry = WorkspaceToolRegistry(context)
        yield OrchestratedStreamEvent(
            type="phase.started",
            agent="worker",
            phase="delegated_implement",
            conversation_id=conversation_id,
            task_id=state.task_id,
            message="Delegated implement worker started.",
        )

        carryover_messages: list[Message] = []
        extra_instructions = (
            f"{self._user_identity_instruction(state)}\n\n"
            f"## TASK TITLE\n{state.task_title}\n\n"
            f"## TASK STATEMENT\n{state.task_statement}\n\n"
            "Anchor on the delegated task artifacts and produce a concise, factual output.md."
        )
        result: AgentRunResult | None = None
        for _attempt in range(3):
            registry = OrchestrationToolRegistry(
                orchestrator=self,
                actor=ACTOR_IMPLEMENT_WORKER,
                route=IMPLEMENT_ROUTE,
                state=worker_state,
                workspace_manager=self._workspace_manager,
                read_registry=read_registry,
            )
            worker = create_agent(
                name="ImplementWorker",
                model=state.model,
                instructions=IMPLEMENT_WORKER_PROMPT,
                provider=self._provider,
                tools=registry.schemas(),
                tool_registry=registry.handlers(),
                max_turns=24,
                reasoning_enabled=state.thinking_enabled,
                history_mode=HistoryMode.DIALOGUE_ONLY,
            )
            try:
                final_result: AgentRunResult | None = None
                boundary_restart_requested = False
                pending_text_chunks: list[str] = []
                cycle_history_messages: list[Message] = []
                stream = worker.stream(
                    "Read task.md, plan.md, and todo.md. "
                    "Follow todo.md sequentially. "
                    "Write a concise, factual output.md, then stop.",
                    carryover_messages=carryover_messages,
                    extra_instructions=extra_instructions,
                )
                for event in stream:
                    if event.type == "text.delta":
                        pending_text_chunks.append(event.delta or "")
                        continue

                    if event.type == "reasoning.delta":
                        yield OrchestratedStreamEvent(
                            type="reasoning.delta",
                            agent="worker",
                            phase="delegated_implement",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            thinking_text=event.delta,
                        )
                        continue

                    if event.type == "usage.update":
                        state.turn_input_tokens = max(state.turn_input_tokens, event.metadata.get("input_tokens", 0))
                        state.turn_output_tokens += event.metadata.get("output_tokens", 0)
                        continue

                    if event.type == "runtime.tool_result":
                        if pending_text_chunks:
                            for chunk in pending_text_chunks:
                                yield OrchestratedStreamEvent(
                                    type="text.delta",
                                    agent="worker",
                                    phase="delegated_implement",
                                    conversation_id=conversation_id,
                                    task_id=state.task_id,
                                    text=chunk,
                                )
                            pending_text_chunks = []
                        tool_name = event.tool_call.name if event.tool_call is not None else "unknown"
                        steering_code = self._steering_code_from_result(event.delta)
                        tool_ok = event.metadata.get("ok", True)
                        if not tool_ok:
                            self._record_runtime_action(
                                task_id=state.task_id,
                                agent_id="worker",
                                action_type=f"tool:{tool_name}",
                                target=None,
                                input_json=event.metadata.get("arguments", {}),
                                result_json={"raw": (event.delta or "")[:500]},
                                status="failed",
                                error_text=(event.delta or "")[:1000],
                            )
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
                            agent="worker",
                            phase="delegated_implement",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            message=self._summarize_tool_result(tool_name, event.delta),
                        )
                        if steering_code == STEERING_WORKER_WRITE_OUTPUT_NOW:
                            boundary_restart_requested = True
                            break
                        if tool_name == "write_task_artifact" and worker_state.output_written:
                            break
                        continue

                    if event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                        final_result = event.result

                if final_result is None and worker_state.output_written:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if boundary_restart_requested:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                    carryover_messages = [*carryover_messages, *cycle_history_messages]
                    extra_instructions = (
                        f"{self._user_identity_instruction(state)}\n"
                        "Use what you already changed. "
                        "Do not continue broad exploration. "
                        "Write output.md now."
                    )
                    yield OrchestratedStreamEvent(
                        type="worker.status",
                        agent="worker",
                        phase="delegated_implement",
                        conversation_id=conversation_id,
                        task_id=state.task_id,
                        message=steering_message(STEERING_IMPLEMENT_WORKER_CONSOLIDATING),
                    )
                    continue
                if final_result is None and not worker_state.output_written:
                    raise RuntimeError("Implement worker stream did not complete.")
                if worker_state.output_written and pending_text_chunks:
                    for chunk in pending_text_chunks:
                        yield OrchestratedStreamEvent(
                            type="text.delta",
                            agent="worker",
                            phase="delegated_implement",
                            conversation_id=conversation_id,
                            task_id=state.task_id,
                            text=chunk,
                        )
                result = final_result
            except ToolPermissionError as exc:
                payload = {"status": "failed", "summary": str(exc), "output_path": None}
                self._write_completion_artifact(state, status="failed", worker_status="failed", worker_summary=str(exc))
                yield OrchestratedStreamEvent(
                    type="phase.completed",
                    agent="worker",
                    phase="delegated_implement",
                    conversation_id=conversation_id,
                    task_id=state.task_id,
                    message=str(exc),
                    payload=payload,
                )
                return payload

            if worker_state.output_written:
                break
            if result is None:
                break
            carryover_messages = result.working_history
            extra_instructions = (
                f"{self._user_identity_instruction(state)}\n"
                "Use what you already changed. "
                "Do not continue broad exploration. "
                "Consolidate the execution into output.md now."
            )
            yield OrchestratedStreamEvent(
                type="worker.status",
                agent="worker",
                phase="delegated_implement",
                conversation_id=conversation_id,
                task_id=state.task_id,
                message=steering_message(STEERING_IMPLEMENT_WORKER_CONSOLIDATING),
            )

        if not worker_state.output_written:
            summary = "Implement worker finished without writing output.md."
            self._write_completion_artifact(state, status="failed", worker_status="failed", worker_summary=summary)
            payload = {"status": "failed", "summary": summary, "output_path": None}
            yield OrchestratedStreamEvent(
                type="phase.completed",
                agent="worker",
                phase="delegated_implement",
                conversation_id=conversation_id,
                task_id=state.task_id,
                message=steering_message(STEERING_IMPLEMENT_WORKER_CONSOLIDATING),
                payload=payload,
            )
            return payload

        self._persist_agent_message(
            state.task_id,
            "master",
            "worker",
            "assignment",
            "delegated_implement",
            "Complete the delegated implementation and write output.md.",
        )
        self._persist_agent_message(
            state.task_id,
            "worker",
            "master",
            "completion",
            "delegated_output",
            worker_state.output_path or "output.md",
        )
        self._write_completion_artifact(
            state,
            status="worker_completed",
            worker_status="completed",
            worker_summary=self._final_text_payload(result) if result is not None else "",
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
                    summary="Delegated implement worker completed.",
                    state_json={"output_path": worker_state.output_path},
                )

        payload = {
            "status": "completed",
            "summary": "Implement worker completed.",
            "output_path": worker_state.output_path,
        }
        yield OrchestratedStreamEvent(
            type="phase.completed",
            agent="worker",
            phase="delegated_implement",
            conversation_id=conversation_id,
            task_id=state.task_id,
            message="Implement worker completed.",
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

        if tool_name in {"spawn_analyze_worker", "spawn_implement_worker"} and payload is not None:
            return str(payload.get("summary", "Worker scheduled."))

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
        """Accumulate real API token usage from all provider runs into the turn-level accumulators.

        input_tokens uses max (peak context window) because each API call's
        input count already includes all prior messages.
        output_tokens uses sum (total generation across all calls).
        """
        for pr in run_result.provider_runs:
            u = pr.usage
            state.turn_input_tokens = max(state.turn_input_tokens, u.input_tokens or 0)
            state.turn_output_tokens += u.output_tokens or 0
