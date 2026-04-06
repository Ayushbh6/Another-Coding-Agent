from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from aca.core_types import MasterClassification, TurnIntent
from aca.llm.history import project_messages_for_carryover
from aca.llm.types import HistoryMode, Message, ToolCall
from aca.master import MasterClassifier
from aca.runtime import AgentRunRequest, AgentRunResult, ToolExecutionRecord, ToolLoopRuntime
from aca.storage.models import (
    Action,
    AgentMessage,
    Checkpoint,
    Conversation,
    ConversationMessage,
    MemoryEntry,
    Task,
)
from aca.token_utils import estimate_json_tokens, estimate_text_tokens


@dataclass(slots=True)
class ConversationTurnRequest:
    model: str
    user_input: str
    conversation_id: str | None = None
    conversation_title: str | None = None
    system_prompt: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    history_mode: HistoryMode = HistoryMode.DIALOGUE_ONLY
    max_iterations: int = 10
    include_reasoning: bool = True
    provider_extra_body: dict[str, Any] = field(default_factory=dict)
    classification_model: str | None = None


@dataclass(slots=True)
class ConversationTurnResult:
    conversation_id: str
    intent: TurnIntent
    task_id: str | None
    classification: MasterClassification
    run_result: AgentRunResult


class ConversationService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        runtime: ToolLoopRuntime,
        classifier: MasterClassifier,
    ) -> None:
        self._session_factory = session_factory
        self._runtime = runtime
        self._classifier = classifier

    def handle_turn(self, request: ConversationTurnRequest) -> ConversationTurnResult:
        with self._session_factory.begin() as session:
            conversation = self._get_or_create_conversation(session, request)
            conversation_history_mode = HistoryMode(conversation.history_mode)
            full_history = self._load_full_history(session, conversation.id)
            carryover_messages = project_messages_for_carryover(full_history, conversation_history_mode)

            classification = self._classifier.classify(
                model=request.classification_model or request.model,
                messages=carryover_messages,
                user_input=request.user_input,
            )

            user_message_row = self._insert_conversation_message(
                session,
                conversation=conversation,
                role="user",
                message_kind="user",
                intent=classification.intent,
                content_text=request.user_input,
                model_name=request.model,
            )
            self._log_action(
                session,
                task_id=None,
                agent_id="master",
                action_type="intent_classification",
                target=conversation.id,
                input_json={"user_input": request.user_input},
                result_json={
                    "intent": classification.intent.value,
                    "task_title": classification.task_title,
                    "task_description": classification.task_description,
                    "reasoning_summary": classification.reasoning_summary,
                },
                status="success",
            )

            task = None
            if classification.intent in {TurnIntent.ANALYZE, TurnIntent.IMPLEMENT}:
                task = self._create_task(session, conversation, classification)
                self._log_agent_message(
                    session,
                    task_id=task.id,
                    from_agent_id="master",
                    to_agent_id="master",
                    channel="decision",
                    message_type="classification",
                    content=classification.reasoning_summary,
                )
                self._log_agent_message(
                    session,
                    task_id=task.id,
                    from_agent_id="master",
                    to_agent_id=task.assigned_to_agent_id or "master",
                    channel="assignment",
                    message_type="assignment_summary",
                    content=f"Assigned task '{task.title}' with intent={task.intent}.",
                )
                self._log_action(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    action_type="task_created",
                    target=task.id,
                    input_json={
                        "intent": classification.intent.value,
                        "task_title": classification.task_title,
                        "task_description": classification.task_description,
                    },
                    result_json={"status": task.status, "phase": task.phase},
                    status="success",
                )
                self._write_checkpoint(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    phase="NEW_TASK",
                    checkpoint_kind="task_created",
                    summary=f"Task created for {classification.intent.value} turn.",
                    state_json={"conversation_id": conversation.id, "user_message_id": user_message_row.id},
                )

        with self._session_factory.begin() as session:
            if task is not None:
                self._write_checkpoint(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    phase="EXECUTE",
                    checkpoint_kind="run_started",
                    summary="Runtime execution started.",
                    state_json={"conversation_id": conversation.id},
                )
                self._log_action(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    action_type="run_started",
                    target=conversation.id,
                    input_json={"model": request.model},
                    result_json={},
                    status="success",
                )

        run_result = self._runtime.run(
            AgentRunRequest(
                model=request.model,
                user_input=request.user_input,
                system_prompt=request.system_prompt,
                carryover_messages=carryover_messages,
                tools=request.tools,
                history_mode=conversation_history_mode,
                max_iterations=request.max_iterations,
                include_reasoning=request.include_reasoning,
                provider_extra_body=request.provider_extra_body,
            )
        )

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation.id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared during execution.")

            persisted_task = session.get(Task, task.id) if task is not None else None

            self._persist_run_messages(
                session,
                conversation=conversation,
                intent=classification.intent,
                model_name=request.model,
                run_result=run_result,
            )

            if persisted_task is not None:
                self._persist_task_artifacts(
                    session,
                    task=persisted_task,
                    conversation=conversation,
                    classification=classification,
                    run_result=run_result,
                )

            conversation.current_task_id = None

        return ConversationTurnResult(
            conversation_id=conversation.id,
            intent=classification.intent,
            task_id=task.id if task is not None else None,
            classification=classification,
            run_result=run_result,
        )

    def _get_or_create_conversation(self, session: Session, request: ConversationTurnRequest) -> Conversation:
        if request.conversation_id:
            conversation = session.get(Conversation, request.conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {request.conversation_id}")
            return conversation

        conversation = Conversation(
            id=self._new_id("conv"),
            title=request.conversation_title or _default_conversation_title(request.user_input),
            status="active",
            history_mode=request.history_mode.value,
            current_task_id=None,
            message_count=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            metadata_json={},
        )
        session.add(conversation)
        session.flush()
        return conversation

    def _create_task(
        self,
        session: Session,
        conversation: Conversation,
        classification: MasterClassification,
    ) -> Task:
        assigned_to_agent_id = "codebase_mapper" if classification.intent is TurnIntent.ANALYZE else "master"
        task = Task(
            id=self._new_id("task"),
            conversation_id=conversation.id,
            parent_task_id=None,
            intent=classification.intent.value,
            title=classification.task_title or _default_task_title(classification.intent),
            description=classification.task_description or classification.reasoning_summary,
            status="in_progress",
            phase="EXECUTE",
            priority="normal",
            assigned_to_agent_id=assigned_to_agent_id,
            created_by="master",
            spec_json={"reasoning_summary": classification.reasoning_summary},
            scope_json={},
            metadata_json={},
        )
        session.add(task)
        session.flush()
        conversation.current_task_id = task.id
        return task

    def _persist_run_messages(
        self,
        session: Session,
        *,
        conversation: Conversation,
        intent: TurnIntent,
        model_name: str,
        run_result: AgentRunResult,
    ) -> None:
        tool_executions_by_iteration: dict[int, list[ToolExecutionRecord]] = {}
        for execution in run_result.tool_executions:
            tool_executions_by_iteration.setdefault(execution.iteration, []).append(execution)

        for iteration, provider_run in enumerate(run_result.provider_runs, start=1):
            if provider_run.tool_calls:
                self._insert_conversation_message(
                    session,
                    conversation=conversation,
                    role="assistant",
                    message_kind="assistant_tool_request",
                    intent=intent,
                    content_text=provider_run.text or None,
                    content_json={
                        "tool_calls": [self._tool_call_payload(tool_call) for tool_call in provider_run.tool_calls],
                        "reasoning": provider_run.reasoning,
                    },
                    provider_name=provider_run.provider,
                    model_name=model_name,
                    response_id=provider_run.response_id,
                    input_tokens=provider_run.usage.input_tokens or 0,
                    output_tokens=provider_run.usage.output_tokens or 0,
                    total_tokens=provider_run.usage.total_tokens or (
                        (provider_run.usage.input_tokens or 0) + (provider_run.usage.output_tokens or 0)
                    ),
                    reasoning_tokens=provider_run.usage.reasoning_tokens or 0,
                    latency_ms=provider_run.latency_ms,
                    metadata_json={
                        "finish_reason": provider_run.finish_reason,
                        "provider_metadata": provider_run.provider_metadata,
                        "raw_usage": provider_run.usage.raw,
                    },
                )
            else:
                self._insert_conversation_message(
                    session,
                    conversation=conversation,
                    role="assistant",
                    message_kind="assistant_final",
                    intent=intent,
                    content_text=provider_run.text or None,
                    content_json={"reasoning": provider_run.reasoning},
                    provider_name=provider_run.provider,
                    model_name=model_name,
                    response_id=provider_run.response_id,
                    input_tokens=provider_run.usage.input_tokens or 0,
                    output_tokens=provider_run.usage.output_tokens or estimate_text_tokens(provider_run.text, model=model_name),
                    total_tokens=provider_run.usage.total_tokens or (
                        (provider_run.usage.input_tokens or 0)
                        + (provider_run.usage.output_tokens or estimate_text_tokens(provider_run.text, model=model_name))
                    ),
                    reasoning_tokens=provider_run.usage.reasoning_tokens or 0,
                    latency_ms=provider_run.latency_ms,
                    metadata_json={
                        "finish_reason": provider_run.finish_reason,
                        "provider_metadata": provider_run.provider_metadata,
                        "raw_usage": provider_run.usage.raw,
                    },
                )

            for execution in tool_executions_by_iteration.get(iteration, []):
                self._insert_conversation_message(
                    session,
                    conversation=conversation,
                    role="tool",
                    message_kind="tool_result",
                    intent=intent,
                    content_text=execution.result,
                    tool_call_id=execution.tool_call_id,
                    tool_name=execution.tool_name,
                    model_name=model_name,
                    input_tokens=estimate_text_tokens(execution.result, model=model_name),
                    output_tokens=0,
                    total_tokens=estimate_text_tokens(execution.result, model=model_name),
                    metadata_json={"arguments": execution.arguments, "ok": execution.ok},
                )

    def _persist_task_artifacts(
        self,
        session: Session,
        *,
        task: Task,
        conversation: Conversation,
        classification: MasterClassification,
        run_result: AgentRunResult,
    ) -> None:
        for iteration, provider_run in enumerate(run_result.provider_runs, start=1):
            self._write_checkpoint(
                session,
                task_id=task.id,
                agent_id="master",
                phase="EXECUTE",
                checkpoint_kind="iteration_boundary",
                summary=f"Completed iteration {iteration}.",
                state_json={
                    "iteration": iteration,
                    "response_id": provider_run.response_id,
                    "finish_reason": provider_run.finish_reason,
                },
                resume_cursor=provider_run.response_id,
            )

        for execution in run_result.tool_executions:
            result_payload = _safe_json_loads(execution.result)
            self._log_action(
                session,
                task_id=task.id,
                agent_id="master",
                action_type="tool_execution",
                target=execution.tool_name,
                input_json={
                    "iteration": execution.iteration,
                    "tool_call_id": execution.tool_call_id,
                    "arguments": execution.arguments,
                },
                result_json=result_payload if isinstance(result_payload, dict) else {"raw_result": execution.result},
                status="success" if execution.ok else "failed",
                error_text=None if execution.ok else execution.result,
            )

        self._log_action(
            session,
            task_id=task.id,
            agent_id="master",
            action_type="run_completed",
            target=conversation.id,
            input_json={"intent": classification.intent.value},
            result_json={
                "status": run_result.status,
                "iterations": run_result.iterations,
                "final_answer": run_result.final_answer,
            },
            status="success" if run_result.status == "complete" else "failed",
        )
        self._write_checkpoint(
            session,
            task_id=task.id,
            agent_id="master",
            phase="COMPLETE",
            checkpoint_kind="run_completed",
            summary="Runtime execution completed.",
            state_json={
                "status": run_result.status,
                "iterations": run_result.iterations,
                "final_answer": run_result.final_answer,
            },
        )

        task.status = "completed" if run_result.status == "complete" else "failed"
        task.phase = "COMPLETE" if run_result.status == "complete" else "EXECUTE"
        task.completed_at = datetime.utcnow() if run_result.status == "complete" else None
        task.spec_json = {
            **task.spec_json,
            "reasoning_summary": classification.reasoning_summary,
            "final_answer": run_result.final_answer,
        }

        if classification.intent in {TurnIntent.ANALYZE, TurnIntent.IMPLEMENT}:
            summary_title = task.title
            summary_content = classification.reasoning_summary
            if run_result.final_answer:
                summary_content = f"{classification.reasoning_summary}\n\nFinal outcome: {run_result.final_answer}"
            session.add(
                MemoryEntry(
                    id=self._new_id("mem"),
                    memory_type="episodic_summary",
                    scope_type="task",
                    scope_id=task.id,
                    title=summary_title,
                    content=summary_content,
                    source_kind="task_completion",
                    source_id=task.id,
                    importance=2,
                    created_by_agent_id="master",
                    metadata_json={"intent": classification.intent.value},
                )
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
                messages.append(Message(role="assistant", content=row.content_text, tool_calls=tool_calls))
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

            messages.append(Message(role=row.role, content=row.content_text))

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

    def _tool_call_payload(self, tool_call: ToolCall) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        }


def _default_conversation_title(user_input: str) -> str:
    return user_input.strip()[:80] or "New conversation"


def _default_task_title(intent: TurnIntent) -> str:
    if intent is TurnIntent.ANALYZE:
        return "Analyze repo request"
    if intent is TurnIntent.IMPLEMENT:
        return "Implement repo change"
    return "Chat turn"


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
