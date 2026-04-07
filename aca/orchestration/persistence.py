from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aca.core_types import TurnIntent
from aca.orchestration.state import AnalyzeWorkerState, ImplementWorkerState, NeonRunState
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


class PersistenceMixin:
    """Database write helpers extracted from NeonOrchestrator.

    The host class must provide ``_session_factory`` and ``_workspace_root``.
    """

    # ------------------------------------------------------------------
    # Conversation messages
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Agent messages
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Todo persistence
    # ------------------------------------------------------------------

    def _persist_todo_state(
        self,
        state: NeonRunState | AnalyzeWorkerState | ImplementWorkerState,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_message_sequence(self, session: Session, conversation_id: str) -> int:
        current = session.scalar(
            select(func.max(ConversationMessage.sequence_no)).where(
                ConversationMessage.conversation_id == conversation_id
            )
        )
        return int(current or 0) + 1

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"
