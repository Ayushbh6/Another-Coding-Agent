from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aca.llm.types import Message, ToolCall
from aca.services.chat import ConversationSummary
from aca.storage.models import Conversation, ConversationMessage


class HistoryMixin:
    """Conversation history loading and summary helpers extracted from NeonOrchestrator.

    The host class must provide ``_session_factory``.
    """

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

    def _delegated_progress_messages(self, state: Any) -> list[Message]:
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
        spawn_tool_name = "spawn_implement_worker" if getattr(state, "route", None) == "implement" else "spawn_analyze_worker"
        spawn_summary = "Implement worker scheduled." if spawn_tool_name == "spawn_implement_worker" else "Analyze worker scheduled."
        messages.append(
            Message(
                role="tool",
                content=json.dumps({"status": "scheduled", "summary": spawn_summary}),
                metadata={"tool_name": spawn_tool_name},
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
        if "output.md" in state.artifact_paths:
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps({"artifact_name": "output.md", "path": state.artifact_paths["output.md"]}),
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
