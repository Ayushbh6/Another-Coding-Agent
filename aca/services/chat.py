from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, sessionmaker

from aca.llm.types import HistoryMode, Message
from aca.prompts import HELPFUL_ASSISTANT_PROMPT
from aca.runtime import AgentRunRequest, AgentRunResult, ToolLoopRuntime
from aca.storage.models import Conversation, ConversationMessage, User
from aca.token_utils import estimate_json_tokens, estimate_text_tokens


VISIBLE_MESSAGE_STATUS = "visible"
CLEARED_MESSAGE_STATUS = "cleared"


@dataclass(slots=True)
class ConversationSummary:
    id: str
    title: str
    visible_message_count: int
    active_model: str | None
    thinking_enabled: bool
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    updated_at: datetime


@dataclass(slots=True)
class ChatStreamEvent:
    type: str
    text: str | None = None
    final_answer: str | None = None
    summary: ConversationSummary | None = None


class ChatService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        runtime: ToolLoopRuntime,
    ) -> None:
        self._session_factory = session_factory
        self._runtime = runtime

    def get_single_user(self) -> User | None:
        with self._session_factory() as session:
            return session.scalars(select(User).order_by(User.created_at.asc())).first()

    def initialize_user(self, name: str) -> User:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("User name cannot be empty.")

        with self._session_factory.begin() as session:
            existing = session.scalars(select(User).order_by(User.created_at.asc())).first()
            if existing is not None:
                return existing

            user = User(
                id=self._new_id("user"),
                name=cleaned_name,
                metadata_json={},
            )
            session.add(user)
            session.flush()

            orphaned_conversations = session.scalars(
                select(Conversation).where(Conversation.user_id.is_(None))
            ).all()
            for conversation in orphaned_conversations:
                conversation.user_id = user.id

            return user

    def create_conversation(self, user_id: str) -> ConversationSummary:
        return self.create_conversation_with_settings(
            user_id=user_id,
            active_model=None,
            thinking_enabled=False,
        )

    def create_conversation_with_settings(
        self,
        *,
        user_id: str,
        active_model: str | None,
        thinking_enabled: bool,
    ) -> ConversationSummary:
        with self._session_factory.begin() as session:
            conversation = Conversation(
                id=self._new_id("conv"),
                user_id=user_id,
                title="New chat",
                status="active",
                history_mode=HistoryMode.DIALOGUE_ONLY.value,
                active_model=active_model,
                thinking_enabled=thinking_enabled,
                current_task_id=None,
                message_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                metadata_json={},
            )
            session.add(conversation)
            session.flush()
            return self._build_conversation_summary(session, conversation)

    def get_or_create_active_conversation(self, user_id: str) -> ConversationSummary:
        with self._session_factory.begin() as session:
            conversation = session.scalars(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            ).first()
            if conversation is None:
                conversation = Conversation(
                    id=self._new_id("conv"),
                    user_id=user_id,
                    title="New chat",
                    status="active",
                    history_mode=HistoryMode.DIALOGUE_ONLY.value,
                    active_model=None,
                    thinking_enabled=False,
                    current_task_id=None,
                    message_count=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_tokens=0,
                    metadata_json={},
                )
                session.add(conversation)
                session.flush()
            return self._build_conversation_summary(session, conversation)

    def update_conversation_model(self, conversation_id: str, model: str) -> ConversationSummary:
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            conversation.active_model = model
            conversation.updated_at = datetime.utcnow()
            self._insert_message(
                session,
                conversation=conversation,
                role="system",
                message_kind="command_model_change",
                content_text=f"Model switched to {model}",
                model_name=model,
                thinking_enabled=conversation.thinking_enabled,
                metadata_json={"command": "/model", "active_model": model},
            )
            return self._build_conversation_summary(session, conversation)

    def update_conversation_thinking(self, conversation_id: str, thinking_enabled: bool) -> ConversationSummary:
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            conversation.thinking_enabled = thinking_enabled
            conversation.updated_at = datetime.utcnow()
            self._insert_message(
                session,
                conversation=conversation,
                role="system",
                message_kind="command_thinking_toggle",
                content_text=f"Thinking turned {'ON' if thinking_enabled else 'OFF'}",
                model_name=conversation.active_model,
                thinking_enabled=thinking_enabled,
                metadata_json={"command": "/thinking", "thinking_enabled": thinking_enabled},
            )
            return self._build_conversation_summary(session, conversation)

    def list_conversations(self, user_id: str) -> list[ConversationSummary]:
        with self._session_factory() as session:
            conversations = session.scalars(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            ).all()
            return [self._build_conversation_summary(session, conversation) for conversation in conversations]

    def get_conversation_summary(self, conversation_id: str) -> ConversationSummary:
        with self._session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            return self._build_conversation_summary(session, conversation)

    def soft_clear_conversation(self, conversation_id: str) -> None:
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")

            visible_messages = session.scalars(
                select(ConversationMessage).where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.visibility_status == VISIBLE_MESSAGE_STATUS,
                )
            ).all()
            for message in visible_messages:
                message.visibility_status = CLEARED_MESSAGE_STATUS
            conversation.updated_at = datetime.utcnow()

    def stream_chat_turn(
        self,
        *,
        conversation_id: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
    ) -> Iterator[ChatStreamEvent]:
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")

            visible_history = self._load_visible_history(session, conversation_id)
            has_visible_user_messages = self._visible_user_message_count(session, conversation_id) > 0

            if not has_visible_user_messages:
                conversation.title = _conversation_title_from_query(user_input)
            conversation.active_model = model
            conversation.thinking_enabled = thinking_enabled

            self._insert_message(
                session,
                conversation=conversation,
                role="user",
                message_kind="user",
                content_text=user_input,
                model_name=model,
                thinking_enabled=thinking_enabled,
            )
            carryover_messages = list(visible_history)

        final_run_result: AgentRunResult | None = None
        for event in self._runtime.stream(
            AgentRunRequest(
                model=model,
                user_input=user_input,
                system_prompt=HELPFUL_ASSISTANT_PROMPT,
                carryover_messages=carryover_messages,
                tools=[],
                history_mode=HistoryMode.DIALOGUE_ONLY,
                include_reasoning=thinking_enabled,
            )
        ):
            if event.type == "text.delta":
                yield ChatStreamEvent(type="text.delta", text=event.delta)
                continue

            if event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                final_run_result = event.result

        if final_run_result is None:
            raise RuntimeError("Chat runtime ended without a final result.")

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared before assistant persistence.")

            final_answer = final_run_result.final_answer or ""
            provider_run = final_run_result.provider_runs[-1] if final_run_result.provider_runs else None
            self._insert_message(
                session,
                conversation=conversation,
                role="assistant",
                message_kind="assistant_final",
                content_text=final_answer,
                content_json={"reasoning": provider_run.reasoning if provider_run is not None else ""},
                provider_name=provider_run.provider if provider_run is not None else None,
                model_name=model,
                thinking_enabled=thinking_enabled,
                response_id=provider_run.response_id if provider_run is not None else None,
                input_tokens=provider_run.usage.input_tokens if provider_run is not None else None,
                output_tokens=provider_run.usage.output_tokens if provider_run is not None else None,
                total_tokens=provider_run.usage.total_tokens if provider_run is not None else None,
                reasoning_tokens=provider_run.usage.reasoning_tokens if provider_run is not None else None,
                latency_ms=provider_run.latency_ms if provider_run is not None else None,
                metadata_json={
                    "provider_metadata": provider_run.provider_metadata if provider_run is not None else {},
                    "raw_usage": provider_run.usage.raw if provider_run is not None else {},
                },
            )
            summary = self._build_conversation_summary(session, conversation)

        yield ChatStreamEvent(
            type="completed",
            final_answer=final_run_result.final_answer,
            summary=summary,
        )

    def _load_visible_history(self, session: Session, conversation_id: str) -> list[Message]:
        rows = session.scalars(
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.visibility_status == VISIBLE_MESSAGE_STATUS,
            )
            .order_by(ConversationMessage.sequence_no)
        ).all()
        return [
            Message(role=row.role, content=row.content_text)
            for row in rows
            if row.role in {"user", "assistant"} and row.message_kind in {"user", "assistant_final"}
        ]

    def _visible_user_message_count(self, session: Session, conversation_id: str) -> int:
        count = session.scalar(
            select(func.count(ConversationMessage.id)).where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.visibility_status == VISIBLE_MESSAGE_STATUS,
                ConversationMessage.role == "user",
            )
        )
        return int(count or 0)

    def _build_conversation_summary(self, session: Session, conversation: Conversation) -> ConversationSummary:
        aggregates = session.execute(
            select(
                func.count(ConversationMessage.id),
                func.coalesce(func.sum(ConversationMessage.input_tokens), 0),
                func.coalesce(func.sum(ConversationMessage.output_tokens), 0),
                func.coalesce(func.sum(ConversationMessage.total_tokens), 0),
            ).where(
                ConversationMessage.conversation_id == conversation.id,
                ConversationMessage.visibility_status == VISIBLE_MESSAGE_STATUS,
            )
        ).one()

        return ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            visible_message_count=int(aggregates[0] or 0),
            active_model=conversation.active_model,
            thinking_enabled=conversation.thinking_enabled,
            total_input_tokens=int(aggregates[1] or 0),
            total_output_tokens=int(aggregates[2] or 0),
            total_tokens=int(aggregates[3] or 0),
            updated_at=conversation.updated_at,
        )

    def _insert_message(
        self,
        session: Session,
        *,
        conversation: Conversation,
        role: str,
        message_kind: str,
        content_text: str | None = None,
        content_json: dict[str, object] | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        thinking_enabled: bool = False,
        response_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        latency_ms: float | None = None,
        metadata_json: dict[str, object] | None = None,
    ) -> None:
        sequence_no = self._next_message_sequence(session, conversation.id)
        token_estimate = estimate_text_tokens(content_text, model=model_name)
        if content_json:
            token_estimate += estimate_json_tokens(content_json, model=model_name)

        resolved_input = input_tokens if input_tokens is not None else (token_estimate if role == "user" else 0)
        resolved_output = output_tokens if output_tokens is not None else (token_estimate if role == "assistant" else 0)
        resolved_total = total_tokens if total_tokens is not None else resolved_input + resolved_output

        session.add(
            ConversationMessage(
                id=self._new_id("msg"),
                conversation_id=conversation.id,
                sequence_no=sequence_no,
                role=role,
                message_kind=message_kind,
                intent=None,
                content_text=content_text,
                content_json=content_json,
                tool_call_id=None,
                tool_name=None,
                provider_name=provider_name,
                model_name=model_name,
                thinking_enabled=thinking_enabled,
                response_id=response_id,
                text_token_estimate=token_estimate,
                input_tokens=resolved_input,
                output_tokens=resolved_output,
                total_tokens=resolved_total,
                reasoning_tokens=reasoning_tokens or 0,
                latency_ms=latency_ms,
                visibility_status=VISIBLE_MESSAGE_STATUS,
                metadata_json=metadata_json or {},
            )
        )
        now = datetime.utcnow()
        conversation.message_count += 1
        conversation.total_input_tokens += resolved_input
        conversation.total_output_tokens += resolved_output
        conversation.total_tokens += resolved_total
        conversation.last_message_at = now
        conversation.updated_at = now

    def _next_message_sequence(self, session: Session, conversation_id: str) -> int:
        current = session.scalar(
            select(func.max(ConversationMessage.sequence_no)).where(
                ConversationMessage.conversation_id == conversation_id
            )
        )
        return int(current or 0) + 1

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"


def _conversation_title_from_query(user_input: str) -> str:
    words = user_input.strip().split()
    if not words:
        return "New chat"
    return " ".join(words[:3])
