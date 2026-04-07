from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from aca.llm.types import HistoryMode
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


class ChatService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

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
            latest_conversation = session.scalars(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            ).first()

            if latest_conversation is None:
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

            if latest_conversation.message_count == 0:
                return self._build_conversation_summary(session, latest_conversation)

            conversation = Conversation(
                id=self._new_id("conv"),
                user_id=user_id,
                title="New chat",
                status="active",
                history_mode=HistoryMode.DIALOGUE_ONLY.value,
                active_model=latest_conversation.active_model,
                thinking_enabled=latest_conversation.thinking_enabled,
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

    def delete_conversation(self, conversation_id: str) -> None:
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            session.execute(delete(Conversation).where(Conversation.id == conversation_id))

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
