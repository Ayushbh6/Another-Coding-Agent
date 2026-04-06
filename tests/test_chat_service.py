from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from rich.console import Console
from sqlalchemy import select

from aca.cli.app import ChatCLIApp, CliSessionState, parse_command
from aca.config import Settings
from aca.llm.providers.base import LLMProvider
from aca.llm.types import Message, ProviderEvent, ProviderRequest, RunResult, UsageStats
from aca.runtime import ToolLoopRuntime
from aca.services.chat import CLEARED_MESSAGE_STATUS, ChatService, ChatStreamEvent, ConversationSummary, VISIBLE_MESSAGE_STATUS
from aca.storage import initialize_storage
from aca.storage.models import Conversation, ConversationMessage, User


class FakeStreamingProvider(LLMProvider):
    provider_name = "fake"

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        text = "Streaming reply from the assistant."
        yield ProviderEvent(type="reasoning.delta", delta="Let me think. ")
        yield ProviderEvent(type="reasoning.delta", delta="Still thinking.")
        yield ProviderEvent(type="text.delta", delta="Streaming ")
        yield ProviderEvent(type="text.delta", delta="reply ")
        yield ProviderEvent(type="text.delta", delta="from the assistant.")
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content=text),
                reasoning="",
                reasoning_details=[],
                text=text,
                structured_output=None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=8, output_tokens=6, total_tokens=14),
                response_id="fake-chat-response",
                latency_ms=3.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )


class ChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        sqlite_path = Path(self.temp_dir.name) / "aca.db"
        chroma_path = Path(self.temp_dir.name) / "chroma"
        settings = Settings(
            sqlite_url=f"sqlite:///{sqlite_path}",
            chroma_path=str(chroma_path),
            chroma_collection="aca-chat-test",
        )
        self.storage = initialize_storage(settings)
        self.chat_service = ChatService(
            session_factory=self.storage.session_factory,
            runtime=ToolLoopRuntime(FakeStreamingProvider(), {}),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initialize_user_creates_single_user(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        self.assertEqual(user.name, "Aparajit")

        same_user = self.chat_service.initialize_user("Ignored")
        self.assertEqual(same_user.id, user.id)

        with self.storage.session_factory() as session:
            users = session.scalars(select(User)).all()
            self.assertEqual(len(users), 1)

    def test_first_conversation_created_and_named_from_first_query(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        conversation = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=False,
        )

        events = list(
            self.chat_service.stream_chat_turn(
                conversation_id=conversation.id,
                user_input="How are tool calls handled here?",
                model="fake-model",
                thinking_enabled=False,
            )
        )

        self.assertEqual(
            [event.type for event in events],
            ["reasoning.delta", "reasoning.delta", "text.delta", "text.delta", "text.delta", "completed"],
        )
        completed = events[-1]
        self.assertIsNotNone(completed.summary)
        self.assertEqual(completed.summary.title, "How are tool")

        with self.storage.session_factory() as session:
            stored_conversation = session.get(Conversation, conversation.id)
            self.assertEqual(stored_conversation.title, "How are tool")
            messages = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation.id)
                .order_by(ConversationMessage.sequence_no)
            ).all()
            self.assertEqual(
                [(message.role, message.visibility_status) for message in messages],
                [("user", VISIBLE_MESSAGE_STATUS), ("assistant", VISIBLE_MESSAGE_STATUS)],
            )
            self.assertEqual([message.thinking_enabled for message in messages], [False, False])
            self.assertEqual(stored_conversation.active_model, "fake-model")

    def test_soft_clear_hides_existing_messages_from_replay_and_totals(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        conversation = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=False,
        )
        list(
            self.chat_service.stream_chat_turn(
                conversation_id=conversation.id,
                user_input="Hello there agent",
                model="fake-model",
                thinking_enabled=False,
            )
        )

        self.chat_service.soft_clear_conversation(conversation.id)
        summary = self.chat_service.get_conversation_summary(conversation.id)
        self.assertEqual(summary.visible_message_count, 0)
        self.assertEqual(summary.total_tokens, 0)

        list(
            self.chat_service.stream_chat_turn(
                conversation_id=conversation.id,
                user_input="Start a fresh thread",
                model="fake-model",
                thinking_enabled=False,
            )
        )

        with self.storage.session_factory() as session:
            messages = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation.id)
                .order_by(ConversationMessage.sequence_no)
            ).all()
            self.assertEqual(
                [message.visibility_status for message in messages[:2]],
                [CLEARED_MESSAGE_STATUS, CLEARED_MESSAGE_STATUS],
            )
            self.assertEqual(
                [message.visibility_status for message in messages[2:]],
                [VISIBLE_MESSAGE_STATUS, VISIBLE_MESSAGE_STATUS],
            )

    def test_list_conversations_returns_recent_first(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        first = self.chat_service.create_conversation(user.id)
        second = self.chat_service.create_conversation(user.id)

        conversations = self.chat_service.list_conversations(user.id)
        self.assertEqual([conversation.id for conversation in conversations], [second.id, first.id])

    def test_get_or_create_active_conversation_reuses_latest_empty_conversation(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        first = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=True,
        )

        reused = self.chat_service.get_or_create_active_conversation(user.id)

        self.assertEqual(reused.id, first.id)
        self.assertEqual(reused.active_model, "fake-model")
        self.assertTrue(reused.thinking_enabled)

    def test_get_or_create_active_conversation_creates_new_when_latest_has_messages(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        first = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=True,
        )
        list(
            self.chat_service.stream_chat_turn(
                conversation_id=first.id,
                user_input="Hello there",
                model="fake-model",
                thinking_enabled=True,
            )
        )

        created = self.chat_service.get_or_create_active_conversation(user.id)

        self.assertNotEqual(created.id, first.id)
        self.assertEqual(created.active_model, "fake-model")
        self.assertTrue(created.thinking_enabled)
        self.assertEqual(created.visible_message_count, 0)

    def test_delete_conversation_removes_conversation_and_messages(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        conversation = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=False,
        )
        list(
            self.chat_service.stream_chat_turn(
                conversation_id=conversation.id,
                user_input="Hello there",
                model="fake-model",
                thinking_enabled=False,
            )
        )

        self.chat_service.delete_conversation(conversation.id)

        with self.storage.session_factory() as session:
            self.assertIsNone(session.get(Conversation, conversation.id))
            remaining_messages = session.scalars(
                select(ConversationMessage).where(ConversationMessage.conversation_id == conversation.id)
            ).all()
            self.assertEqual(remaining_messages, [])

    def test_model_and_thinking_changes_are_persisted(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        conversation = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=False,
        )

        model_summary = self.chat_service.update_conversation_model(conversation.id, "moonshotai/kimi-k2.5:nitro")
        self.assertEqual(model_summary.active_model, "moonshotai/kimi-k2.5:nitro")

        thinking_summary = self.chat_service.update_conversation_thinking(conversation.id, True)
        self.assertTrue(thinking_summary.thinking_enabled)

        list(
            self.chat_service.stream_chat_turn(
                conversation_id=conversation.id,
                user_input="Keep chatting",
                model="moonshotai/kimi-k2.5:nitro",
                thinking_enabled=True,
            )
        )

        with self.storage.session_factory() as session:
            stored_conversation = session.get(Conversation, conversation.id)
            self.assertEqual(stored_conversation.active_model, "moonshotai/kimi-k2.5:nitro")
            self.assertTrue(stored_conversation.thinking_enabled)

            messages = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation.id)
                .order_by(ConversationMessage.sequence_no)
            ).all()
            self.assertEqual(
                [message.message_kind for message in messages],
                ["command_model_change", "command_thinking_toggle", "user", "assistant_final"],
            )
            self.assertEqual(
                [message.model_name for message in messages],
                [
                    "moonshotai/kimi-k2.5:nitro",
                    "moonshotai/kimi-k2.5:nitro",
                    "moonshotai/kimi-k2.5:nitro",
                    "moonshotai/kimi-k2.5:nitro",
                ],
            )
            self.assertEqual(
                [message.thinking_enabled for message in messages],
                [False, True, True, True],
            )

    def test_thinking_events_are_streamed_when_provider_emits_reasoning(self) -> None:
        user = self.chat_service.initialize_user("Aparajit")
        conversation = self.chat_service.create_conversation_with_settings(
            user_id=user.id,
            active_model="fake-model",
            thinking_enabled=True,
        )

        events = list(
            self.chat_service.stream_chat_turn(
                conversation_id=conversation.id,
                user_input="Think before replying",
                model="fake-model",
                thinking_enabled=True,
            )
        )

        reasoning_chunks = [event.thinking_text for event in events if event.type == "reasoning.delta"]
        self.assertEqual(reasoning_chunks, ["Let me think. ", "Still thinking."])


class CommandParsingTests(unittest.TestCase):
    def test_parse_command(self) -> None:
        self.assertEqual(parse_command("/thinking OFF"), ("thinking", "OFF"))
        self.assertEqual(parse_command("/model kimi_k2_5"), ("model", "kimi_k2_5"))
        self.assertEqual(parse_command("hello"), ("", "hello"))


class FakeChatServiceForCli:
    def stream_chat_turn(self, **_: object) -> Iterator[ChatStreamEvent]:
        yield ChatStreamEvent(type="reasoning.delta", thinking_text="Analyzing the request. ")
        yield ChatStreamEvent(type="reasoning.delta", thinking_text="Choosing next step.")
        yield ChatStreamEvent(type="text.delta", text="Final answer.")
        yield ChatStreamEvent(
            type="completed",
            final_answer="Final answer.",
            summary=ConversationSummary(
                id="conv-1",
                title="Test",
                visible_message_count=2,
                active_model="fake-model",
                thinking_enabled=True,
                total_input_tokens=10,
                total_output_tokens=12,
                total_tokens=22,
                updated_at=datetime.utcnow(),
            ),
        )


class ChatCliRenderingTests(unittest.TestCase):
    def test_send_chat_message_renders_thinking_and_answer_separately(self) -> None:
        app = object.__new__(ChatCLIApp)
        app.console = Console(record=True, force_terminal=False, color_system=None)
        app.chat_service = FakeChatServiceForCli()
        app.state = CliSessionState(
            active_user_id="user-1",
            active_conversation_id="conv-1",
            active_model="fake-model",
            thinking_enabled=True,
        )

        app._send_chat_message("hello")

        output = app.console.export_text()
        self.assertIn("thinking> Analyzing the request. Choosing next step.", output)
        self.assertIn("assistant> Final answer.", output)


if __name__ == "__main__":
    unittest.main()
