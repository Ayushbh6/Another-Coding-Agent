from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import select

from aca.config import Settings
from aca.core_types import MasterClassification, TurnIntent
from aca.llm.providers.base import LLMProvider
from aca.llm.types import Message, ProviderEvent, ProviderRequest, RunResult, ToolCall, UsageStats
from aca.runtime import ToolLoopRuntime
from aca.services import ConversationService, ConversationTurnRequest
from aca.storage import initialize_storage
from aca.storage.models import (
    Action,
    Agent,
    AgentMessage,
    Checkpoint,
    Conversation,
    ConversationMessage,
    MemoryEntry,
    Task,
)


class FakeMasterClassifier:
    def classify(self, *, model: str, messages: list[Message], user_input: str) -> MasterClassification:
        lowered = user_input.lower()
        if "hello" in lowered:
            return MasterClassification(
                intent=TurnIntent.CHAT,
                task_title="",
                task_description="",
                reasoning_summary="Simple conversational chat turn.",
            )
        if "go through the repo" in lowered or "tell me how" in lowered:
            return MasterClassification(
                intent=TurnIntent.ANALYZE,
                task_title="Analyze tool call flow",
                task_description="Inspect the repo and explain how tool calls are made.",
                reasoning_summary="This asks for repo inspection and explanation without file changes.",
            )
        return MasterClassification(
            intent=TurnIntent.IMPLEMENT,
            task_title="Add chatbot tool",
            task_description="Add a new tool to the chatbot and integrate it into the runtime.",
            reasoning_summary="This requires making code or tool-behavior changes.",
        )


class FakeProvider(LLMProvider):
    provider_name = "fake"

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        user_input = next(message.content for message in reversed(request.messages) if message.role == "user")
        lowered = (user_input or "").lower()
        tool_results_present = any(message.role == "tool" for message in request.messages)

        if "hello" in lowered:
            result = self._result(
                request=request,
                text="Hi, I'm doing well.",
                reasoning="Respond conversationally.",
                tool_calls=[],
                usage=UsageStats(input_tokens=4, output_tokens=6, total_tokens=10),
                response_id="chat-response",
            )
            yield ProviderEvent(type="response.completed", result=result)
            return

        if "go through the repo" in lowered or "tell me how" in lowered:
            result = self._result(
                request=request,
                text="Tool calls are routed through the runtime and executed against the tool registry.",
                reasoning="Analyze the code path and answer without changes.",
                tool_calls=[],
                usage=UsageStats(input_tokens=18, output_tokens=15, total_tokens=33),
                response_id="analyze-response",
            )
            yield ProviderEvent(type="response.completed", result=result)
            return

        if not tool_results_present:
            tool_call = ToolCall(id="call-1", name="echo_tool", arguments='{"text":"new tool"}')
            result = self._result(
                request=request,
                text="I will inspect the tool output first.",
                reasoning="Need to use a tool before answering.",
                tool_calls=[tool_call],
                usage=UsageStats(input_tokens=22, output_tokens=9, total_tokens=31, reasoning_tokens=2),
                response_id="implement-response-1",
            )
            yield ProviderEvent(type="response.completed", result=result)
            return

        result = self._result(
            request=request,
            text="The new tool has been integrated into the chatbot flow.",
            reasoning="Tool result processed; finalize the answer.",
            tool_calls=[],
            usage=UsageStats(input_tokens=25, output_tokens=12, total_tokens=37, reasoning_tokens=1),
            response_id="implement-response-2",
        )
        yield ProviderEvent(type="response.completed", result=result)

    def _result(
        self,
        *,
        request: ProviderRequest,
        text: str,
        reasoning: str,
        tool_calls: list[ToolCall],
        usage: UsageStats,
        response_id: str,
    ) -> RunResult:
        return RunResult(
            provider=self.provider_name,
            model=request.model,
            assistant_message=Message(role="assistant", content=text, tool_calls=tool_calls),
            reasoning=reasoning,
            text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
            response_id=response_id,
            latency_ms=5.0,
            provider_metadata={"provider": "fake"},
            raw_response={"id": response_id},
            raw_chunks=[],
        )


def echo_tool(text: str) -> str:
    return json.dumps({"echo": text})


class ConversationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        sqlite_path = Path(self.temp_dir.name) / "aca.db"
        chroma_path = Path(self.temp_dir.name) / "chroma"
        settings = Settings(
            sqlite_url=f"sqlite:///{sqlite_path}",
            chroma_path=str(chroma_path),
            chroma_collection="aca-memory-test",
        )
        self.storage = initialize_storage(settings)
        runtime = ToolLoopRuntime(FakeProvider(), {"echo_tool": echo_tool})
        self.service = ConversationService(
            session_factory=self.storage.session_factory,
            runtime=runtime,
            classifier=FakeMasterClassifier(),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bootstrap_is_idempotent_and_seeds_agents(self) -> None:
        initialize_storage(
            Settings(
                sqlite_url=str(self.storage.engine.url),
                chroma_path=str(Path(self.temp_dir.name) / "chroma"),
                chroma_collection="aca-memory-test",
            )
        )

        with self.storage.session_factory() as session:
            agents = session.scalars(select(Agent).order_by(Agent.id)).all()
            self.assertEqual([agent.id for agent in agents], ["challenger", "codebase_mapper", "master"])

        self.assertEqual(self.storage.chroma_collection.name, "aca-memory-test")

    def test_chat_turn_persists_messages_without_task(self) -> None:
        result = self.service.handle_turn(
            ConversationTurnRequest(
                model="fake-model",
                user_input="Hello, how are you?",
            )
        )

        self.assertEqual(result.intent, TurnIntent.CHAT)
        self.assertIsNone(result.task_id)

        with self.storage.session_factory() as session:
            conversation = session.get(Conversation, result.conversation_id)
            self.assertIsNotNone(conversation)
            self.assertEqual(conversation.message_count, 2)
            self.assertGreater(conversation.total_tokens, 0)

            messages = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == result.conversation_id)
                .order_by(ConversationMessage.sequence_no)
            ).all()
            self.assertEqual(
                [(message.role, message.message_kind) for message in messages],
                [("user", "user"), ("assistant", "assistant_final")],
            )
            self.assertEqual(session.scalar(select(func_count(Task.id))), 0)

    def test_analyze_turn_creates_task_and_memory(self) -> None:
        result = self.service.handle_turn(
            ConversationTurnRequest(
                model="fake-model",
                user_input="Go through the repo and tell me how the tool calls are being made in this app.",
            )
        )

        self.assertEqual(result.intent, TurnIntent.ANALYZE)
        self.assertIsNotNone(result.task_id)

        with self.storage.session_factory() as session:
            task = session.get(Task, result.task_id)
            self.assertIsNotNone(task)
            self.assertEqual(task.intent, "analyze")
            self.assertEqual(task.status, "completed")

            checkpoints = session.scalars(select(Checkpoint).where(Checkpoint.task_id == result.task_id)).all()
            self.assertEqual(len(checkpoints), 4)

            agent_messages = session.scalars(select(AgentMessage).where(AgentMessage.task_id == result.task_id)).all()
            self.assertEqual(len(agent_messages), 2)

            memory_entries = session.scalars(select(MemoryEntry).where(MemoryEntry.scope_id == result.task_id)).all()
            self.assertEqual(len(memory_entries), 1)

    def test_implement_turn_persists_full_working_transcript_and_actions(self) -> None:
        result = self.service.handle_turn(
            ConversationTurnRequest(
                model="fake-model",
                user_input="Please add a new tool to my existing chatbot.",
            )
        )

        self.assertEqual(result.intent, TurnIntent.IMPLEMENT)
        self.assertIsNotNone(result.task_id)

        with self.storage.session_factory() as session:
            messages = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == result.conversation_id)
                .order_by(ConversationMessage.sequence_no)
            ).all()
            self.assertEqual(
                [(message.role, message.message_kind) for message in messages],
                [
                    ("user", "user"),
                    ("assistant", "assistant_tool_request"),
                    ("tool", "tool_result"),
                    ("assistant", "assistant_final"),
                ],
            )

            tool_request = messages[1]
            self.assertEqual(tool_request.output_tokens, 9)
            self.assertEqual(tool_request.input_tokens, 22)

            actions = session.scalars(select(Action).where(Action.task_id == result.task_id)).all()
            action_types = sorted(action.action_type for action in actions)
            self.assertEqual(
                action_types,
                ["run_completed", "run_started", "task_created", "tool_execution"],
            )

            task = session.get(Task, result.task_id)
            self.assertEqual(task.status, "completed")

            conversation = session.get(Conversation, result.conversation_id)
            self.assertGreaterEqual(conversation.total_input_tokens, 22)
            self.assertGreaterEqual(conversation.total_output_tokens, 21)

            checkpoints = session.scalars(select(Checkpoint).where(Checkpoint.task_id == result.task_id)).all()
            self.assertEqual(len(checkpoints), 5)


def func_count(column):
    from sqlalchemy import func

    return func.count(column)


if __name__ == "__main__":
    unittest.main()
