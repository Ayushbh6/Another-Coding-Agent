from __future__ import annotations

import unittest
from collections.abc import Iterator

from aca.agent import Agent, AgentSpec, create_agent, normalize_structured_output
from aca.llm.providers.base import LLMProvider
from aca.llm.types import (
    ImageContentPart,
    Message,
    ProviderEvent,
    ProviderRequest,
    RunResult,
    TextContentPart,
    ToolCall,
    UsageStats,
)


class RecordingProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        self.requests.append(request)
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content='{"title":"ready"}'),
                reasoning="deliberating",
                reasoning_details=[],
                text='{"title":"ready"}',
                structured_output={"title": "ready"} if request.response_format else None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=5, output_tokens=4, total_tokens=9),
                response_id="resp-1",
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )


class StructuredStreamingProvider(LLMProvider):
    provider_name = "fake-streaming"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        self.requests.append(request)
        json_text = '{"title":"ready"}'
        yield ProviderEvent(type="reasoning.delta", delta="deliberating ")
        yield ProviderEvent(type="text.delta", delta='{"title":')
        yield ProviderEvent(type="text.delta", delta='"ready"}')
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content=json_text, reasoning="deliberating "),
                reasoning="deliberating ",
                reasoning_details=[],
                text=json_text,
                structured_output={"title": "ready"} if request.response_format else None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=5, output_tokens=4, total_tokens=9),
                response_id="resp-structured-1",
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )


class ToolThenStructuredStreamingProvider(LLMProvider):
    provider_name = "fake-tool-streaming"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []
        self.calls = 0

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        self.requests.append(request)
        self.calls += 1

        if self.calls == 1:
            tool_call = ToolCall(id="call-1", name="echo_tool", arguments='{"text":"hello"}')
            yield ProviderEvent(
                type="response.completed",
                result=RunResult(
                    provider=self.provider_name,
                    model=request.model,
                    assistant_message=Message(
                        role="assistant",
                        content="Need tool output first.",
                        tool_calls=[tool_call],
                        reasoning="step-1",
                    ),
                    reasoning="step-1",
                    reasoning_details=[],
                    text="Need tool output first.",
                    structured_output=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    usage=UsageStats(input_tokens=5, output_tokens=4, total_tokens=9),
                    response_id="resp-tool-1",
                    latency_ms=1.0,
                    provider_metadata={},
                    raw_response={},
                    raw_chunks=[],
                ),
            )
            return

        if self.calls == 2:
            yield ProviderEvent(
                type="response.completed",
                result=RunResult(
                    provider=self.provider_name,
                    model=request.model,
                    assistant_message=Message(role="assistant", content=""),
                    reasoning="",
                    reasoning_details=[],
                    text="",
                    structured_output=None,
                    tool_calls=[],
                    finish_reason="stop",
                    usage=UsageStats(input_tokens=5, output_tokens=1, total_tokens=6),
                    response_id="resp-tool-2",
                    latency_ms=1.0,
                    provider_metadata={},
                    raw_response={},
                    raw_chunks=[],
                ),
            )
            return

        json_text = '{"title":"done"}'
        yield ProviderEvent(type="reasoning.delta", delta="step-2 ")
        yield ProviderEvent(type="text.delta", delta='{"title":')
        yield ProviderEvent(type="text.delta", delta='"done"}')
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content=json_text, reasoning="step-2 "),
                reasoning="step-2 ",
                reasoning_details=[],
                text=json_text,
                structured_output={"title": "done"} if request.response_format else None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=5, output_tokens=4, total_tokens=9),
                response_id="resp-tool-3",
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )


class AgentClassTests(unittest.TestCase):
    def test_agent_run_uses_spec_configuration(self) -> None:
        provider = RecordingProvider()
        agent = create_agent(
            name="Master",
            model="fake-model",
            instructions="You are the master agent.",
            provider=provider,
            tools=[{"type": "function", "function": {"name": "echo_tool"}}],
            tool_registry={},
            max_turns=30,
            structured_output={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
            reasoning_enabled=True,
        )

        result = agent.run("Plan the task.")

        self.assertEqual(result.status, "complete")
        request = provider.requests[-1]
        self.assertEqual(request.model, "fake-model")
        self.assertEqual(request.messages[0].role, "system")
        self.assertEqual(request.messages[0].content, "You are the master agent.")
        self.assertEqual(request.response_format["type"], "json_schema")
        self.assertEqual(request.response_format["json_schema"]["name"], "master")
        self.assertEqual(request.reasoning, None)
        self.assertTrue(request.include_reasoning)

    def test_agent_reasoning_boolean_can_disable_reasoning(self) -> None:
        provider = RecordingProvider()
        agent = Agent(
            spec=AgentSpec(
                name="Mapper",
                model="fake-model",
                instructions="Map the repo.",
                reasoning_enabled=False,
            ),
            provider=provider,
        )

        agent.run("Analyze ownership boundaries.")

        request = provider.requests[-1]
        self.assertFalse(request.include_reasoning)
        self.assertEqual(request.reasoning, {"exclude": True})

    def test_agent_supports_multimodal_user_input(self) -> None:
        provider = RecordingProvider()
        agent = create_agent(
            name="VisionAgent",
            model="fake-model",
            instructions="Inspect the image.",
            provider=provider,
        )

        agent.run(
            [
                TextContentPart(text="Describe the image."),
                ImageContentPart(image_url="https://example.com/cat.png", detail="high"),
            ]
        )

        request = provider.requests[-1]
        self.assertIsInstance(request.messages[1].content, list)
        content_parts = request.messages[1].to_provider_dict()["content"]
        self.assertEqual(content_parts[0]["type"], "text")
        self.assertEqual(content_parts[1]["type"], "image_url")

    def test_normalize_structured_output_wraps_bare_schema(self) -> None:
        response_format = normalize_structured_output(
            "Codebase Mapper",
            {
                "type": "object",
                "properties": {"owners": {"type": "array"}},
                "required": ["owners"],
                "additionalProperties": False,
            },
        )

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "codebase_mapper")
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_agent_stream_preserves_json_text_deltas_with_structured_output(self) -> None:
        provider = StructuredStreamingProvider()
        agent = create_agent(
            name="Master",
            model="fake-model",
            instructions="Return structured output.",
            provider=provider,
            structured_output={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
            reasoning_enabled=True,
        )

        events = list(agent.stream("Return a JSON object."))
        text_deltas = [event.delta for event in events if event.type == "text.delta"]
        reasoning_deltas = [event.delta for event in events if event.type == "reasoning.delta"]
        completed = [event for event in events if event.type == "run.completed"][-1]

        self.assertEqual(text_deltas, ['{"title":', '"ready"}'])
        self.assertEqual(reasoning_deltas, ["deliberating "])
        self.assertEqual(completed.result.final_answer, '{"title":"ready"}')
        self.assertEqual(completed.result.provider_runs[-1].structured_output, {"title": "ready"})
        self.assertEqual(provider.requests[-1].response_format["type"], "json_schema")

    def test_agent_streams_structured_output_after_tool_phase(self) -> None:
        provider = ToolThenStructuredStreamingProvider()
        agent = create_agent(
            name="Master",
            model="fake-model",
            instructions="Use the tool, then return structured output.",
            provider=provider,
            tools=[{"type": "function", "function": {"name": "echo_tool"}}],
            tool_registry={"echo_tool": lambda text: f'{{"echo":"{text}"}}'},
            structured_output={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
            reasoning_enabled=True,
        )

        events = list(agent.stream("Use the tool and finish with JSON."))
        text_deltas = [event.delta for event in events if event.type == "text.delta"]
        reasoning_deltas = [event.delta for event in events if event.type == "reasoning.delta"]
        completed = [event for event in events if event.type == "run.completed"][-1]

        self.assertEqual(text_deltas, ['{"title":', '"done"}'])
        self.assertEqual(reasoning_deltas, ["step-2 "])
        self.assertIsNone(provider.requests[0].response_format)
        self.assertIsNone(provider.requests[1].response_format)
        self.assertEqual(provider.requests[2].response_format["type"], "json_schema")
        user_messages = [message for message in provider.requests[2].messages if message.role == "user"]
        self.assertTrue(any("schema-compliant JSON object" in str(message.content) for message in user_messages))
        self.assertEqual(completed.result.final_answer, '{"title":"done"}')
        self.assertEqual(completed.result.provider_runs[-1].structured_output, {"title": "done"})


if __name__ == "__main__":
    unittest.main()
