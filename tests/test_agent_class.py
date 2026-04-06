from __future__ import annotations

import unittest
from collections.abc import Iterator

from aca.agent import Agent, AgentSpec, create_agent, normalize_structured_output
from aca.llm.providers.base import LLMProvider
from aca.llm.types import ImageContentPart, Message, ProviderEvent, ProviderRequest, RunResult, TextContentPart, UsageStats


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


if __name__ == "__main__":
    unittest.main()
