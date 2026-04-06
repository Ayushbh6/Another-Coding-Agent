from __future__ import annotations

import json
import unittest
from collections.abc import Iterator

from aca.llm.providers.base import LLMProvider
from aca.llm.providers.openrouter import OpenRouterProvider
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
from aca.runtime import AgentRunRequest, ToolLoopRuntime


class RuntimeReasoningCarryoverProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.carryover_reasoning: str | None = None

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
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
                        content="Need to inspect the tool result.",
                        tool_calls=[tool_call],
                        reasoning="step-1 reasoning",
                        reasoning_details=[{"type": "reasoning.text", "text": "step-1 reasoning"}],
                    ),
                    reasoning="step-1 reasoning",
                    reasoning_details=[{"type": "reasoning.text", "text": "step-1 reasoning"}],
                    text="Need to inspect the tool result.",
                    structured_output=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
                    response_id="resp-1",
                    latency_ms=1.0,
                    provider_metadata={},
                    raw_response={},
                    raw_chunks=[],
                ),
            )
            return

        assistant_messages = [message for message in request.messages if message.role == "assistant"]
        self.carryover_reasoning = assistant_messages[-1].reasoning if assistant_messages else None

        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content="Done."),
                reasoning="",
                reasoning_details=[],
                text="Done.",
                structured_output=None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=8, output_tokens=3, total_tokens=11),
                response_id="resp-2",
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )


def echo_tool(text: str) -> str:
    return json.dumps({"echo": text})


class ProviderStackTests(unittest.TestCase):
    def test_message_serialization_supports_multimodal_content_and_reasoning(self) -> None:
        message = Message(
            role="user",
            content=[
                TextContentPart(text="Describe this image."),
                ImageContentPart(image_url="https://example.com/image.png", detail="high"),
            ],
            reasoning="previous reasoning",
            reasoning_details=[{"type": "reasoning.text", "text": "previous reasoning"}],
        )

        payload = message.to_provider_dict()

        self.assertEqual(payload["role"], "user")
        self.assertEqual(payload["content"][0], {"type": "text", "text": "Describe this image."})
        self.assertEqual(
            payload["content"][1],
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
        )
        self.assertEqual(payload["reasoning"], "previous reasoning")
        self.assertEqual(payload["reasoning_details"][0]["text"], "previous reasoning")

    def test_openrouter_build_request_kwargs_supports_reasoning_and_structure(self) -> None:
        provider = object.__new__(OpenRouterProvider)
        request = ProviderRequest(
            model="fake-model",
            messages=[Message(role="user", content="hello")],
            tools=[{"type": "function", "function": {"name": "echo_tool"}}],
            tool_choice="required",
            parallel_tool_calls=False,
            provider={"require_parameters": True},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "master_plan",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "additionalProperties": False,
                    },
                },
            },
            reasoning={"effort": "high"},
        )

        kwargs = provider._build_request_kwargs(request)

        self.assertEqual(kwargs["tool_choice"], "required")
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(kwargs["extra_body"]["provider"], {"require_parameters": True})
        self.assertEqual(kwargs["extra_body"]["reasoning"], {"effort": "high"})

    def test_openrouter_parses_structured_output_when_json_is_returned(self) -> None:
        provider = object.__new__(OpenRouterProvider)

        structured = provider._parse_structured_output(
            '{"title":"Spec Freeze","owner":"master"}',
            {
                "type": "json_schema",
                "json_schema": {"name": "plan", "strict": True, "schema": {"type": "object"}},
            },
        )

        self.assertEqual(structured, {"title": "Spec Freeze", "owner": "master"})

    def test_runtime_preserves_reasoning_between_tool_iterations(self) -> None:
        provider = RuntimeReasoningCarryoverProvider()
        runtime = ToolLoopRuntime(provider=provider, tool_registry={"echo_tool": echo_tool})

        result = runtime.run(
            AgentRunRequest(
                model="fake-model",
                user_input="Use the tool and answer.",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "echo_tool",
                            "parameters": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                    }
                ],
            )
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(provider.carryover_reasoning, "step-1 reasoning")


if __name__ == "__main__":
    unittest.main()
