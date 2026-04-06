from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]
ImageDetail = Literal["auto", "low", "high"]


class HistoryMode(str, Enum):
    FULL = "full"
    DIALOGUE_ONLY = "dialogue_only"
    COMPACT_TOOLS = "compact_tools"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class TextContentPart:
    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True)
class ImageContentPart:
    image_url: str
    detail: ImageDetail | None = None
    type: Literal["image_url"] = "image_url"


ContentPart = TextContentPart | ImageContentPart


@dataclass(slots=True)
class Message:
    role: MessageRole
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_provider_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            payload["content"] = (
                [self._serialize_content_part(part) for part in self.content]
                if isinstance(self.content, list)
                else self.content
            )

        if self.name:
            payload["name"] = self.name

        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]

        if self.reasoning:
            payload["reasoning"] = self.reasoning

        if self.reasoning_details:
            payload["reasoning_details"] = self.reasoning_details

        return payload

    def _serialize_content_part(self, part: ContentPart) -> dict[str, Any]:
        if isinstance(part, TextContentPart):
            return {"type": part.type, "text": part.text}

        image_payload: dict[str, Any] = {"url": part.image_url}
        if part.detail:
            image_payload["detail"] = part.detail
        return {"type": part.type, "image_url": image_payload}


@dataclass(slots=True)
class UsageStats:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    include_reasoning: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    provider: str
    model: str
    assistant_message: Message
    reasoning: str
    reasoning_details: list[dict[str, Any]]
    text: str
    structured_output: Any | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: UsageStats
    response_id: str | None
    latency_ms: float
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    raw_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ProviderEvent:
    type: str
    delta: str | None = None
    message: str | None = None
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
