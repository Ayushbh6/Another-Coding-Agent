from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


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
class Message:
    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_provider_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            payload["content"] = self.content

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

        return payload


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
    text: str
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
