from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from aca.json_utils import parse_json_object_loose

from ..openrouter_client import create_openrouter_client
from ..types import Message, ProviderEvent, ProviderRequest, RunResult, ToolCall, UsageStats
from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        title: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        self._base_url = base_url
        self._headers = {
            "HTTP-Referer": http_referer or "http://localhost:8000",
            "X-Title": title or "ACA-Coding-Agent",
        }
        self._client = client or create_openrouter_client(
            api_key=resolved_api_key,
            base_url=self._base_url,
            http_referer=self._headers["HTTP-Referer"],
            title=self._headers["X-Title"],
        )

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        started_at = time.perf_counter()
        raw_chunks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_details: list[dict[str, Any]] = []
        tool_call_builders: dict[int, dict[str, Any]] = {}
        response_id: str | None = None
        finish_reason: str | None = None
        provider_metadata: dict[str, Any] = {}
        usage = UsageStats()

        stream = self._client.chat.completions.create(
            **self._build_request_kwargs(request),
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            raw_chunk = self._to_dict(chunk)
            raw_chunks.append(raw_chunk)

            if response_id is None:
                response_id = getattr(chunk, "id", None) or raw_chunk.get("id")

            provider_metadata.update(self._extract_chunk_metadata(chunk, raw_chunk))
            chunk_usage = self._extract_usage(getattr(chunk, "usage", None), raw_chunk)
            if chunk_usage.total_tokens is not None:
                usage = chunk_usage

            for choice in getattr(chunk, "choices", []) or []:
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning_delta = self._extract_reasoning_delta(delta, raw_chunk)
                reasoning_detail_deltas = self._extract_reasoning_details_delta(delta, raw_chunk)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    yield ProviderEvent(
                        type="reasoning.delta",
                        delta=reasoning_delta,
                        raw=raw_chunk,
                    )

                if reasoning_detail_deltas:
                    reasoning_details.extend(reasoning_detail_deltas)
                    if not reasoning_delta:
                        detail_text = self._extract_reasoning_text_from_details(reasoning_detail_deltas)
                        if detail_text:
                            reasoning_parts.append(detail_text)
                            yield ProviderEvent(
                                type="reasoning.delta",
                                delta=detail_text,
                                raw=raw_chunk,
                            )

                text_delta = self._extract_text_delta(delta)
                if text_delta:
                    text_parts.append(text_delta)
                    yield ProviderEvent(
                        type="text.delta",
                        delta=text_delta,
                        raw=raw_chunk,
                    )

                for tool_event in self._extract_tool_call_deltas(delta, tool_call_builders, raw_chunk):
                    yield tool_event

        tool_calls = self._finalize_tool_calls(tool_call_builders)
        for tool_call in tool_calls:
            yield ProviderEvent(
                type="tool_call.completed",
                tool_call=tool_call,
                metadata={"tool_name": tool_call.name},
            )

        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        assistant_message = Message(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls,
            reasoning=reasoning or None,
            reasoning_details=list(reasoning_details),
        )
        latency_ms = (time.perf_counter() - started_at) * 1000

        result = RunResult(
            provider=self.provider_name,
            model=request.model,
            assistant_message=assistant_message,
            reasoning=reasoning,
            reasoning_details=list(reasoning_details),
            text=text,
            structured_output=self._parse_structured_output(text, request.response_format),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
            latency_ms=latency_ms,
            provider_metadata=provider_metadata,
            raw_response=raw_chunks[-1] if raw_chunks else {},
            raw_chunks=raw_chunks,
        )

        yield ProviderEvent(
            type="response.completed",
            result=result,
            metadata={
                "response_id": response_id,
                "finish_reason": finish_reason,
                "latency_ms": latency_ms,
            },
            raw=result.raw_response,
        )

    def _build_request_kwargs(self, request: ProviderRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_provider_dict() for message in request.messages],
            "tools": request.tools,
            "tool_choice": request.tool_choice,
        }

        if request.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = request.parallel_tool_calls

        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        extra_body = dict(request.extra_body)
        if request.provider is not None:
            extra_body["provider"] = request.provider
        if request.reasoning is not None:
            extra_body["reasoning"] = request.reasoning
        elif "reasoning" not in extra_body and "include_reasoning" not in extra_body:
            extra_body["reasoning"] = {} if request.include_reasoning else {"exclude": True}

        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    def _extract_text_delta(self, delta: Any) -> str | None:
        content = getattr(delta, "content", None)
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts) or None
        return None

    def _extract_reasoning_delta(self, delta: Any, raw_chunk: dict[str, Any]) -> str | None:
        for key in ("reasoning", "reasoning_content"):
            value = getattr(delta, key, None)
            if value:
                return value

        choices = raw_chunk.get("choices") or []
        if not choices:
            return None

        raw_delta = choices[0].get("delta") or {}
        for key in ("reasoning", "reasoning_content"):
            value = raw_delta.get(key)
            if value:
                return value

        return None

    def _extract_reasoning_details_delta(self, delta: Any, raw_chunk: dict[str, Any]) -> list[dict[str, Any]]:
        details = getattr(delta, "reasoning_details", None)
        if details:
            return [self._to_dict(item) for item in details if self._to_dict(item)]

        choices = raw_chunk.get("choices") or []
        if not choices:
            return []

        raw_delta = choices[0].get("delta") or {}
        raw_details = raw_delta.get("reasoning_details") or []
        return [detail for detail in raw_details if isinstance(detail, dict)]

    def _extract_reasoning_text_from_details(self, details: list[dict[str, Any]]) -> str | None:
        text_parts = [str(detail.get("text", "")) for detail in details if detail.get("text")]
        combined = "".join(text_parts)
        return combined or None

    def _extract_tool_call_deltas(
        self,
        delta: Any,
        tool_call_builders: dict[int, dict[str, Any]],
        raw_chunk: dict[str, Any],
    ) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        tool_calls = getattr(delta, "tool_calls", None) or []

        for tool_call_delta in tool_calls:
            index = getattr(tool_call_delta, "index", None)
            if index is None:
                index = 0

            builder = tool_call_builders.setdefault(
                index,
                {"id": "", "name": "", "arguments_parts": []},
            )

            tool_call_id = getattr(tool_call_delta, "id", None)
            if tool_call_id:
                builder["id"] = tool_call_id

            function = getattr(tool_call_delta, "function", None)
            if function is None:
                continue

            tool_name = getattr(function, "name", None)
            if tool_name:
                builder["name"] = tool_name

            arguments_delta = getattr(function, "arguments", None)
            if arguments_delta:
                builder["arguments_parts"].append(arguments_delta)
                events.append(
                    ProviderEvent(
                        type="tool_call.delta",
                        delta=arguments_delta,
                        tool_call=ToolCall(
                            id=builder["id"],
                            name=builder["name"],
                            arguments="".join(builder["arguments_parts"]),
                        ),
                        raw=raw_chunk,
                    )
                )

        return events

    def _finalize_tool_calls(self, tool_call_builders: dict[int, dict[str, Any]]) -> list[ToolCall]:
        finalized: list[ToolCall] = []
        for index in sorted(tool_call_builders):
            builder = tool_call_builders[index]
            finalized.append(
                ToolCall(
                    id=builder["id"],
                    name=builder["name"],
                    arguments="".join(builder["arguments_parts"]),
                )
            )
        return finalized

    def _extract_chunk_metadata(self, chunk: Any, raw_chunk: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        for key in ("id", "model", "created", "system_fingerprint", "object"):
            value = getattr(chunk, key, None)
            if value is None:
                value = raw_chunk.get(key)
            if value is not None:
                metadata[key] = value

        provider_info = raw_chunk.get("provider")
        if provider_info is not None:
            metadata["provider"] = provider_info

        return metadata

    def _extract_usage(self, usage_obj: Any, raw_chunk: dict[str, Any]) -> UsageStats:
        usage_data = self._to_dict(usage_obj) if usage_obj is not None else raw_chunk.get("usage") or {}
        if not usage_data:
            return UsageStats()

        completion_details = usage_data.get("completion_tokens_details") or {}
        prompt_details = usage_data.get("prompt_tokens_details") or {}

        return UsageStats(
            input_tokens=usage_data.get("prompt_tokens"),
            output_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            cache_creation_input_tokens=prompt_details.get("cached_tokens_internal"),
            cache_read_input_tokens=prompt_details.get("cached_tokens"),
            raw=usage_data,
        )

    def _to_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return {}

    def _parse_structured_output(self, text: str, response_format: dict[str, Any] | None) -> Any | None:
        if not text or not response_format:
            return None

        response_type = response_format.get("type")
        if response_type not in {"json_object", "json_schema"}:
            return None

        try:
            return parse_json_object_loose(text)
        except ValueError:
            return None
