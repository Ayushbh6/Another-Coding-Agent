"""
Core LLM call function for ACA.

This is the single place where the OpenAI-compatible API is called.
No other module should initialise an LLM client or call the API directly.

Supports:
- OpenRouter (default) and OpenAI via the OpenAI-compatible API
- Interleaved thinking (reasoning tokens preserved across turns)
- Streaming (reassembled internally; caller always receives an LLMResponse;
  optional ``stream_callback`` per SSE chunk for incremental UI/tests)
- Structured output via response_format / json_schema
- Image inputs (URL or base64, injected into the last user message)
- tool_call_limit enforcement via Option-B soft-stop (final forced text call)
- Full DB logging to the llm_calls table when a db + context are supplied
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

from aca.llm.models import DEFAULT_MODEL
from aca.llm.providers import ProviderName, get_provider


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    call_id: str
    content: str | None
    tool_calls: list[dict]          # list of raw tool call dicts from the API
    stop_reason: str                # "end_turn" | "tool_use" | "tool_call_limit" | "max_tokens" | "error"
    thinking_blocks: list[dict]     # populated when thinking=True; empty list otherwise
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str
    raw: Any = field(repr=False)    # the raw API response object, always preserved


# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    """Read a local image file and return a base64 data-URL string."""
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
    mime = mime_map.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _inject_images(messages: list[dict], images: list[dict]) -> list[dict]:
    """
    Inject image content blocks into the last user message.

    Each entry in `images` must be one of:
      {"type": "url",  "url": "https://..."}
      {"type": "file", "path": "/local/path/to/image.png"}

    Per OpenRouter docs: send the text prompt first, then images.
    This function appends the image blocks after existing text content.
    """
    if not images:
        return messages

    msgs = [m.copy() for m in messages]

    # Find the last user message
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return msgs

    target = msgs[last_user_idx]

    # Normalise existing content to a list
    existing = target.get("content", "")
    if isinstance(existing, str):
        content_list: list[dict] = [{"type": "text", "text": existing}]
    else:
        content_list = list(existing)

    # Append each image block
    for img in images:
        if img["type"] == "url":
            content_list.append({
                "type": "image_url",
                "image_url": {"url": img["url"]},
            })
        elif img["type"] == "file":
            content_list.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(img["path"])},
            })

    target["content"] = content_list
    return msgs


# ── Streaming assembler ───────────────────────────────────────────────────────

def _assemble_stream(
    stream: Stream[ChatCompletionChunk],
    *,
    stream_callback: Callable[[ChatCompletionChunk], None] | None = None,
) -> tuple[str | None, list[dict], str, int, int]:
    """
    Consume a streaming response and reassemble it into:
      (content, tool_calls, stop_reason, input_tokens, output_tokens)

    Tool call deltas are accumulated by index and merged into complete dicts.
    If ``stream_callback`` is set, it is invoked once per SSE chunk (before
    aggregation) so callers can observe incremental content and tool deltas.
    """
    content_parts: list[str] = []
    tool_call_accum: dict[int, dict] = {}
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    for chunk in stream:
        if stream_callback is not None:
            stream_callback(chunk)
        choice = chunk.choices[0] if chunk.choices else None

        # Token usage (usually only in the final chunk)
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens or 0
            output_tokens = chunk.usage.completion_tokens or 0

        if choice is None:
            continue

        delta = choice.delta

        if delta.content:
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_call_accum:
                    tool_call_accum[idx] = {
                        "id": tc_delta.id or "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_call_accum[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_call_accum[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_call_accum[idx]["function"]["arguments"] += tc_delta.function.arguments

        if choice.finish_reason:
            stop_reason = _normalise_stop_reason(choice.finish_reason)

    content = "".join(content_parts) or None
    tool_calls = [tool_call_accum[i] for i in sorted(tool_call_accum)]
    return content, tool_calls, stop_reason, input_tokens, output_tokens


def _normalise_stop_reason(raw: str | None) -> str:
    mapping = {
        "stop":         "end_turn",
        "end_turn":     "end_turn",
        "tool_calls":   "tool_use",
        "tool_use":     "tool_use",
        "max_tokens":   "max_tokens",
        "length":       "max_tokens",
        "error":        "error",
    }
    return mapping.get(raw or "", raw or "end_turn")


_PSEUDO_TOOL_BLOCK_RE = re.compile(
    r"<minimax:tool_call>\s*(?P<body>.*?)\s*</minimax:tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PSEUDO_INVOKE_RE = re.compile(
    r"<invoke\s+name=\"(?P<name>[^\"]+)\">\s*(?P<body>.*?)\s*</invoke>",
    re.DOTALL | re.IGNORECASE,
)
_PSEUDO_PARAM_RE = re.compile(
    r"<parameter\s+name=\"(?P<name>[^\"]+)\">(?P<value>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_pseudo_tool_markup(content: str | None) -> tuple[str | None, list[dict]]:
    """
    Parse Minimax-style pseudo-tool XML emitted as plain text.

    Returns ``(clean_content, tool_calls)``. ``clean_content`` is the original
    assistant text with the pseudo-tool block removed. If no pseudo markup is
    present, returns the original content and an empty list.
    """
    if not content:
        return content, []

    match = _PSEUDO_TOOL_BLOCK_RE.search(content)
    if not match:
        return content, []

    tool_calls: list[dict] = []
    for invoke in _PSEUDO_INVOKE_RE.finditer(match.group("body")):
        args = {
            param.group("name"): param.group("value").strip()
            for param in _PSEUDO_PARAM_RE.finditer(invoke.group("body"))
        }
        tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": invoke.group("name").strip(),
                    "arguments": json.dumps(args),
                },
            }
        )

    if not tool_calls:
        return content, []

    clean_content = _PSEUDO_TOOL_BLOCK_RE.sub("", content).strip()
    return clean_content or None, tool_calls


# ── DB logging helper ─────────────────────────────────────────────────────────

def _log_llm_call(
    db: Any,
    call_id: str,
    turn_id: str,
    session_id: str,
    agent: str,
    call_index: int,
    model: str,
    system_prompt: str,
    messages_json: str,
    response: LLMResponse,
    started_at: int,
    temperature: float | None,
    attempt_number: int,
) -> None:
    """
    Write one row to the llm_calls table.

    `db` is expected to be a sqlite3.Connection (or compatible).
    Logging failures are caught and printed rather than crashing the call.
    """
    try:
        db.execute(
            """
            INSERT INTO llm_calls (
                call_id, turn_id, session_id, agent, call_index,
                system_prompt, messages_json, response_text, stop_reason,
                input_tokens, output_tokens, latency_ms, model,
                temperature, started_at, error, attempt_number
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                call_id, turn_id, session_id, agent, call_index,
                system_prompt, messages_json,
                response.content if response.stop_reason != "error" else None,
                response.stop_reason,
                response.input_tokens, response.output_tokens, response.latency_ms,
                model, temperature, started_at,
                None if response.stop_reason != "error" else response.content,
                attempt_number,
            ),
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[ACA][llm_calls] DB log failed: {exc}")


# ── Core call function ────────────────────────────────────────────────────────

def call_llm(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict = "auto",
    response_format: dict | None = None,
    images: list[dict] | None = None,
    thinking: bool = False,
    stream: bool = False,
    stream_callback: Callable[[ChatCompletionChunk], None] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    provider: str | ProviderName = ProviderName.OPENROUTER,
    # DB logging context — all four must be supplied together or logging is skipped
    db: Any = None,
    turn_id: str | None = None,
    session_id: str | None = None,
    agent: str | None = None,
    call_index: int = 0,
    attempt_number: int = 1,
) -> LLMResponse:
    """
    Make a single LLM API call and return a structured LLMResponse.

    Parameters
    ----------
    messages:
        The conversation messages list (excluding the system prompt).
        The system prompt is prepended internally if provided.
    model:
        Model identifier string. Defaults to DEFAULT_MODEL.
    system_prompt:
        If provided, prepended as a {"role": "system", ...} message.
    tools:
        List of tool schema dicts in OpenAI function-calling format.
        Pass None to make a plain completion call.
    tool_choice:
        "auto" (default), "none", or a forced-tool dict.
    response_format:
        Structured output schema dict. Example:
          {"type": "json_schema", "json_schema": {"name": "...", "strict": True, "schema": {...}}}
    images:
        List of image dicts to inject into the last user message.
        Each entry:  {"type": "url", "url": "..."}
                  or {"type": "file", "path": "/path/to/image.png"}
    thinking:
        If True, enables interleaved reasoning tokens (extra_body reasoning flag).
        The returned LLMResponse.thinking_blocks carries the raw reasoning_details
        so the caller can pass them back in the next turn's message history.
    stream:
        If True, streams the response and reassembles it before returning.
        The caller always receives the same LLMResponse shape regardless.
    stream_callback:
        If ``stream`` is True, invoked once per raw ``ChatCompletionChunk``
        from the API (including chunks with only usage metadata). Ignored when
        ``stream`` is False. Use for tests, logging, or UI token-by-token display.
    temperature:
        Sampling temperature. Provider default if None.
    max_tokens:
        Max output tokens. Provider default if None.
    provider:
        Provider name string or ProviderName enum. Defaults to "openrouter".
    db / turn_id / session_id / agent / call_index / attempt_number:
        DB logging context. All of db, turn_id, session_id, agent must be
        non-None for a log row to be written.
    """
    prov = get_provider(provider)
    client = OpenAI(base_url=prov.base_url, api_key=prov.api_key())

    # Build the full message list
    full_messages: list[dict] = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(_inject_images(messages, images or []))

    # Build API kwargs
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "stream": stream,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if thinking:
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    started_at = int(time.time() * 1000)
    call_id = str(uuid.uuid4())

    try:
        raw = client.chat.completions.create(**kwargs)

        if stream:
            content, tool_calls, stop_reason, input_tokens, output_tokens = _assemble_stream(
                raw,
                stream_callback=stream_callback,
            )
            thinking_blocks: list[dict] = []
        else:
            choice = raw.choices[0]
            msg = choice.message
            content = msg.content
            stop_reason = _normalise_stop_reason(choice.finish_reason)

            # Raw tool calls → plain dicts
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

            # Preserve thinking/reasoning blocks when thinking=True
            thinking_blocks = []
            if thinking and hasattr(msg, "reasoning_details") and msg.reasoning_details:
                thinking_blocks = [
                    rd if isinstance(rd, dict) else rd.model_dump()
                    for rd in msg.reasoning_details
                ]

            input_tokens = raw.usage.prompt_tokens if raw.usage else 0
            output_tokens = raw.usage.completion_tokens if raw.usage else 0

        if not tool_calls:
            content, pseudo_tool_calls = _parse_pseudo_tool_markup(content)
            if pseudo_tool_calls:
                tool_calls = pseudo_tool_calls
                stop_reason = "tool_use"

        latency_ms = int(time.time() * 1000) - started_at

        response = LLMResponse(
            call_id=call_id,
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            thinking_blocks=thinking_blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=model,
            raw=raw,
        )

    except Exception as exc:  # noqa: BLE001
        latency_ms = int(time.time() * 1000) - started_at
        response = LLMResponse(
            call_id=call_id,
            content=str(exc),
            tool_calls=[],
            stop_reason="error",
            thinking_blocks=[],
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            model=model,
            raw=None,
        )

    # DB logging (skipped if any context field is missing)
    if db is not None and turn_id and session_id and agent:
        _log_llm_call(
            db=db,
            call_id=call_id,
            turn_id=turn_id,
            session_id=session_id,
            agent=agent,
            call_index=call_index,
            model=model,
            system_prompt=system_prompt or "",
            messages_json=json.dumps(full_messages),
            response=response,
            started_at=started_at,
            temperature=temperature,
            attempt_number=attempt_number,
        )

    return response


# ── Soft-stop wrapper (Option B tool_call_limit enforcement) ──────────────────

def call_llm_with_limit(
    messages: list[dict],
    tool_call_limit: int,
    *,
    # All call_llm kwargs forwarded
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict = "auto",
    response_format: dict | None = None,
    images: list[dict] | None = None,
    thinking: bool = False,
    stream: bool = False,
    stream_callback: Callable[[ChatCompletionChunk], None] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    provider: str | ProviderName = ProviderName.OPENROUTER,
    db: Any = None,
    turn_id: str | None = None,
    session_id: str | None = None,
    agent: str | None = None,
    # Mutable call index shared across the loop — pass a list([0]) so we can increment
    call_index_ref: list[int] | None = None,
) -> tuple[LLMResponse, list[dict]]:
    """
    Agentic tool-call loop with a hard upper bound (tool_call_limit).

    When the limit is reached, one final call is made with tool_choice="none"
    to force a clean text response (Option B soft-stop).

    Returns
    -------
    (final_response, updated_messages)
        updated_messages includes all assistant + tool turns appended during
        the loop so the caller can persist them as conversation history.
    """
    if call_index_ref is None:
        call_index_ref = [0]

    loop_messages = list(messages)
    tool_calls_made = 0

    while True:
        # If we've hit the limit, force a final text response
        forced_stop = tool_calls_made >= tool_call_limit
        effective_tool_choice = "none" if forced_stop else tool_choice
        effective_tools = None if forced_stop else tools

        response = call_llm(
            loop_messages,
            model=model,
            system_prompt=system_prompt,
            tools=effective_tools,
            tool_choice=effective_tool_choice,
            response_format=response_format,
            images=images if call_index_ref[0] == 0 else None,  # images only on first call
            thinking=thinking,
            stream=stream,
            stream_callback=stream_callback,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            db=db,
            turn_id=turn_id,
            session_id=session_id,
            agent=agent,
            call_index=call_index_ref[0],
        )
        call_index_ref[0] += 1

        if response.stop_reason == "error":
            return response, loop_messages

        # Append assistant message to the running history
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant_msg["tool_calls"] = response.tool_calls
        if thinking and response.thinking_blocks:
            assistant_msg["reasoning_details"] = response.thinking_blocks
        loop_messages.append(assistant_msg)

        # If no tool calls requested (or we forced stop), we're done
        if not response.tool_calls or forced_stop:
            if forced_stop and response.tool_calls:
                # Shouldn't happen with tool_choice="none", but guard anyway
                response = LLMResponse(
                    call_id=response.call_id,
                    content=response.content,
                    tool_calls=[],
                    stop_reason="tool_call_limit",
                    thinking_blocks=response.thinking_blocks,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    model=response.model,
                    raw=response.raw,
                )
            return response, loop_messages

        # Hand tool calls back to the caller for dispatch.
        # The caller must append tool result messages to loop_messages and call
        # this function again — OR use BaseAgent which handles the full loop.
        # Here we return after each batch so BaseAgent can dispatch and log.
        tool_calls_made += len(response.tool_calls)
        return response, loop_messages
