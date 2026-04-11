"""
Live OpenRouter tests for `call_llm`.

Requires `OPENROUTER_API_KEY` (set in the shell or in repo-root `.env`;
`tests/integration/conftest.py` loads `.env` via python-dotenv). Uses
`DEFAULT_MODEL` from `aca.llm.models`.

Run (use **-s** so you see model output on stdout — pytest hides prints otherwise):

  cd <repo-root> && PYTHONPATH=. pytest tests/integration/test_llm_openrouter_live.py -v -s

Exclude from routine runs:
  pytest -m "not integration"
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import pytest
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field

from aca.llm.client import call_llm
from aca.llm.models import DEFAULT_MODEL

pytestmark = pytest.mark.integration


# ── Structured final answer (user-requested schema) ───────────────────────────


class FinalOutput(BaseModel):
    final_number: float = Field(description="Numeric result after weather × 7")
    is_prime: bool = Field(description="True if final_number is a prime integer")
    random_joke: str = Field(description="A random funny one-liner")


# ── OpenAI-format tools (fake weather + calculator) ───────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Return fake current weather for a city. "
                "Always includes temperature_c (Celsius) for use in calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Paris"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers and return the product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def _fake_weather_payload(city: str) -> dict[str, Any]:
    if "paris" in city.lower():
        return {"city": "Paris", "temperature_c": 22, "condition": "clear", "note": "fake_stub"}
    return {"city": city, "temperature_c": 15, "condition": "cloudy", "note": "fake_stub"}


def _dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_weather":
        city = str(arguments.get("city", ""))
        return json.dumps(_fake_weather_payload(city))
    if name == "multiply":
        a = float(arguments["a"])
        b = float(arguments["b"])
        return json.dumps({"product": a * b})
    raise AssertionError(f"unknown tool: {name}")


def _parse_json_content(raw: str | None) -> dict[str, Any]:
    if not raw:
        raise ValueError("empty model content")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def _is_prime_int(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(n**0.5) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def _stream_tap_and_stats(phase: str) -> tuple[Callable[[ChatCompletionChunk], None], dict[str, int]]:
    """Build a ``stream_callback`` that prints incremental deltas and counts chunks."""

    stats: dict[str, int] = {
        "chunks": 0,
        "content_deltas": 0,
        "tool_delta_chunks": 0,
    }

    def callback(chunk: ChatCompletionChunk) -> None:
        stats["chunks"] += 1
        if not chunk.choices:
            return
        delta = chunk.choices[0].delta
        if delta.content:
            stats["content_deltas"] += 1
            print(delta.content, end="", flush=True)
        if delta.tool_calls:
            stats["tool_delta_chunks"] += 1
            for tc in delta.tool_calls:
                bits: list[str] = []
                if tc.function:
                    if tc.function.name:
                        bits.append(f"name+={tc.function.name!r}")
                    if tc.function.arguments:
                        bits.append(f"args+={tc.function.arguments!r}")
                print(f"\n[{phase}] tool idx={tc.index} " + " ".join(bits), flush=True)

    return callback, stats


def _emit_live_response_banner(
    *,
    phase: str,
    model: str,
    user_prompt: str | None,
    response,
) -> None:
    sep = "=" * 72
    print(f"\n{sep}", flush=True)
    print(f"ACA live LLM — {phase}", flush=True)
    print(sep, flush=True)
    print(f"model (requested):     {model}", flush=True)
    if user_prompt is not None:
        print(f"user prompt:           {user_prompt!r}", flush=True)
    print("--- assistant text ---", flush=True)
    print(response.content if response.content else "(empty)", flush=True)
    print("--- metadata ---", flush=True)
    print(f"stop_reason:           {response.stop_reason}", flush=True)
    print(f"input_tokens:          {response.input_tokens}", flush=True)
    print(f"output_tokens:         {response.output_tokens}", flush=True)
    print(f"latency_ms:            {response.latency_ms}", flush=True)
    print(f"tool_calls:            {response.tool_calls}", flush=True)
    print(f"thinking_blocks len:   {len(response.thinking_blocks)}", flush=True)
    print(f"raw is None:           {response.raw is None}", flush=True)
    print(f"{sep}\n", flush=True)


def test_call_llm_openrouter_smoke() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — skipping live OpenRouter call")

    user_prompt = "What is the capital of France? Answer in one short phrase."
    response = call_llm(
        [{"role": "user", "content": user_prompt}],
        model=DEFAULT_MODEL,
    )

    _emit_live_response_banner(
        phase="smoke (no tools)",
        model=DEFAULT_MODEL,
        user_prompt=user_prompt,
        response=response,
    )

    assert response.stop_reason != "error", f"API error: {response.content}"
    assert response.content
    assert "paris" in response.content.lower()


def test_call_llm_tools_then_structured_final_output() -> None:
    """
    Weather + multiply via tools, then ``FinalOutput`` via json_schema — all with
    ``stream=True`` and ``stream_callback`` so tool deltas and JSON text print
    incrementally; stream stats assert multiple SSE chunks.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — skipping live OpenRouter call")

    system = (
        "You have tools get_weather and multiply. "
        "For questions about Paris weather and multiplying by 7: "
        "first call get_weather for Paris "
        "then call multiply with a=temperature_c and b=7. "
        "Do not invent temperatures; use tool results only."
    )
    user_ask = (
        "What is the weather in Paris? Then multiply the Celsius temperature by 7. "
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_ask}]
    max_tool_rounds = 8
    expected_product = 22.0 * 7.0  # fake Paris temperature_c

    for round_idx in range(max_tool_rounds):
        phase = f"tool round {round_idx + 1} (stream)"
        cb, st = _stream_tap_and_stats(phase)
        print(f"\n--- {phase} (live tape) ---", flush=True)
        response = call_llm(
            messages,
            model=DEFAULT_MODEL,
            system_prompt=system,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            stream_callback=cb,
        )
        print(f"\n--- end {phase} ---", flush=True)
        print(
            f"[{phase}] stream stats: chunks={st['chunks']} "
            f"content_deltas={st['content_deltas']} tool_delta_chunks={st['tool_delta_chunks']}",
            flush=True,
        )

        assert response.stop_reason != "error", f"API error: {response.content}"
        assert st["chunks"] >= 2, "streaming should yield at least two SSE chunks"

        _emit_live_response_banner(
            phase=f"{phase} (reassembled)",
            model=DEFAULT_MODEL,
            user_prompt=None,
            response=response,
        )

        if response.tool_calls:
            assert st["tool_delta_chunks"] >= 1, (
                "tool calls should arrive as at least one streamed tool delta chunk"
            )

        if not response.tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": response.tool_calls,
            }
        )
        for tc in response.tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result_json = _dispatch_tool(name, args)
            print(f"--- tool result ({name}) ---\n{result_json}\n", flush=True)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_json,
                }
            )
    else:
        pytest.fail("too many tool rounds without finishing")

    messages.append(
        {
            "role": "user",
            "content": (
                "Now output ONLY a JSON object matching the provided schema: "
                "final_number = the product from multiply, "
                "is_prime = whether that integer is prime, "
                "random_joke = one short funny one-liner. No markdown, no extra keys."
            ),
        }
    )

    schema = FinalOutput.model_json_schema()
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "FinalOutput",
            "strict": True,
            "schema": schema,
        },
    }

    phase_final = "structured FinalOutput (stream)"
    cb_final, st_final = _stream_tap_and_stats(phase_final)
    print(f"\n--- {phase_final} (live tape, JSON fragments) ---", flush=True)
    final = call_llm(
        messages,
        model=DEFAULT_MODEL,
        system_prompt=system,
        response_format=response_format,
        stream=True,
        stream_callback=cb_final,
    )
    print(f"\n--- end {phase_final} ---", flush=True)
    print(
        f"[{phase_final}] stream stats: chunks={st_final['chunks']} "
        f"content_deltas={st_final['content_deltas']} "
        f"tool_delta_chunks={st_final['tool_delta_chunks']}",
        flush=True,
    )

    assert final.stop_reason != "error", f"API error: {final.content}"
    assert st_final["chunks"] >= 2
    assert st_final["content_deltas"] >= 2, (
        "structured JSON should stream as multiple text deltas (not a single blob)"
    )

    _emit_live_response_banner(
        phase="structured FinalOutput (reassembled)",
        model=DEFAULT_MODEL,
        user_prompt="(see prior turns + schema instruction)",
        response=final,
    )

    assert final.content
    parsed = _parse_json_content(final.content)
    out = FinalOutput.model_validate(parsed)

    print("--- parsed FinalOutput ---", flush=True)
    print(out.model_dump_json(indent=2), flush=True)
    print(flush=True)

    assert out.final_number == pytest.approx(expected_product, rel=0, abs=1e-6)
    n_int = int(round(out.final_number))
    assert n_int == 154
    assert out.is_prime == _is_prime_int(n_int)
    assert out.random_joke.strip()
