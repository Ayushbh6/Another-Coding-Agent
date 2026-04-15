from __future__ import annotations

import json

from aca.llm.client import _build_extra_body, _parse_pseudo_tool_markup
from aca.llm.providers import ProviderName


def test_parse_pseudo_tool_markup_extracts_tool_calls() -> None:
    raw = """
Let me inspect the repo.
<minimax:tool_call>
<invoke name="read_file">
<parameter name="path">docs/ARCHITECTURE.md</parameter>
</invoke>
<invoke name="search_repo">
<parameter name="pattern">JamesAgent</parameter>
</invoke>
</minimax:tool_call>
"""

    content, tool_calls = _parse_pseudo_tool_markup(raw)

    assert content == "Let me inspect the repo."
    assert [tc["function"]["name"] for tc in tool_calls] == ["read_file", "search_repo"]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "docs/ARCHITECTURE.md"}
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"pattern": "JamesAgent"}


def test_parse_pseudo_tool_markup_no_markup_is_noop() -> None:
    content, tool_calls = _parse_pseudo_tool_markup("plain response")
    assert content == "plain response"
    assert tool_calls == []


def test_build_extra_body_is_empty_for_non_kimi_openrouter_when_thinking_off() -> None:
    extra = _build_extra_body(
        model="minimax/minimax-m2.7:nitro",
        provider=ProviderName.OPENROUTER,
        thinking=False,
    )

    assert extra is None


def test_build_extra_body_adds_kimi_thinking_disable_flag() -> None:
    extra = _build_extra_body(
        model="moonshotai/kimi-k2.5:nitro",
        provider=ProviderName.OPENROUTER,
        thinking=False,
    )

    assert extra == {
        "reasoning": {"effort": "none", "exclude": True},
        "thinking": {"type": "disabled"},
    }


def test_build_extra_body_enables_reasoning_for_kimi_when_thinking_on() -> None:
    extra = _build_extra_body(
        model="moonshotai/kimi-k2.5:nitro",
        provider=ProviderName.OPENROUTER,
        thinking=True,
    )

    assert extra == {
        "reasoning": {"enabled": True},
        "thinking": {"type": "enabled"},
    }
