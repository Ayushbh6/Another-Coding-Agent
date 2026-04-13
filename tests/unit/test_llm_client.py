from __future__ import annotations

import json

from aca.llm.client import _parse_pseudo_tool_markup


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
