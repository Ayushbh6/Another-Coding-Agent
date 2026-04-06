from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def parse_json_object_loose(content: str) -> dict[str, Any]:
    cleaned = strip_markdown_code_fence(content.strip())
    try:
        parsed = json.loads(cleaned)
    except JSONDecodeError:
        parsed = _extract_first_json_value(cleaned)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")

    return parsed


def strip_markdown_code_fence(content: str) -> str:
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    if not lines:
        return content
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_value(content: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[index:])
            return parsed
        except JSONDecodeError:
            continue
    raise ValueError("No JSON object found.")
