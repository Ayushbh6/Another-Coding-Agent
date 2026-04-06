from __future__ import annotations

import json
from typing import Any

import tiktoken


def estimate_text_tokens(text: str | None, *, model: str | None = None) -> int:
    if not text:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def estimate_json_tokens(payload: Any, *, model: str | None = None) -> int:
    if payload is None:
        return 0
    return estimate_text_tokens(json.dumps(payload, sort_keys=True), model=model)

