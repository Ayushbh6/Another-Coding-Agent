from __future__ import annotations

import os

from openai import OpenAI


def create_openrouter_client(
    api_key: str | None = None,
    *,
    base_url: str = "https://openrouter.ai/api/v1",
    http_referer: str | None = None,
    title: str | None = None,
) -> OpenAI:
    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return OpenAI(
        base_url=base_url,
        api_key=resolved_api_key,
        default_headers={
            "HTTP-Referer": http_referer or "http://localhost:8000",
            "X-Title": title or "ACA-Coding-Agent",
        },
    )

