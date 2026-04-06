from __future__ import annotations

import json
from typing import Any, Protocol

from openai import OpenAI

from aca.core_types import MasterClassification, TurnIntent
from aca.json_utils import parse_json_object_loose
from aca.llm.openrouter_client import create_openrouter_client
from aca.llm.types import Message
from aca.prompts import MASTER_CLASSIFICATION_PROMPT


MASTER_CLASSIFICATION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "master_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [intent.value for intent in TurnIntent],
                },
                "task_title": {"type": "string"},
                "task_description": {"type": "string"},
                "reasoning_summary": {"type": "string"},
            },
            "required": ["intent", "task_title", "task_description", "reasoning_summary"],
            "additionalProperties": False,
        },
    },
}


class MasterClassifier(Protocol):
    def classify(self, *, model: str, messages: list[Message], user_input: str) -> MasterClassification:
        ...


class OpenRouterMasterClassifier:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        title: str = "ACA-Master-Classifier",
        client: OpenAI | None = None,
    ) -> None:
        self._client = client or create_openrouter_client(
            api_key=api_key,
            base_url=base_url,
            http_referer=http_referer,
            title=title,
        )

    def classify(self, *, model: str, messages: list[Message], user_input: str) -> MasterClassification:
        request_messages = [
            {"role": "system", "content": MASTER_CLASSIFICATION_PROMPT},
            *[message.to_provider_dict() for message in messages],
            {"role": "user", "content": user_input},
        ]
        completion = self._client.chat.completions.create(
            model=model,
            messages=request_messages,
            response_format=MASTER_CLASSIFICATION_RESPONSE_FORMAT,
        )

        content = (completion.choices[0].message.content or "").strip()
        payload = _parse_json_object(content)

        intent_raw = str(payload.get("intent", "chat")).strip().lower()
        try:
            intent = TurnIntent(intent_raw)
        except ValueError as exc:
            raise ValueError(f"Unsupported turn intent returned by classifier: {intent_raw}") from exc

        if intent is TurnIntent.CHAT:
            return MasterClassification(
                intent=intent,
                task_title="",
                task_description="",
                reasoning_summary="",
            )

        return MasterClassification(
            intent=intent,
            task_title=str(payload.get("task_title", "")).strip(),
            task_description=str(payload.get("task_description", "")).strip(),
            reasoning_summary=str(payload.get("reasoning_summary", "")).strip(),
        )


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        parsed = parse_json_object_loose(content)
    except ValueError as exc:
        raise ValueError(f"Master classifier did not return valid JSON: {content}") from exc

    return parsed
