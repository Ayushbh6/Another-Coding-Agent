from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI

from aca.core_types import MasterClassification, TurnIntent
from aca.llm.openrouter_client import create_openrouter_client
from aca.llm.types import Message


MASTER_CLASSIFICATION_PROMPT = """
You are the Master classifier for a coding agent.

Classify the current user turn into exactly one intent:
- chat: casual or conversational, no meaningful repo work required
- analyze: inspect or explain code/repo behavior, but do not change code
- implement: make or plan to make code or file changes

Return a strict JSON object with these keys:
- intent
- task_title
- task_description
- reasoning_summary

Rules:
- intent must be one of: chat, analyze, implement
- task_title and task_description must be concise but useful
- for chat, task_title and task_description may be empty strings
- reasoning_summary must be brief and explain the classification
- do not include markdown or any text outside the JSON object
""".strip()


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
        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=request_messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            completion = self._client.chat.completions.create(
                model=model,
                messages=request_messages,
            )

        content = (completion.choices[0].message.content or "").strip()
        payload = _parse_json_object(content)

        intent_raw = str(payload.get("intent", "chat")).strip().lower()
        try:
            intent = TurnIntent(intent_raw)
        except ValueError as exc:
            raise ValueError(f"Unsupported turn intent returned by classifier: {intent_raw}") from exc

        return MasterClassification(
            intent=intent,
            task_title=str(payload.get("task_title", "")).strip(),
            task_description=str(payload.get("task_description", "")).strip(),
            reasoning_summary=str(payload.get("reasoning_summary", "")).strip(),
        )


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Master classifier did not return valid JSON: {content}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Master classifier returned a non-object JSON payload.")

    return parsed
