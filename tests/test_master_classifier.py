from __future__ import annotations

import json
import unittest

from aca.llm.types import Message
from aca.master import MASTER_CLASSIFICATION_RESPONSE_FORMAT, OpenRouterMasterClassifier


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletion(
            json.dumps(
                {
                    "intent": "implement",
                    "task_title": "Add feature",
                    "task_description": "Implement the requested feature.",
                    "reasoning_summary": "The user asked for a change.",
                }
            )
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


class MasterClassifierTests(unittest.TestCase):
    def test_classifier_uses_strict_json_schema_response_format(self) -> None:
        client = _FakeClient()
        classifier = OpenRouterMasterClassifier(client=client)

        result = classifier.classify(
            model="fake-model",
            messages=[Message(role="assistant", content="Previous reply")],
            user_input="Please implement this change.",
        )

        self.assertEqual(result.intent.value, "implement")
        call = client.chat.completions.calls[-1]
        self.assertEqual(call["response_format"], MASTER_CLASSIFICATION_RESPONSE_FORMAT)
        self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertTrue(call["response_format"]["json_schema"]["strict"])
        self.assertEqual(call["response_format"]["json_schema"]["name"], "master_classification")

    def test_classifier_accepts_fenced_json(self) -> None:
        client = _FakeClient()
        client.chat.completions.create = lambda **kwargs: _FakeCompletion(
            "```json\n"
            + json.dumps(
                {
                    "intent": "chat",
                    "task_title": "Greeting",
                    "task_description": "Casual greeting.",
                    "reasoning_summary": "No repo work requested.",
                }
            )
            + "\n```"
        )
        classifier = OpenRouterMasterClassifier(client=client)

        result = classifier.classify(
            model="fake-model",
            messages=[],
            user_input="hey there",
        )

        self.assertEqual(result.intent.value, "chat")
        self.assertEqual(result.task_title, "")

    def test_classifier_normalizes_chat_to_empty_task_fields(self) -> None:
        client = _FakeClient()
        client.chat.completions.create = lambda **kwargs: _FakeCompletion(
            json.dumps(
                {
                    "intent": "chat",
                    "task_title": "General greeting",
                    "task_description": "Casual conversation.",
                    "reasoning_summary": "No repo work requested.",
                }
            )
        )
        classifier = OpenRouterMasterClassifier(client=client)

        result = classifier.classify(
            model="fake-model",
            messages=[],
            user_input="hey there",
        )

        self.assertEqual(result.intent.value, "chat")
        self.assertEqual(result.task_title, "")
        self.assertEqual(result.task_description, "")
        self.assertEqual(result.reasoning_summary, "")


if __name__ == "__main__":
    unittest.main()
