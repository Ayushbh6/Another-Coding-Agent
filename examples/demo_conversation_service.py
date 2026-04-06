from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aca.config import get_settings
from aca.llm.providers import OpenRouterProvider
from aca.master import OpenRouterMasterClassifier
from aca.runtime import ToolLoopRuntime
from aca.services import ConversationService, ConversationTurnRequest
from aca.storage import initialize_storage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Supports +, -, *, /, ^, (), %.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate."}
                },
                "required": ["expression"],
            },
        },
    }
]


def calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/().^% ")
    if not all(char in allowed_chars for char in expression):
        return json.dumps({"error": "Invalid characters in expression"})

    try:
        result = eval(expression.replace("^", "**"), {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})

    return json.dumps({"expression": expression, "result": result})


def main() -> None:
    load_dotenv()
    settings = get_settings()
    storage = initialize_storage()
    provider = OpenRouterProvider(title="ACA-Conversation-Service-Demo")
    service = ConversationService(
        session_factory=storage.session_factory,
        runtime=ToolLoopRuntime(
            provider=provider,
            tool_registry={"calculator": calculator},
        ),
        classifier=OpenRouterMasterClassifier(title="ACA-Master-Classifier-Demo"),
    )

    result = service.handle_turn(
        ConversationTurnRequest(
            model=settings.default_openrouter_model,
            classification_model=settings.default_classification_model,
            user_input="Calculate 18 * 7 using the calculator tool, then tell me the result.",
            tools=TOOLS,
        )
    )

    print(f"conversation_id: {result.conversation_id}")
    print(f"intent: {result.intent.value}")
    print(f"task_id: {result.task_id}")
    print(f"final_answer: {result.run_result.final_answer}")


if __name__ == "__main__":
    main()
