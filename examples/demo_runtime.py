from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aca.config import get_settings
from aca.llm.providers import OpenRouterProvider
from aca.llm.types import HistoryMode
from aca.runtime import AgentRunRequest, ToolLoopRuntime

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Supports +, -, *, /, ^, (), %.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '25 * 4 + 10'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "Analyze text and return statistics like word count and case conversions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


def calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/().^% ")
    if not all(char in allowed_chars for char in expression):
        return json.dumps({"error": "Invalid characters in expression"})

    try:
        expression = expression.replace("^", "**")
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Evaluation error: {exc}"})

    return json.dumps({"expression": expression, "result": result})


def text_analyzer(text: str) -> str:
    words = text.split()
    return json.dumps(
        {
            "original": text,
            "word_count": len(words),
            "char_count": len(text),
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "words_list": words,
        }
    )


def main() -> None:
    load_dotenv()
    settings = get_settings()

    provider = OpenRouterProvider(title="ACA-Runtime-Demo")
    runtime = ToolLoopRuntime(
        provider=provider,
        tool_registry={
            "calculator": calculator,
            "text_analyzer": text_analyzer,
        },
    )

    result = runtime.run(
        AgentRunRequest(
            model=settings.default_openrouter_model,
            user_input="Calculate (500 - 125) * 8, then analyze the text 'The result is X' where X is your calculated value.",
            tools=TOOLS,
            history_mode=HistoryMode.DIALOGUE_ONLY,
            max_iterations=6,
        )
    )

    print(f"status: {result.status}")
    print(f"iterations: {result.iterations}")
    print(f"final answer: {result.final_answer}")
    print(f"tool executions: {len(result.tool_executions)}")

    if result.provider_runs:
        last_run = result.provider_runs[-1]
        print(f"last model: {last_run.model}")
        print(f"last latency_ms: {last_run.latency_ms:.2f}")
        print(f"last usage: {last_run.usage.raw}")


if __name__ == "__main__":
    main()
