from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aca import create_agent
from aca.config import get_settings
from aca.llm.providers import OpenRouterProvider
from aca.llm.types import ProviderEvent


class FinalAns(BaseModel):
    final_ans: str
    short_reason: str
    confidence_score: float


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
            "description": "Analyze text and return statistics like word count, character count, and case conversions.",
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
        result = eval(expression.replace("^", "**"), {"__builtins__": {}}, {})  # noqa: S307
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
            "char_count_no_spaces": len(text.replace(" ", "")),
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "words_list": words,
        }
    )


TOOL_REGISTRY = {
    "calculator": calculator,
    "text_analyzer": text_analyzer,
}


TEST_QUERIES = [
    "What is 345 multiplied by 89, then add 1234 to the result?",
    "Calculate (500 - 125) * 8, then analyze the text 'The result is X' where X is your calculated value. Use both tools before answering.",
    "I have a rectangle with length 47 and width 23. Calculate the area and perimeter, then analyze the word 'rectangle'. Use the available tools.",
]


def build_agent(model: str) -> object:
    provider = OpenRouterProvider(title="ACA-Create-Agent-Experiment")

    return create_agent(
        name="Tool Test Agent",
        model=model,
        instructions=(
            "You are a tool-using test agent. "
            "You must use the provided tools when the user asks for calculations or text analysis. "
            "For this experiment, do not solve math or text-analysis tasks from memory. "
            "Call calculator first for numeric work, then call text_analyzer when text analysis is requested. "
            "After tool results are available, return only the final structured result."
        ),
        provider=provider,
        tools=TOOLS,
        tool_registry=TOOL_REGISTRY,
        max_turns=12,
        tool_choice="auto",
        structured_output={
            "name": "final_ans",
            "strict": True,
            "schema": FinalAns.model_json_schema(),
        },
        reasoning_enabled=True,
    )


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the reusable ACA agent abstraction against OpenRouter.")
    parser.add_argument(
        "--query-index",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Pick one of the built-in tool-usage test prompts.",
    )
    parser.add_argument(
        "--model",
        default=settings.default_openrouter_model,
        help="OpenRouter model id to test.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    query = TEST_QUERIES[args.query_index - 1]

    agent = build_agent(args.model)
    print("=" * 70)
    print("CREATE_AGENT REAL API TEST")
    print("=" * 70)
    print(f"model: {args.model}")
    print(f"query_index: {args.query_index}")
    print(f"query: {query}")
    print("stream: starting")

    final_result = None
    for event in agent.stream(query):
        if event.type == "reasoning.delta":
            print(f"[thinking] {event.delta}")
            continue

        if event.type == "tool_call.delta" and event.tool_call is not None:
            print(
                f"[tool-call.delta] name={event.tool_call.name} "
                f"args_chunk={event.delta!r}"
            )
            continue

        if event.type == "tool_call.completed" and event.tool_call is not None:
            print(
                f"[tool-call.completed] name={event.tool_call.name} "
                f"args={event.tool_call.arguments}"
            )
            continue

        if event.type == "runtime.tool_result":
            tool_name = event.tool_call.name if event.tool_call is not None else "unknown"
            print(f"[tool-result] name={tool_name} ok={event.metadata.get('ok')} result={event.delta}")
            continue

        if event.type == "text.delta":
            print(f"[text.delta] {event.delta}")
            continue

        if event.type == "run.completed":
            final_result = event.result
            print("[run.completed]")

    result = final_result
    if result is None:
        raise RuntimeError("Agent stream ended without a final result.")

    print(f"status: {result.status}")
    print(f"iterations: {result.iterations}")
    print(f"reasoning_steps: {len(result.reasoning_trace)}")
    print(f"tool_executions: {len(result.tool_executions)}")

    for index, execution in enumerate(result.tool_executions, start=1):
        print(f"\nTOOL #{index}")
        print(f"name: {execution.tool_name}")
        print(f"arguments: {json.dumps(execution.arguments)}")
        print(f"ok: {execution.ok}")
        print(f"result: {execution.result}")

    if result.provider_runs:
        last_run = result.provider_runs[-1]
        print("\nLAST PROVIDER RUN")
        print(f"model: {last_run.model}")
        print(f"finish_reason: {last_run.finish_reason}")
        print(f"raw_text: {last_run.text}")
        print(f"structured_output: {json.dumps(last_run.structured_output, indent=2)}")

        if last_run.structured_output is not None:
            parsed = FinalAns.model_validate(last_run.structured_output)
            print("\nVALIDATED STRUCTURED OUTPUT")
            print(parsed.model_dump_json(indent=2))

    print("\nFINAL ANSWER")
    print(result.final_answer)


if __name__ == "__main__":
    main()
