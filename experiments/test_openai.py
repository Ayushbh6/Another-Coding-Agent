from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
WEB_SEARCH_TOOL = {"type": "web_search"}


def create_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def extract_sources(response: Any) -> list[str]:
    sources: list[str] = []
    seen_urls: set[str] = set()

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue

        for content in getattr(item, "content", []) or []:
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue

                url = getattr(annotation, "url", None)
                title = getattr(annotation, "title", None)
                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                sources.append(f"{title} ({url})" if title else url)

    return sources


def chat_loop(model: str) -> None:
    client = create_client()
    previous_response_id = None

    print(f"OpenAI Responses CLI started with {model}.")
    print("Commands: /exit, /quit, /reset\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/exit", "/quit"}:
            break
        if lowered == "/reset":
            previous_response_id = None
            print("assistant> conversation reset\n")
            continue

        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": DEFAULT_REASONING_EFFORT},
                input=user_text,
                previous_response_id=previous_response_id,
                tools=[WEB_SEARCH_TOOL],
                tool_choice="auto",
            )
        except Exception as exc:
            print(f"error> {exc}\n")
            continue

        answer = (getattr(response, "output_text", "") or "").strip()
        print(f"assistant> {answer}\n")

        sources = extract_sources(response)
        if sources:
            print("sources>")
            for source in sources:
                print(f"- {source}")
            print()

        previous_response_id = response.id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OpenAI chat with web search.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model to use for chat.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    chat_loop(args.model)


if __name__ == "__main__":
    main()