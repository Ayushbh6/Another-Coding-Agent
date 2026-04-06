from __future__ import annotations

from .types import HistoryMode, Message


def project_messages_for_carryover(
    messages: list[Message],
    mode: HistoryMode,
) -> list[Message]:
    if mode is HistoryMode.FULL:
        return [_copy_message(message) for message in messages]

    projected: list[Message] = []

    for message in messages:
        if message.role == "tool":
            if mode is HistoryMode.COMPACT_TOOLS and projected:
                previous = projected[-1]
                if previous.role == "assistant":
                    summary = _format_tool_summary(message)
                    previous.content = _append_summary(previous.content, summary)
            continue

        if message.role == "assistant" and message.tool_calls:
            if mode is HistoryMode.DIALOGUE_ONLY:
                if message.content:
                    projected.append(
                        Message(
                            role="assistant",
                            content=message.content,
                            reasoning=message.reasoning,
                            reasoning_details=list(message.reasoning_details),
                            metadata=dict(message.metadata),
                        )
                    )
                continue

            summary = _summarize_tool_calls(message)
            content = _append_summary(message.content, summary)
            projected.append(
                Message(
                    role="assistant",
                    content=content,
                    reasoning=message.reasoning,
                    reasoning_details=list(message.reasoning_details),
                    metadata=dict(message.metadata),
                )
            )
            continue

        projected.append(_copy_message(message))

    return projected


def _copy_message(message: Message) -> Message:
    return Message(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=list(message.tool_calls),
        reasoning=message.reasoning,
        reasoning_details=list(message.reasoning_details),
        metadata=dict(message.metadata),
    )


def _append_summary(content: str | None, summary: str) -> str:
    if not content:
        return summary
    return f"{content}\n\n{summary}"


def _summarize_tool_calls(message: Message) -> str:
    names = ", ".join(tool_call.name for tool_call in message.tool_calls) or "unknown tools"
    return f"[Prior tool activity omitted from carryover. Tools used: {names}]"


def _format_tool_summary(message: Message) -> str:
    tool_name = message.metadata.get("tool_name") or "tool"
    return f"[Prior {tool_name} result omitted from carryover.]"
