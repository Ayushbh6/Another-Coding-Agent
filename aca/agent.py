from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from aca.llm.providers.base import LLMProvider
from aca.llm.types import ContentPart, HistoryMode, Message, ProviderEvent
from aca.runtime import AgentRunRequest, AgentRunResult, ToolHandler, ToolLoopRuntime


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    model: str
    instructions: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_registry: Mapping[str, ToolHandler] = field(default_factory=dict)
    max_turns: int = 10
    tool_choice: str | dict[str, Any] = "auto"
    parallel_tool_calls: bool | None = None
    structured_output: dict[str, Any] | None = None
    reasoning_enabled: bool = True
    reasoning: dict[str, Any] | None = None
    provider_preferences: dict[str, Any] = field(default_factory=lambda: {"require_parameters": True})
    history_mode: HistoryMode = HistoryMode.DIALOGUE_ONLY
    provider_extra_body: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        *,
        spec: AgentSpec,
        provider: LLMProvider | None = None,
        runtime: ToolLoopRuntime | None = None,
    ) -> None:
        if runtime is None and provider is None:
            raise ValueError("Either provider or runtime must be supplied.")

        self.spec = spec
        self.runtime = runtime or ToolLoopRuntime(
            provider=provider,
            tool_registry=spec.tool_registry,
        )

    def stream(
        self,
        user_input: str | list[ContentPart],
        *,
        carryover_messages: list[Message] | None = None,
        extra_instructions: str | None = None,
        max_turns: int | None = None,
        structured_output: dict[str, Any] | None = None,
    ) -> Iterator[ProviderEvent]:
        return self._stream_internal(
            user_input=user_input,
            carryover_messages=carryover_messages,
            extra_instructions=extra_instructions,
            max_turns=max_turns,
            structured_output=structured_output,
        )

    def run(
        self,
        user_input: str | list[ContentPart],
        *,
        carryover_messages: list[Message] | None = None,
        extra_instructions: str | None = None,
        max_turns: int | None = None,
        structured_output: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        final_result: AgentRunResult | None = None
        for event in self._stream_internal(
            user_input=user_input,
            carryover_messages=carryover_messages,
            extra_instructions=extra_instructions,
            max_turns=max_turns,
            structured_output=structured_output,
        ):
            if event.type == "run.completed" and event.result is not None:
                final_result = event.result

        if final_result is None:
            raise RuntimeError("Agent run completed without a final result.")

        return final_result

    def _stream_internal(
        self,
        *,
        user_input: str | list[ContentPart],
        carryover_messages: list[Message] | None,
        extra_instructions: str | None,
        max_turns: int | None,
        structured_output: dict[str, Any] | None,
    ) -> Iterator[ProviderEvent]:
        remaining_turns = max_turns or self.spec.max_turns
        resolved_structured_output = normalize_structured_output(
            self.spec.name,
            structured_output if structured_output is not None else self.spec.structured_output,
        )

        tool_phase_request = self._build_request(
            user_input=user_input,
            carryover_messages=carryover_messages,
            extra_instructions=extra_instructions,
            max_turns=remaining_turns,
            structured_output=None if self.spec.tools and resolved_structured_output is not None else resolved_structured_output,
            append_user_message=True,
            tools=list(self.spec.tools),
            tool_choice=self.spec.tool_choice,
            include_system_prompt=True,
        )

        tool_phase_result = yield from self._execute_request(tool_phase_request)
        if tool_phase_result.status != "complete":
            yield ProviderEvent(type="run.completed", result=tool_phase_result)
            return

        if not self.spec.tools or resolved_structured_output is None:
            yield ProviderEvent(type="run.completed", result=tool_phase_result)
            return

        remaining_turns = max(1, remaining_turns - tool_phase_result.iterations)
        final_phase_request = self._build_request(
            user_input=None,
            carryover_messages=tool_phase_result.working_history,
            extra_instructions=None,
            max_turns=remaining_turns,
            structured_output=resolved_structured_output,
            append_user_message=False,
            tools=[],
            tool_choice="auto",
            include_system_prompt=False,
        )
        final_phase_result = yield from self._execute_request(final_phase_request)

        yield ProviderEvent(
            type="run.completed",
            result=self._merge_results(tool_phase_result, final_phase_result),
        )

    def _build_request(
        self,
        *,
        user_input: str | list[ContentPart] | None,
        carryover_messages: list[Message] | None,
        extra_instructions: str | None,
        max_turns: int | None,
        structured_output: dict[str, Any] | None,
        append_user_message: bool,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any],
        include_system_prompt: bool,
    ) -> AgentRunRequest:
        resolved_reasoning = self.spec.reasoning if self.spec.reasoning_enabled else {"exclude": True}

        return AgentRunRequest(
            model=self.spec.model,
            user_input=user_input,
            system_prompt=self._compose_instructions(extra_instructions) if include_system_prompt else None,
            carryover_messages=list(carryover_messages or []),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=self.spec.parallel_tool_calls,
            response_format=structured_output,
            reasoning=resolved_reasoning,
            provider=dict(self.spec.provider_preferences),
            history_mode=self.spec.history_mode,
            max_iterations=max_turns or self.spec.max_turns,
            include_reasoning=self.spec.reasoning_enabled,
            provider_extra_body=dict(self.spec.provider_extra_body),
            append_user_message=append_user_message,
        )

    def _compose_instructions(self, extra_instructions: str | None) -> str:
        base = self.spec.instructions.strip()
        if not extra_instructions:
            return base
        return f"{base}\n\nAdditional run instructions:\n{extra_instructions.strip()}"

    def _execute_request(self, request: AgentRunRequest) -> Iterator[ProviderEvent]:
        final_result: AgentRunResult | None = None
        for event in self.runtime.stream(request):
            if event.type == "run.completed" and event.result is not None:
                final_result = event.result
            else:
                yield event

        if final_result is None:
            raise RuntimeError("Agent phase completed without a final result.")

        return final_result

    def _merge_results(self, initial: AgentRunResult, final: AgentRunResult) -> AgentRunResult:
        return AgentRunResult(
            status=final.status,
            final_answer=final.final_answer,
            iterations=initial.iterations + final.iterations,
            working_history=final.working_history,
            carryover_history=final.carryover_history,
            tool_executions=[*initial.tool_executions, *final.tool_executions],
            provider_runs=[*initial.provider_runs, *final.provider_runs],
            reasoning_trace=[*initial.reasoning_trace, *final.reasoning_trace],
        )


def create_agent(
    *,
    name: str,
    model: str,
    instructions: str,
    provider: LLMProvider | None = None,
    runtime: ToolLoopRuntime | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_registry: Mapping[str, ToolHandler] | None = None,
    max_turns: int = 10,
    tool_choice: str | dict[str, Any] = "auto",
    parallel_tool_calls: bool | None = None,
    structured_output: dict[str, Any] | None = None,
    reasoning_enabled: bool = True,
    reasoning: dict[str, Any] | None = None,
    provider_preferences: dict[str, Any] | None = None,
    history_mode: HistoryMode = HistoryMode.DIALOGUE_ONLY,
    provider_extra_body: dict[str, Any] | None = None,
) -> Agent:
    return Agent(
        spec=AgentSpec(
            name=name,
            model=model,
            instructions=instructions,
            tools=list(tools or []),
            tool_registry=dict(tool_registry or {}),
            max_turns=max_turns,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            structured_output=structured_output,
            reasoning_enabled=reasoning_enabled,
            reasoning=reasoning,
            provider_preferences=dict(provider_preferences or {"require_parameters": True}),
            history_mode=history_mode,
            provider_extra_body=dict(provider_extra_body or {}),
        ),
        provider=provider,
        runtime=runtime,
    )


def normalize_structured_output(agent_name: str, structured_output: dict[str, Any] | None) -> dict[str, Any] | None:
    if structured_output is None:
        return None

    if structured_output.get("type") in {"json_object", "json_schema"}:
        return structured_output

    if "schema" in structured_output or "name" in structured_output or "strict" in structured_output:
        json_schema = dict(structured_output)
        json_schema.setdefault("name", _slugify_agent_name(agent_name))
        json_schema.setdefault("strict", True)
        return {"type": "json_schema", "json_schema": json_schema}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": _slugify_agent_name(agent_name),
            "strict": True,
            "schema": structured_output,
        },
    }


def _slugify_agent_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "agent_output"
