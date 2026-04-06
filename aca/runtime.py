from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from aca.llm.history import project_messages_for_carryover
from aca.llm.providers.base import LLMProvider
from aca.llm.types import HistoryMode, Message, ProviderEvent, ProviderRequest, RunResult, ToolCall


ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class ToolExecutionRecord:
    iteration: int
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    result: str
    ok: bool


@dataclass(slots=True)
class AgentRunRequest:
    model: str
    user_input: str
    system_prompt: str | None = None
    carryover_messages: list[Message] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    history_mode: HistoryMode = HistoryMode.DIALOGUE_ONLY
    max_iterations: int = 10
    include_reasoning: bool = True
    provider_extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    status: str
    final_answer: str | None
    iterations: int
    working_history: list[Message]
    carryover_history: list[Message]
    tool_executions: list[ToolExecutionRecord]
    provider_runs: list[RunResult]
    reasoning_trace: list[str]


class ToolLoopRuntime:
    def __init__(
        self,
        provider: LLMProvider,
        tool_registry: Mapping[str, ToolHandler] | None = None,
    ) -> None:
        self._provider = provider
        self._tool_registry = dict(tool_registry or {})

    def stream(self, request: AgentRunRequest) -> Iterator[ProviderEvent]:
        working_history: list[Message] = []
        if request.system_prompt:
            working_history.append(Message(role="system", content=request.system_prompt))
        working_history.extend(request.carryover_messages)
        working_history.append(Message(role="user", content=request.user_input))

        tool_executions: list[ToolExecutionRecord] = []
        provider_runs: list[RunResult] = []
        reasoning_trace: list[str] = []

        for iteration in range(1, request.max_iterations + 1):
            provider_request = ProviderRequest(
                model=request.model,
                messages=working_history,
                tools=request.tools,
                include_reasoning=request.include_reasoning,
                extra_body=request.provider_extra_body,
            )

            run_result: RunResult | None = None
            for event in self._provider.stream_turn(provider_request):
                event.metadata = {"iteration": iteration, **event.metadata}
                if event.type == "response.completed" and isinstance(event.result, RunResult):
                    run_result = event.result
                    continue
                yield event

            if run_result is None:
                raise RuntimeError("Provider stream ended without a RunResult.")

            provider_runs.append(run_result)
            working_history.append(run_result.assistant_message)

            if run_result.reasoning:
                reasoning_trace.append(run_result.reasoning)

            if not run_result.tool_calls:
                carryover_history = project_messages_for_carryover(working_history, request.history_mode)
                yield ProviderEvent(
                    type="run.completed",
                    result=self._build_result(
                        status="complete",
                        final_answer=run_result.text or None,
                        iterations=iteration,
                        working_history=working_history,
                        carryover_history=carryover_history,
                        tool_executions=tool_executions,
                        provider_runs=provider_runs,
                        reasoning_trace=reasoning_trace,
                    ),
                )
                return

            for tool_call in run_result.tool_calls:
                execution = self._execute_tool(tool_call, iteration)
                tool_executions.append(execution)
                working_history.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=execution.result,
                        metadata={"tool_name": tool_call.name},
                    )
                )

                yield ProviderEvent(
                    type="runtime.tool_result",
                    tool_call=tool_call,
                    delta=execution.result,
                    metadata={
                        "iteration": iteration,
                        "ok": execution.ok,
                        "arguments": execution.arguments,
                    },
                )

        carryover_history = project_messages_for_carryover(working_history, request.history_mode)
        yield ProviderEvent(
            type="run.completed",
            result=self._build_result(
                status="max_iterations",
                final_answer=None,
                iterations=request.max_iterations,
                working_history=working_history,
                carryover_history=carryover_history,
                tool_executions=tool_executions,
                provider_runs=provider_runs,
                reasoning_trace=reasoning_trace,
            ),
        )

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        final_result: AgentRunResult | None = None

        for event in self.stream(request):
            if event.type == "run.completed" and event.result is not None:
                final_result = event.result

        if final_result is None:
            raise RuntimeError("Tool loop runtime ended without a final result.")

        return final_result

    def _execute_tool(self, tool_call: ToolCall, iteration: int) -> ToolExecutionRecord:
        handler = self._tool_registry.get(tool_call.name)
        arguments = self._parse_arguments(tool_call.arguments)

        if handler is None:
            return ToolExecutionRecord(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=arguments,
                result=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
                ok=False,
            )

        try:
            raw_result = handler(**arguments)
            result = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
            return ToolExecutionRecord(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=arguments,
                result=result,
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolExecutionRecord(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=arguments,
                result=json.dumps({"error": str(exc)}),
                ok=False,
            )

    def _parse_arguments(self, arguments: str) -> dict[str, Any]:
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    def _build_result(
        self,
        *,
        status: str,
        final_answer: str | None,
        iterations: int,
        working_history: list[Message],
        carryover_history: list[Message],
        tool_executions: list[ToolExecutionRecord],
        provider_runs: list[RunResult],
        reasoning_trace: list[str],
    ) -> AgentRunResult:
        return AgentRunResult(
            status=status,
            final_answer=final_answer,
            iterations=iterations,
            working_history=list(working_history),
            carryover_history=list(carryover_history),
            tool_executions=list(tool_executions),
            provider_runs=list(provider_runs),
            reasoning_trace=list(reasoning_trace),
        )
