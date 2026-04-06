from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from aca.agent import AgentRunResult, create_agent
from aca.approval import ApprovalPolicy
from aca.contracts import (
    ChallengerCritique,
    MasterAnalyzeBrief,
    MasterFinalPlan,
    MasterImplementationPlan,
    ParallelStepGroup,
    WorkerTaskResult,
    WorkerTaskSpec,
)
from aca.core_types import MasterClassification, TurnIntent
from aca.json_utils import parse_json_object_loose
from aca.llm.providers.base import LLMProvider
from aca.llm.types import HistoryMode, Message, ProviderEvent
from aca.master import MasterClassifier
from aca.prompts import (
    HELPFUL_ASSISTANT_PROMPT,
    MASTER_ANALYZE_BRIEF_PROMPT,
    MASTER_FINAL_PLAN_PROMPT,
    MASTER_FINAL_SUMMARY_PROMPT,
    MASTER_IMPLEMENTATION_PLAN_PROMPT,
    WORKER_EXECUTION_PROMPT,
    CHALLENGER_CRITIQUE_PROMPT,
)
from aca.runtime import ToolLoopRuntime
from aca.services.chat import ConversationSummary
from aca.services.conversation import ConversationService
from aca.storage.models import Conversation, Task
from aca.workspace_tools import ToolPermissionError, WorkspaceToolContext, WorkspaceToolRegistry


@dataclass(slots=True)
class OrchestratedStreamEvent:
    type: str
    agent: str | None = None
    phase: str | None = None
    text: str | None = None
    thinking_text: str | None = None
    message: str | None = None
    final_answer: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None
    intent: TurnIntent | None = None
    summary: ConversationSummary | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class TriageOrchestrator(ConversationService):
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        provider: LLMProvider,
        classifier: MasterClassifier,
        approval_policy: ApprovalPolicy | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(session_factory=session_factory, runtime=ToolLoopRuntime(provider, {}), classifier=classifier)
        self._provider = provider
        self._approval_policy = approval_policy
        self._workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def stream_turn(
        self,
        *,
        conversation_id: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
    ) -> Iterator[OrchestratedStreamEvent]:
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")

            full_history = self._load_full_history(session, conversation.id)
            carryover_messages = full_history
            visible_user_count = session.scalar(
                select(func.count()).select_from(Task).where(Task.conversation_id == conversation.id)
            )
            if conversation.message_count == 0 and not visible_user_count:
                conversation.title = user_input.strip()[:40] or "New chat"
            conversation.active_model = model
            conversation.thinking_enabled = thinking_enabled

            classification = self._classifier.classify(
                model=model,
                messages=carryover_messages,
                user_input=user_input,
            )
            self._insert_conversation_message(
                session,
                conversation=conversation,
                role="user",
                message_kind="user",
                intent=classification.intent,
                content_text=user_input,
                model_name=model,
            )
            task = None
            if classification.intent in {TurnIntent.ANALYZE, TurnIntent.IMPLEMENT}:
                task = Task(
                    id=self._new_id("task"),
                    conversation_id=conversation.id,
                    parent_task_id=None,
                    intent=classification.intent.value,
                    title=classification.task_title or ("Analyze repo request" if classification.intent is TurnIntent.ANALYZE else "Implement repo change"),
                    description=classification.task_description or classification.reasoning_summary,
                    status="in_progress",
                    phase="PLAN",
                    priority="normal",
                    assigned_to_agent_id="master",
                    created_by="master",
                    spec_json={"reasoning_summary": classification.reasoning_summary},
                    scope_json={},
                    metadata_json={},
                )
                session.add(task)
                session.flush()
                conversation.current_task_id = task.id
                self._log_action(
                    session,
                    task_id=task.id,
                    agent_id="master",
                    action_type="intent_classification",
                    target=conversation.id,
                    input_json={"user_input": user_input},
                    result_json={"intent": classification.intent.value},
                    status="success",
                )

        if classification.intent is TurnIntent.CHAT:
            yield from self._stream_chat_path(
                conversation_id=conversation_id,
                user_input=user_input,
                model=model,
                thinking_enabled=thinking_enabled,
                carryover_messages=carryover_messages,
            )
            return

        if task is None:
            raise RuntimeError("Task was not created for a non-chat intent.")

        yield OrchestratedStreamEvent(
            type="phase.started",
            agent="master",
            phase="classification",
            message=f"Intent: {classification.intent.value}",
            conversation_id=conversation_id,
            task_id=task.id,
            intent=classification.intent,
        )

        if classification.intent is TurnIntent.ANALYZE:
            final_answer = yield from self._stream_analyze_path(
                conversation_id=conversation_id,
                task_id=task.id,
                user_input=user_input,
                model=model,
                thinking_enabled=thinking_enabled,
                carryover_messages=carryover_messages,
                classification=classification,
            )
        else:
            final_answer = yield from self._stream_implement_path(
                conversation_id=conversation_id,
                task_id=task.id,
                user_input=user_input,
                model=model,
                thinking_enabled=thinking_enabled,
                carryover_messages=carryover_messages,
                classification=classification,
            )

        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared during triage execution.")
            persisted_task = session.get(Task, task.id)
            if persisted_task is None:
                raise RuntimeError("Task disappeared during triage execution.")

            self._insert_conversation_message(
                session,
                conversation=conversation,
                role="assistant",
                message_kind="assistant_final",
                intent=classification.intent,
                content_text=final_answer,
                model_name=model,
                metadata_json={"triage": True},
            )
            persisted_task.status = "completed"
            persisted_task.phase = "COMPLETE"
            persisted_task.completed_at = datetime.utcnow()
            persisted_task.spec_json = {**persisted_task.spec_json, "final_answer": final_answer}
            conversation.current_task_id = None
            self._write_checkpoint(
                session,
                task_id=task.id,
                agent_id="master",
                phase="COMPLETE",
                checkpoint_kind="triage_completed",
                summary="Triage orchestration completed.",
                state_json={"intent": classification.intent.value},
            )
            summary = self._build_summary(session, conversation)

        yield OrchestratedStreamEvent(
            type="completed",
            agent="master",
            phase="complete",
            final_answer=final_answer,
            conversation_id=conversation_id,
            task_id=task.id,
            intent=classification.intent,
            summary=summary,
        )

    def _stream_chat_path(
        self,
        *,
        conversation_id: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
    ) -> Iterator[OrchestratedStreamEvent]:
        runtime = ToolLoopRuntime(self._provider, {})
        final_run: AgentRunResult | None = None
        yield OrchestratedStreamEvent(type="phase.started", agent="assistant", phase="chat", conversation_id=conversation_id)
        for event in runtime.stream(
            request=self._build_runtime_request(
                model=model,
                user_input=user_input,
                carryover_messages=carryover_messages,
                thinking_enabled=thinking_enabled,
            )
        ):
            if event.type == "reasoning.delta":
                yield OrchestratedStreamEvent(type="reasoning.delta", agent="assistant", phase="chat", thinking_text=event.delta)
            elif event.type == "text.delta":
                yield OrchestratedStreamEvent(type="text.delta", agent="assistant", phase="chat", text=event.delta)
            elif event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                final_run = event.result

        if final_run is None:
            raise RuntimeError("Chat run did not produce a final result.")

        final_answer = final_run.final_answer or ""
        with self._session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared during chat execution.")
            self._insert_conversation_message(
                session,
                conversation=conversation,
                role="assistant",
                message_kind="assistant_final",
                content_text=final_answer,
                model_name=model,
            )
            summary = self._build_summary(session, conversation)
        yield OrchestratedStreamEvent(
            type="completed",
            agent="assistant",
            phase="chat",
            final_answer=final_answer,
            conversation_id=conversation_id,
            summary=summary,
            intent=TurnIntent.CHAT,
        )

    def _stream_analyze_path(
        self,
        *,
        conversation_id: str,
        task_id: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        classification: MasterClassification,
    ) -> Iterator[str | OrchestratedStreamEvent]:
        master_brief_run = yield from self._run_agent_phase(
            agent_name="master",
            phase="analyze_brief",
            user_input=(
                f"User request:\n{user_input}\n\n"
                "Return the structured analysis brief for the worker."
            ),
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            instructions=MASTER_ANALYZE_BRIEF_PROMPT,
            structured_output={
                "name": "master_analyze_brief",
                "strict": True,
                "schema": MasterAnalyzeBrief.model_json_schema(),
            },
            task_id=task_id,
        )
        brief = self._extract_structured_output(master_brief_run, MasterAnalyzeBrief)

        self._persist_agent_message(task_id, "master", "worker", "assignment", "analyze_brief", brief.model_dump_json())

        worker_step = WorkerTaskSpec(
            step_id="analyze-1",
            title=brief.task_title,
            instructions=brief.worker_brief,
            allowed_mutation=False,
            acceptance_checks=brief.expected_answer_shape,
        )
        worker_result_run = yield from self._run_worker_step_stream(
            step=worker_step,
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            task_id=task_id,
            worker_label="worker[1]",
        )
        worker_result = self._extract_worker_result(worker_result_run)
        self._persist_agent_message(task_id, "worker", "master", "completion", "worker_result", worker_result.model_dump_json())

        summary_input = (
            f"User request:\n{user_input}\n\n"
            f"Analysis brief:\n{brief.model_dump_json(indent=2)}\n\n"
            f"Worker result:\n{worker_result.model_dump_json(indent=2)}\n\n"
            "Write the final user-facing analysis answer."
        )
        final_run = yield from self._run_agent_phase(
            agent_name="master",
            phase="analyze_summary",
            user_input=summary_input,
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            instructions=MASTER_FINAL_SUMMARY_PROMPT,
            structured_output=None,
            task_id=task_id,
        )
        with self._session_factory.begin() as session:
            task = session.get(Task, task_id)
            if task is not None:
                task.phase = "REVIEW"
                task.spec_json = {
                    **task.spec_json,
                    "brief": brief.model_dump(),
                    "worker_result": worker_result.model_dump(),
                }
        return final_run.final_answer or ""

    def _stream_implement_path(
        self,
        *,
        conversation_id: str,
        task_id: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        classification: MasterClassification,
    ) -> Iterator[str | OrchestratedStreamEvent]:
        draft_run = yield from self._run_agent_phase(
            agent_name="master",
            phase="draft_plan",
            user_input=f"User request:\n{user_input}\n\nReturn the structured implementation plan.",
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            instructions=MASTER_IMPLEMENTATION_PLAN_PROMPT,
            structured_output={
                "name": "master_implementation_plan",
                "strict": True,
                "schema": MasterImplementationPlan.model_json_schema(),
            },
            task_id=task_id,
        )
        draft_plan = self._extract_structured_output(draft_run, MasterImplementationPlan)
        self._persist_agent_message(task_id, "master", "challenger", "decision", "draft_plan", draft_plan.model_dump_json())
        with self._session_factory.begin() as session:
            task = session.get(Task, task_id)
            if task is not None:
                task.phase = "CHALLENGE"
                task.spec_json = {**task.spec_json, "draft_plan": draft_plan.model_dump()}

        critique_run = yield from self._run_agent_phase(
            agent_name="challenger",
            phase="critique",
            user_input=(
                f"User request:\n{user_input}\n\n"
                f"Draft plan:\n{draft_plan.model_dump_json(indent=2)}\n\n"
                "Return the structured critique."
            ),
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            instructions=CHALLENGER_CRITIQUE_PROMPT,
            structured_output={
                "name": "challenger_critique",
                "strict": True,
                "schema": ChallengerCritique.model_json_schema(),
            },
            task_id=task_id,
        )
        critique = self._extract_structured_output(critique_run, ChallengerCritique)
        self._persist_agent_message(task_id, "challenger", "master", "decision", "critique", critique.model_dump_json())

        final_plan_run = yield from self._run_agent_phase(
            agent_name="master",
            phase="final_plan",
            user_input=(
                f"User request:\n{user_input}\n\n"
                f"Draft plan:\n{draft_plan.model_dump_json(indent=2)}\n\n"
                f"Challenger critique:\n{critique.model_dump_json(indent=2)}\n\n"
                "Return the final structured implementation plan."
            ),
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            instructions=MASTER_FINAL_PLAN_PROMPT,
            structured_output={
                "name": "master_final_plan",
                "strict": True,
                "schema": MasterFinalPlan.model_json_schema(),
            },
            task_id=task_id,
        )
        final_plan = self._extract_structured_output(final_plan_run, MasterFinalPlan)
        self._persist_agent_message(task_id, "master", "worker", "assignment", "final_plan", final_plan.model_dump_json())

        worker_results: list[dict[str, Any]] = []
        with self._session_factory.begin() as session:
            task = session.get(Task, task_id)
            if task is not None:
                task.phase = "EXECUTE"
                task.spec_json = {
                    **task.spec_json,
                    "draft_plan": draft_plan.model_dump(),
                    "critique": critique.model_dump(),
                    "final_plan": final_plan.model_dump(),
                }

        for step in final_plan.sequential_steps:
            run_result = yield from self._run_worker_step_stream(
                step=step,
                model=model,
                thinking_enabled=thinking_enabled,
                carryover_messages=carryover_messages,
                task_id=task_id,
                worker_label=f"worker[{step.step_id}]",
            )
            parsed = self._extract_worker_result(run_result)
            worker_results.append({"step_id": step.step_id, **parsed.model_dump()})
            if parsed.status != "completed":
                break

        if all(result["status"] == "completed" for result in worker_results):
            for group in final_plan.parallel_step_groups:
                group_results = yield from self._run_parallel_group(
                    group=group,
                    model=model,
                    thinking_enabled=thinking_enabled,
                    carryover_messages=carryover_messages,
                    task_id=task_id,
                )
                worker_results.extend(group_results)
                if any(result["status"] != "completed" for result in group_results):
                    break

        summary_input = (
            f"User request:\n{user_input}\n\n"
            f"Final plan:\n{final_plan.model_dump_json(indent=2)}\n\n"
            f"Worker results:\n{json.dumps(worker_results, indent=2)}\n\n"
            "Write the final user-facing implementation summary. If there are failures, explain them clearly."
        )
        final_run = yield from self._run_agent_phase(
            agent_name="master",
            phase="final_summary",
            user_input=summary_input,
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            instructions=MASTER_FINAL_SUMMARY_PROMPT,
            structured_output=None,
            task_id=task_id,
        )
        with self._session_factory.begin() as session:
            task = session.get(Task, task_id)
            if task is not None:
                task.phase = "REVIEW"
                task.spec_json = {**task.spec_json, "worker_results": worker_results}
        return final_run.final_answer or ""

    def _run_parallel_group(
        self,
        *,
        group: ParallelStepGroup,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        task_id: str,
    ) -> Iterator[list[dict[str, Any]] | OrchestratedStreamEvent]:
        yield OrchestratedStreamEvent(
            type="phase.started",
            agent="worker",
            phase="parallel_group",
            task_id=task_id,
            message=f"Running parallel group {group.group_id} with {len(group.steps)} steps.",
            payload={"group_id": group.group_id},
        )
        for index, step in enumerate(group.steps, start=1):
            yield OrchestratedStreamEvent(
                type="worker.status",
                agent=f"worker[{index}]",
                phase="parallel_group",
                task_id=task_id,
                message=f"Queued {step.title}",
                payload={"step_id": step.step_id},
            )

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(group.steps))) as executor:
            future_map = {
                executor.submit(
                    self._run_worker_step_collect,
                    step,
                    model,
                    thinking_enabled,
                    carryover_messages,
                    task_id,
                ): step
                for step in group.steps
            }
            failed = False
            for future in as_completed(future_map):
                step = future_map[future]
                try:
                    run_result = future.result()
                    parsed = self._extract_worker_result(run_result)
                    payload = {"step_id": step.step_id, **parsed.model_dump()}
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "step_id": step.step_id,
                        "status": "failed",
                        "summary": str(exc),
                        "changed_files": [],
                        "commands_run": [],
                        "checks": [],
                        "open_issues": [str(exc)],
                    }
                results.append(payload)
                yield OrchestratedStreamEvent(
                    type="worker.status",
                    agent=f"worker[{step.step_id}]",
                    phase="parallel_group",
                    task_id=task_id,
                    message=payload["summary"],
                    payload=payload,
                )
                if payload["status"] != "completed":
                    failed = True
                    for pending in future_map:
                        if not pending.done():
                            pending.cancel()
                    break
            if failed:
                executor.shutdown(wait=False, cancel_futures=True)
        return results

    def _run_worker_step_collect(
        self,
        step: WorkerTaskSpec,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        task_id: str,
    ) -> AgentRunResult:
        return self._run_worker_agent(
            step=step,
            model=model,
            thinking_enabled=thinking_enabled,
            carryover_messages=carryover_messages,
            task_id=task_id,
        )

    def _run_worker_step_stream(
        self,
        *,
        step: WorkerTaskSpec,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        task_id: str,
        worker_label: str,
    ) -> Iterator[AgentRunResult | OrchestratedStreamEvent]:
        context = self._build_workspace_context(task_id=task_id)
        registry = WorkspaceToolRegistry(context)
        worker = create_agent(
            name="Worker",
            model=model,
            instructions=WORKER_EXECUTION_PROMPT,
            provider=self._provider,
            tools=registry.schemas(allow_mutation=step.allowed_mutation),
            tool_registry=registry.handlers(allow_mutation=step.allowed_mutation),
            max_turns=30,
            structured_output={
                "name": "worker_task_result",
                "strict": True,
                "schema": WorkerTaskResult.model_json_schema(),
            },
            reasoning_enabled=thinking_enabled,
            history_mode=HistoryMode.DIALOGUE_ONLY,
        )
        user_input = (
            f"Step title: {step.title}\n"
            f"Step id: {step.step_id}\n"
            f"Allowed mutation: {step.allowed_mutation}\n"
            f"Instructions:\n{step.instructions}\n\n"
            f"Acceptance checks:\n{json.dumps(step.acceptance_checks, indent=2)}\n\n"
            "Execute only this step and return the structured result."
        )
        yield OrchestratedStreamEvent(type="phase.started", agent=worker_label, phase="worker_step", task_id=task_id, message=step.title)
        final_result: AgentRunResult | None = None
        try:
            for event in worker.stream(user_input, carryover_messages=carryover_messages):
                if event.type == "runtime.tool_result":
                    tool_name = event.tool_call.name if event.tool_call is not None else "unknown"
                    yield OrchestratedStreamEvent(
                        type="worker.status",
                        agent=worker_label,
                        phase="worker_step",
                        task_id=task_id,
                        message=self._summarize_tool_result(tool_name, event.delta),
                    )
                elif event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                    final_result = event.result
        except ToolPermissionError as exc:
            final_result = AgentRunResult(
                status="failed",
                final_answer=json.dumps(
                    WorkerTaskResult(
                        status="failed",
                        summary=str(exc),
                        changed_files=[],
                        commands_run=[],
                        checks=[],
                        open_issues=[str(exc)],
                    ).model_dump()
                ),
                iterations=0,
                working_history=[],
                carryover_history=[],
                tool_executions=[],
                provider_runs=[],
                reasoning_trace=[],
            )
        if final_result is None:
            raise RuntimeError("Worker stream did not produce a final result.")
        worker_result = self._extract_worker_result(final_result)
        yield OrchestratedStreamEvent(
            type="phase.completed",
            agent=worker_label,
            phase="worker_step",
            task_id=task_id,
            message=worker_result.summary,
            payload=worker_result.model_dump(),
        )
        return final_result

    def _summarize_tool_result(self, tool_name: str, raw_result: str | None) -> str:
        payload: dict[str, Any] | None = None
        if raw_result:
            try:
                parsed = json.loads(raw_result)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = None

        if tool_name == "list_files" and payload is not None:
            count = payload.get("count", 0)
            files = payload.get("files", [])
            scope = payload.get("path", ".")
            sample = ", ".join(str(item) for item in files[:3]) if isinstance(files, list) else ""
            return f"listed {count} files under {scope}" + (f": {sample}" if sample else "")

        if tool_name == "read_file" and payload is not None:
            path = payload.get("path", "file")
            start_line = payload.get("start_line")
            end_line = payload.get("end_line")
            truncated = payload.get("truncated", False)
            if isinstance(start_line, int) and isinstance(end_line, int):
                slice_label = f"lines {start_line}-{end_line}"
            else:
                content = str(payload.get("content", ""))
                slice_label = f"{len(content.splitlines())} lines"
            return f"read {path} {slice_label}" + (" (truncated)" if truncated else "")

        if tool_name == "search_code" and payload is not None:
            count = payload.get("count", 0)
            matches = payload.get("matches", [])
            sample = ""
            if isinstance(matches, list) and matches:
                first = matches[0]
                if isinstance(first, dict) and first.get("path"):
                    sample = str(first["path"])
                    if first.get("line"):
                        sample += f":{first['line']}"
            return f"found {count} matches" + (f" starting at {sample}" if sample else "")

        if tool_name == "run_command" and payload is not None:
            command = payload.get("command", "command")
            exit_code = payload.get("exit_code", "?")
            return f"ran `{command}` (exit {exit_code})"

        if tool_name == "write_file" and payload is not None:
            path = payload.get("path", "file")
            return f"wrote {path}"

        if tool_name == "apply_patch" and payload is not None:
            path = payload.get("path", "file")
            replacements = payload.get("replacements", "?")
            return f"patched {path} ({replacements} replacements)"

        compact = (raw_result or "").strip()
        if len(compact) > 160:
            compact = compact[:157] + "..."
        return f"{tool_name}: {compact}" if compact else tool_name

    def _run_worker_agent(
        self,
        *,
        step: WorkerTaskSpec,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        task_id: str,
    ) -> AgentRunResult:
        context = self._build_workspace_context(task_id=task_id)
        registry = WorkspaceToolRegistry(context)
        worker = create_agent(
            name="Worker",
            model=model,
            instructions=WORKER_EXECUTION_PROMPT,
            provider=self._provider,
            tools=registry.schemas(allow_mutation=step.allowed_mutation),
            tool_registry=registry.handlers(allow_mutation=step.allowed_mutation),
            max_turns=30,
            structured_output={
                "name": "worker_task_result",
                "strict": True,
                "schema": WorkerTaskResult.model_json_schema(),
            },
            reasoning_enabled=thinking_enabled,
            history_mode=HistoryMode.DIALOGUE_ONLY,
        )
        user_input = (
            f"Step title: {step.title}\n"
            f"Step id: {step.step_id}\n"
            f"Allowed mutation: {step.allowed_mutation}\n"
            f"Instructions:\n{step.instructions}\n\n"
            f"Acceptance checks:\n{json.dumps(step.acceptance_checks, indent=2)}\n\n"
            "Execute only this step and return the structured result."
        )
        try:
            return worker.run(user_input, carryover_messages=carryover_messages)
        except ToolPermissionError as exc:
            return AgentRunResult(
                status="failed",
                final_answer=json.dumps(
                    WorkerTaskResult(
                        status="failed",
                        summary=str(exc),
                        changed_files=[],
                        commands_run=[],
                        checks=[],
                        open_issues=[str(exc)],
                    ).model_dump()
                ),
                iterations=0,
                working_history=[],
                carryover_history=[],
                tool_executions=[],
                provider_runs=[],
                reasoning_trace=[],
            )

    def _extract_worker_result(self, run_result: AgentRunResult) -> WorkerTaskResult:
        try:
            return self._extract_structured_output(run_result, WorkerTaskResult)
        except ValueError:
            last_text = ""
            if run_result.provider_runs:
                last_text = run_result.provider_runs[-1].text or ""
            summary = "Worker did not return the required structured result."
            if run_result.status == "max_iterations":
                summary = "Worker exceeded its tool-turn budget before returning the required structured result."
            elif run_result.status != "complete":
                summary = f"Worker ended with status `{run_result.status}` before returning the required structured result."
            open_issues = [summary]
            if last_text.strip():
                compact = last_text.strip()
                if len(compact) > 500:
                    compact = compact[:497] + "..."
                open_issues.append(f"Last assistant text: {compact}")
            return WorkerTaskResult(
                status="failed",
                summary=summary,
                changed_files=[],
                commands_run=[],
                checks=[],
                open_issues=open_issues,
            )

    def _run_agent_phase(
        self,
        *,
        agent_name: str,
        phase: str,
        user_input: str,
        model: str,
        thinking_enabled: bool,
        carryover_messages: list[Message],
        instructions: str,
        structured_output: dict[str, Any] | None,
        task_id: str,
    ) -> Iterator[AgentRunResult | OrchestratedStreamEvent]:
        agent = create_agent(
            name=agent_name.title(),
            model=model,
            instructions=instructions,
            provider=self._provider,
            structured_output=structured_output,
            reasoning_enabled=thinking_enabled,
            history_mode=HistoryMode.DIALOGUE_ONLY,
        )
        yield OrchestratedStreamEvent(type="phase.started", agent=agent_name, phase=phase, task_id=task_id)
        final_result: AgentRunResult | None = None
        for event in agent.stream(user_input, carryover_messages=carryover_messages):
            if structured_output is None and event.type == "reasoning.delta":
                yield OrchestratedStreamEvent(type="reasoning.delta", agent=agent_name, phase=phase, task_id=task_id, thinking_text=event.delta)
            elif structured_output is None and event.type == "text.delta":
                yield OrchestratedStreamEvent(type="text.delta", agent=agent_name, phase=phase, task_id=task_id, text=event.delta)
            elif event.type == "run.completed" and isinstance(event.result, AgentRunResult):
                final_result = event.result
        if final_result is None:
            raise RuntimeError(f"{agent_name} phase {phase} did not complete.")
        payload = self._final_text_payload(final_result)
        self._persist_agent_message(task_id, agent_name, agent_name, "decision", phase, payload)
        with self._session_factory.begin() as session:
            self._write_checkpoint(
                session,
                task_id=task_id,
                agent_id=agent_name,
                phase=phase.upper(),
                checkpoint_kind="phase_completed",
                summary=f"{agent_name} completed {phase}.",
                state_json={"final_text": payload},
            )
        if structured_output is not None:
            yield OrchestratedStreamEvent(
                type="phase.completed",
                agent=agent_name,
                phase=phase,
                task_id=task_id,
                message=self._phase_completion_message(final_result, structured_output),
            )
        return final_result

    def _build_runtime_request(
        self,
        *,
        model: str,
        user_input: str,
        carryover_messages: list[Message],
        thinking_enabled: bool,
    ):
        from aca.runtime import AgentRunRequest

        return AgentRunRequest(
            model=model,
            user_input=user_input,
            system_prompt=HELPFUL_ASSISTANT_PROMPT,
            carryover_messages=carryover_messages,
            tools=[],
            history_mode=HistoryMode.DIALOGUE_ONLY,
            include_reasoning=thinking_enabled,
        )

    def _build_workspace_context(self, *, task_id: str) -> WorkspaceToolContext:
        return WorkspaceToolContext(
            root=self._workspace_root,
            session_factory=self._session_factory,
            task_id=task_id,
            agent_id="worker",
            approval_policy=self._approval_policy,
        )

    def _persist_agent_message(
        self,
        task_id: str,
        from_agent_id: str,
        to_agent_id: str,
        channel: str,
        message_type: str,
        content: str,
    ) -> None:
        with self._session_factory.begin() as session:
            self._log_agent_message(
                session,
                task_id=task_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                channel=channel,
                message_type=message_type,
                content=content,
            )

    def _extract_structured_output(self, run_result: AgentRunResult, model_cls):
        if run_result.provider_runs:
            for provider_run in reversed(run_result.provider_runs):
                if provider_run.structured_output is not None:
                    return model_cls.model_validate(provider_run.structured_output)
        if run_result.final_answer:
            return model_cls.model_validate(parse_json_object_loose(run_result.final_answer))
        raise ValueError("No structured output found.")

    def _final_text_payload(self, run_result: AgentRunResult) -> str:
        if run_result.provider_runs:
            last = run_result.provider_runs[-1]
            if last.text:
                return last.text
        return run_result.final_answer or ""

    def _phase_completion_message(
        self,
        run_result: AgentRunResult,
        structured_output: dict[str, Any] | None,
    ) -> str:
        if structured_output is None:
            return "completed"

        payload: dict[str, Any] | None = None
        if run_result.provider_runs:
            for provider_run in reversed(run_result.provider_runs):
                if isinstance(provider_run.structured_output, dict):
                    payload = provider_run.structured_output
                    break

        if not payload:
            return "structured output ready"

        for key in ("task_title", "summary", "goal", "status"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if "sequential_steps" in payload and isinstance(payload["sequential_steps"], list):
            return f"plan ready with {len(payload['sequential_steps'])} sequential steps"

        return "structured output ready"

    def _build_summary(self, session: Session, conversation: Conversation) -> ConversationSummary:
        from aca.storage.models import ConversationMessage

        message_aggregates = session.execute(
            select(
                func.count(ConversationMessage.id),
                func.coalesce(func.sum(ConversationMessage.input_tokens), 0),
                func.coalesce(func.sum(ConversationMessage.output_tokens), 0),
                func.coalesce(func.sum(ConversationMessage.total_tokens), 0),
            ).where(ConversationMessage.conversation_id == conversation.id)
        ).one()

        return ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            visible_message_count=int(message_aggregates[0] or 0),
            active_model=conversation.active_model,
            thinking_enabled=conversation.thinking_enabled,
            total_input_tokens=int(message_aggregates[1] or 0),
            total_output_tokens=int(message_aggregates[2] or 0),
            total_tokens=int(message_aggregates[3] or 0),
            updated_at=conversation.updated_at,
        )
