from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import select

from aca.agent import AgentRunResult
from aca.approval import AllowAllApprovalPolicy, DenyAllApprovalPolicy
from aca.config import Settings
from aca.contracts import MasterAnalyzeBrief, WorkerTaskSpec
from aca.core_types import MasterClassification, TurnIntent
from aca.llm.providers.base import LLMProvider
from aca.llm.types import Message, ProviderEvent, ProviderRequest, RunResult, ToolCall, UsageStats
from aca.prompts import HELPFUL_ASSISTANT_PROMPT
from aca.services import TriageOrchestrator
from aca.storage import initialize_storage
from aca.storage.models import Action, AgentMessage, Conversation, ConversationMessage, Task


class FakeMasterClassifier:
    def classify(self, *, model: str, messages: list[Message], user_input: str) -> MasterClassification:
        lowered = user_input.lower()
        if "hello" in lowered:
            return MasterClassification(
                intent=TurnIntent.CHAT,
                task_title="",
                task_description="",
                reasoning_summary="Simple chat turn.",
            )
        if "analyze" in lowered or "repo" in lowered:
            return MasterClassification(
                intent=TurnIntent.ANALYZE,
                task_title="Analyze the repo",
                task_description="Inspect the repo and answer questions.",
                reasoning_summary="This is a read-only analysis request.",
            )
        return MasterClassification(
            intent=TurnIntent.IMPLEMENT,
            task_title="Implement the change",
            task_description="Plan, challenge, and execute the requested implementation.",
            reasoning_summary="This requires making code or file changes.",
        )


class FakeTriageProvider(LLMProvider):
    provider_name = "fake-triage"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker_structured_calls: list[float] = []

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        system_prompt = next((str(message.content) for message in request.messages if message.role == "system"), "")
        user_input = next((str(message.content) for message in reversed(request.messages) if message.role == "user"), "")
        tool_messages = [message for message in request.messages if message.role == "tool"]
        response_name = None
        if request.response_format:
            response_name = request.response_format.get("json_schema", {}).get("name")

        if system_prompt == HELPFUL_ASSISTANT_PROMPT:
            yield from self._text_response(request, "Simple chat reply.", response_id="chat-1")
            return

        if "[ANALYSIS_LEAD]" in system_prompt:
            payload = {
                "task_title": "Analyze repo",
                "questions_to_answer": ["What does the app currently do?"],
                "worker_brief": "Inspect src/app.py and summarize the current behavior.",
                "expected_answer_shape": ["Current behavior", "Relevant file"],
            }
            yield from self._json_response(request, payload, response_id="master-analyze-1")
            return

        if "[MASTER_PLANNER]" in system_prompt:
            payload = self._plan_payload(user_input)
            yield from self._json_response(request, payload, response_id="master-plan-1")
            return

        if "[CHALLENGER]" in system_prompt:
            payload = {
                "summary": "Looks mostly fine.",
                "risks": ["Check the existing file before editing."],
                "missing_checks": ["Verify the created files."],
                "bad_assumptions": [],
                "recommended_plan_changes": ["Keep edits minimal."],
            }
            yield from self._json_response(request, payload, response_id="challenger-1")
            return

        if "[MASTER_ARCHITECT_FINAL]" in system_prompt:
            payload = self._plan_payload(user_input)
            yield from self._json_response(request, payload, response_id="master-plan-2")
            return

        if "[MASTER_COMMUNICATOR]" in system_prompt:
            text = "Final summary: " + ("worker failed." if '"status": "failed"' in user_input else "worker completed.")
            yield from self._text_response(request, text, response_id="master-summary-1")
            return

        if "[WORKER_ENGINEER]" in system_prompt:
            if request.response_format is None:
                if tool_messages:
                    yield from self._text_response(request, "Tool work completed.", response_id="worker-tool-phase")
                    return

                if "Allowed mutation: False" in user_input:
                    tool_call = ToolCall(id="tool-read-1", name="read_file", arguments='{"path":"src/app.py"}')
                    yield from self._tool_call_response(request, tool_call, response_id="worker-read-1")
                    return

                if "parallel-a" in user_input:
                    tool_call = ToolCall(id="tool-write-a", name="write_file", arguments='{"path":"parallel_a.txt","content":"A"}')
                    yield from self._tool_call_response(request, tool_call, response_id="worker-write-a")
                    return

                if "parallel-b" in user_input:
                    tool_call = ToolCall(id="tool-write-b", name="write_file", arguments='{"path":"parallel_b.txt","content":"B"}')
                    yield from self._tool_call_response(request, tool_call, response_id="worker-write-b")
                    return

                tool_call = ToolCall(id="tool-write-1", name="write_file", arguments='{"path":"notes.txt","content":"done"}')
                yield from self._tool_call_response(request, tool_call, response_id="worker-write-1")
                return

            if "parallel-a" in user_input or "parallel-b" in user_input:
                with self.lock:
                    self.worker_structured_calls.append(time.perf_counter())
                time.sleep(0.3)

            failed = any("Approval denied" in (message.content or "") for message in tool_messages)
            payload = {
                "status": "failed" if failed else "completed",
                "summary": "Worker blocked by approval." if failed else "Worker finished the step.",
                "changed_files": [] if failed else self._changed_files_from_tool_messages(tool_messages),
                "commands_run": [],
                "checks": ["Verified expected file output."] if not failed else [],
                "open_issues": ["Approval denied for write_file."] if failed else [],
            }
            yield from self._json_response(request, payload, response_id="worker-structured-1")
            return

        raise AssertionError(f"Unhandled request for system prompt: {system_prompt!r}")

    def _plan_payload(self, user_input: str) -> dict[str, object]:
        lowered = user_input.lower()
        parallel_groups = []
        if "parallel" in lowered:
            parallel_groups = [
                {
                    "group_id": "parallel-1",
                    "steps": [
                        {
                            "step_id": "parallel-a",
                            "title": "Write file A",
                            "instructions": "Create parallel_a.txt",
                            "allowed_mutation": True,
                            "acceptance_checks": ["parallel_a.txt exists"],
                        },
                        {
                            "step_id": "parallel-b",
                            "title": "Write file B",
                            "instructions": "Create parallel_b.txt",
                            "allowed_mutation": True,
                            "acceptance_checks": ["parallel_b.txt exists"],
                        },
                    ],
                }
            ]

        return {
            "task_title": "Implement the requested change",
            "goal": "Confirm the triage workflow works.",
            "todo": ["Plan", "Review", "Execute"],
            "sequential_steps": [
                {
                    "step_id": "sequential-1",
                    "title": "Write notes",
                    "instructions": "Create notes.txt for the change.",
                    "allowed_mutation": True,
                    "acceptance_checks": ["notes.txt exists"],
                }
            ],
            "parallel_step_groups": parallel_groups,
            "acceptance_criteria": ["Files are created as requested."],
            "worker_global_instructions": "Inspect the workspace before editing and keep changes minimal.",
        }

    def _changed_files_from_tool_messages(self, tool_messages: list[Message]) -> list[str]:
        changed: list[str] = []
        for message in tool_messages:
            if not message.content:
                continue
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            path = payload.get("path")
            if path:
                changed.append(str(path))
        return changed

    def _json_response(self, request: ProviderRequest, payload: dict[str, object], *, response_id: str) -> Iterator[ProviderEvent]:
        text = json.dumps(payload)
        yield ProviderEvent(type="reasoning.delta", delta="thinking ")
        midpoint = max(1, len(text) // 2)
        yield ProviderEvent(type="text.delta", delta=text[:midpoint])
        yield ProviderEvent(type="text.delta", delta=text[midpoint:])
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content=text, reasoning="thinking "),
                reasoning="thinking ",
                reasoning_details=[],
                text=text,
                structured_output=payload if request.response_format else None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=10, output_tokens=10, total_tokens=20),
                response_id=response_id,
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )

    def _text_response(self, request: ProviderRequest, text: str, *, response_id: str) -> Iterator[ProviderEvent]:
        yield ProviderEvent(type="reasoning.delta", delta="thinking ")
        yield ProviderEvent(type="text.delta", delta=text)
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(role="assistant", content=text, reasoning="thinking "),
                reasoning="thinking ",
                reasoning_details=[],
                text=text,
                structured_output=None,
                tool_calls=[],
                finish_reason="stop",
                usage=UsageStats(input_tokens=8, output_tokens=6, total_tokens=14),
                response_id=response_id,
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )

    def _tool_call_response(self, request: ProviderRequest, tool_call: ToolCall, *, response_id: str) -> Iterator[ProviderEvent]:
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(
                    role="assistant",
                    content="Need to use a tool first.",
                    tool_calls=[tool_call],
                    reasoning="Use the tool first.",
                    reasoning_details=[],
                ),
                reasoning="Use the tool first.",
                reasoning_details=[],
                text="Need to use a tool first.",
                structured_output=None,
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                usage=UsageStats(input_tokens=10, output_tokens=4, total_tokens=14),
                response_id=response_id,
                latency_ms=1.0,
                provider_metadata={},
                raw_response={},
                raw_chunks=[],
            ),
        )


class TriageOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "app.py").write_text("print('hello from app')\n", encoding="utf-8")
        settings = Settings(
            sqlite_url=f"sqlite:///{self.root / 'aca.db'}",
            chroma_path=str(self.root / "chroma"),
            chroma_collection="aca-triage-test",
        )
        self.storage = initialize_storage(settings)
        self.provider = FakeTriageProvider()

        with self.storage.session_factory.begin() as session:
            session.add(
                Conversation(
                    id="conv-1",
                    user_id=None,
                    title="New chat",
                    status="active",
                    history_mode="dialogue_only",
                    active_model="fake-model",
                    thinking_enabled=True,
                    current_task_id=None,
                    message_count=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_tokens=0,
                    metadata_json={},
                )
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _orchestrator(self, approval_policy) -> TriageOrchestrator:
        return TriageOrchestrator(
            session_factory=self.storage.session_factory,
            provider=self.provider,
            classifier=FakeMasterClassifier(),
            approval_policy=approval_policy,
            workspace_root=self.root,
        )

    def test_chat_turn_uses_simple_path(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        events = list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input="hello there",
                model="fake-model",
                thinking_enabled=True,
            )
        )

        self.assertEqual(events[-1].intent, TurnIntent.CHAT)
        self.assertEqual(events[-1].final_answer, "Simple chat reply.")

    def test_analyze_turn_runs_master_worker_master_without_challenger(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        events = list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input="Analyze this repo for me.",
                model="fake-model",
                thinking_enabled=True,
            )
        )

        phase_agents = [event.agent for event in events if event.type == "phase.started"]
        self.assertIn("master", phase_agents)
        self.assertIn("worker[1]", phase_agents)
        self.assertNotIn("challenger", phase_agents)

        with self.storage.session_factory() as session:
            agent_messages = session.scalars(select(AgentMessage).order_by(AgentMessage.created_at.asc())).all()
            self.assertTrue(all(message.from_agent_id != "challenger" for message in agent_messages))
            task = session.scalars(select(Task).order_by(Task.created_at.desc())).first()
            self.assertIsNotNone(task)
            self.assertEqual(task.intent, "analyze")

    def test_implement_turn_runs_master_challenger_master_worker_master(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        events = list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input="Implement the change now.",
                model="fake-model",
                thinking_enabled=True,
            )
        )

        phase_agents = [event.agent for event in events if event.type == "phase.started"]
        self.assertEqual(
            phase_agents[:5],
            ["master", "master", "challenger", "master", "worker[sequential-1]"],
        )
        self.assertEqual(events[-1].intent, TurnIntent.IMPLEMENT)

        with self.storage.session_factory() as session:
            messages = session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == "conv-1")
                .order_by(ConversationMessage.sequence_no)
            ).all()
            self.assertEqual(
                [(message.role, message.message_kind) for message in messages],
                [("user", "user"), ("assistant", "assistant_final")],
            )
            actions = session.scalars(select(Action).order_by(Action.started_at.asc())).all()
            self.assertTrue(any(action.agent_id == "worker" and action.action_type == "tool:write_file" for action in actions))

    def test_parallel_worker_group_executes_concurrently(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        started_at = time.perf_counter()
        events = list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input="Implement the change and use parallel steps.",
                model="fake-model",
                thinking_enabled=False,
            )
        )
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.55)
        statuses = [event for event in events if event.type == "worker.status" and event.phase == "parallel_group"]
        self.assertGreaterEqual(len(statuses), 4)
        self.assertTrue((self.root / "parallel_a.txt").exists())
        self.assertTrue((self.root / "parallel_b.txt").exists())

    def test_denied_mutation_causes_failed_worker_summary(self) -> None:
        orchestrator = self._orchestrator(DenyAllApprovalPolicy())
        events = list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input="Implement the change with file edits.",
                model="fake-model",
                thinking_enabled=False,
            )
        )

        self.assertIn("worker failed", (events[-1].final_answer or "").lower())
        with self.storage.session_factory() as session:
            denied_actions = session.scalars(select(Action).where(Action.status == "denied")).all()
            self.assertGreaterEqual(len(denied_actions), 1)

    def test_extract_structured_output_rejects_wrapped_payloads(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        run_result = AgentRunResult(
            status="completed",
            final_answer=None,
            iterations=1,
            working_history=[],
            carryover_history=[],
            tool_executions=[],
            provider_runs=[
                RunResult(
                    provider="fake-triage",
                    model="fake-model",
                    assistant_message=Message(role="assistant", content='{"analysis_brief":{"task_title":"x","worker_brief":"y"}}'),
                    reasoning="",
                    reasoning_details=[],
                    text='{"analysis_brief":{"task_title":"x","worker_brief":"y"}}',
                    structured_output={"analysis_brief": {"task_title": "x", "worker_brief": "y"}},
                    tool_calls=[],
                    finish_reason="stop",
                    usage=UsageStats(),
                    response_id="wrapped-1",
                    latency_ms=1.0,
                    provider_metadata={},
                    raw_response={},
                    raw_chunks=[],
                )
            ],
            reasoning_trace=[],
        )

        with self.assertRaises(ValidationError):
            orchestrator._extract_structured_output(run_result, MasterAnalyzeBrief)

    def test_extract_worker_result_converts_missing_structured_output_to_failure(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        run_result = AgentRunResult(
            status="max_iterations",
            final_answer=None,
            iterations=24,
            working_history=[],
            carryover_history=[],
            tool_executions=[],
            provider_runs=[
                RunResult(
                    provider="fake-triage",
                    model="fake-model",
                    assistant_message=Message(role="assistant", content="I inspected a number of files."),
                    reasoning="",
                    reasoning_details=[],
                    text="I inspected a number of files.",
                    structured_output=None,
                    tool_calls=[],
                    finish_reason="stop",
                    usage=UsageStats(),
                    response_id="plain-1",
                    latency_ms=1.0,
                    provider_metadata={},
                    raw_response={},
                    raw_chunks=[],
                )
            ],
            reasoning_trace=[],
        )

        parsed = orchestrator._extract_worker_result(run_result)

        self.assertEqual(parsed.status, "failed")
        self.assertIn("tool-turn budget", parsed.summary)

    def test_worker_agents_use_thirty_turn_budget(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        captured_max_turns: list[int] = []

        class _StubAgent:
            def __init__(self, max_turns: int) -> None:
                self._max_turns = max_turns

            def stream(self, user_input, carryover_messages=None):
                yield ProviderEvent(
                    type="run.completed",
                    result=AgentRunResult(
                        status="complete",
                        final_answer=json.dumps(
                            {
                                "status": "completed",
                                "summary": "done",
                                "changed_files": [],
                                "commands_run": [],
                                "checks": [],
                                "open_issues": [],
                            }
                        ),
                        iterations=1,
                        working_history=[],
                        carryover_history=[],
                        tool_executions=[],
                        provider_runs=[],
                        reasoning_trace=[],
                    ),
                )

            def run(self, user_input, carryover_messages=None):
                return AgentRunResult(
                    status="complete",
                    final_answer=json.dumps(
                        {
                            "status": "completed",
                            "summary": "done",
                            "changed_files": [],
                            "commands_run": [],
                            "checks": [],
                            "open_issues": [],
                        }
                    ),
                    iterations=1,
                    working_history=[],
                    carryover_history=[],
                    tool_executions=[],
                    provider_runs=[],
                    reasoning_trace=[],
                )

        def _fake_create_agent(**kwargs):
            captured_max_turns.append(kwargs["max_turns"])
            return _StubAgent(kwargs["max_turns"])

        step = WorkerTaskSpec(
            step_id="s1",
            title="Inspect repo",
            instructions="Inspect the repo.",
            allowed_mutation=False,
            acceptance_checks=["done"],
        )

        with patch("aca.services.triage.create_agent", side_effect=_fake_create_agent):
            list(
                orchestrator._run_worker_step_stream(
                    step=step,
                    model="fake-model",
                    thinking_enabled=False,
                    carryover_messages=[],
                    task_id="task-1",
                    worker_label="worker[1]",
                )
            )
            orchestrator._run_worker_agent(
                step=step,
                model="fake-model",
                thinking_enabled=False,
                carryover_messages=[],
                task_id="task-1",
            )

        self.assertEqual(captured_max_turns, [30, 30])

    def test_analyze_summary_does_not_emit_plain_text_phase_completed(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())
        events = list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input="Analyze this repo for me.",
                model="fake-model",
                thinking_enabled=True,
            )
        )

        completed_phases = [event.phase for event in events if event.type == "phase.completed"]
        self.assertIn("analyze_brief", completed_phases)
        self.assertIn("worker_step", completed_phases)
        self.assertNotIn("analyze_summary", completed_phases)

    def test_worker_status_messages_are_compact(self) -> None:
        orchestrator = self._orchestrator(AllowAllApprovalPolicy())

        message = orchestrator._summarize_tool_result(
            "read_file",
            json.dumps(
                {
                    "path": "aca/agent.py",
                    "content": "40: def x\n41: pass\n",
                    "start_line": 40,
                    "end_line": 41,
                    "truncated": True,
                }
            ),
        )

        self.assertEqual(message, "read aca/agent.py lines 40-41 (truncated)")
