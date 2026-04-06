from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select

from aca.approval import AllowAllApprovalPolicy
from aca.config import Settings
from aca.llm.providers.base import LLMProvider
from aca.llm.types import Message, ProviderEvent, ProviderRequest, RunResult, ToolCall, UsageStats
from aca.orchestration import NeonOrchestrator
from aca.orchestration.state import ACTOR_NEON, NeonRunState
from aca.orchestration.tools import OrchestrationToolRegistry
from aca.orchestration.prompts import ANALYZE_WORKER_PROMPT, NEON_PROMPT
from aca.storage import initialize_storage
from aca.storage.models import Action, Conversation, Task
from aca.task_workspace import TaskWorkspaceManager
from aca.workspace_tools import WorkspaceToolContext, WorkspaceToolRegistry


class FakeNeonProvider(LLMProvider):
    provider_name = "fake-neon"

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        system_prompt = next((str(message.content) for message in request.messages if message.role == "system"), "")
        user_messages = [str(message.content) for message in request.messages if message.role == "user" and message.content]
        full_user_text = "\n".join(user_messages)
        tool_messages = [message for message in request.messages if message.role == "tool"]

        if NEON_PROMPT.splitlines()[0] in system_prompt:
            yield from self._neon_response(request, full_user_text, tool_messages)
            return

        if ANALYZE_WORKER_PROMPT.splitlines()[0] in system_prompt:
            yield from self._worker_response(request, tool_messages)
            return

        raise AssertionError(f"Unhandled system prompt: {system_prompt!r}")

    def _neon_response(self, request: ProviderRequest, full_user_text: str, tool_messages: list[Message]) -> Iterator[ProviderEvent]:
        route = self._route_for_request(full_user_text)
        tool_names = [self._tool_name(message) for message in tool_messages]
        tool_payloads = [self._tool_payload(message) for message in tool_messages]
        written_artifacts = {
            payload.get("artifact_name"): payload
            for payload in tool_payloads
            if isinstance(payload, dict) and payload.get("artifact_name")
        }
        spawn_scheduled = any(
            isinstance(payload, dict) and payload.get("status") == "scheduled"
            for payload in tool_payloads
        )
        system_prompt = next((str(message.content) for message in request.messages if message.role == "system"), "")
        task_id = self._extract_task_id(system_prompt)

        if route == "chat":
            yield from self._text_response(request, "Neon chat reply.", response_id="neon-chat-1")
            return

        if route == "implement_disabled":
            yield from self._text_response(request, "Implementation mode is currently disabled. I can analyze the repo, but I will not mutate it.", response_id="neon-impl-1")
            return

        if route == "analyze_simple":
            if not tool_messages:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="neon-list-1", name="list_files", arguments='{"path":".","limit":20,"max_depth":2}'),
                    response_id="neon-list-1",
                    narration="I’m going to inspect the repo structure first.",
                )
                return
            if "task.md" not in written_artifacts:
                yield from self._tool_call_response(
                    request,
                    ToolCall(
                        id="neon-task-1",
                        name="write_task_artifact",
                        arguments=json.dumps(
                            {
                                "artifact_name": "task.md",
                                "content": (
                                    "---\n"
                                    f"task_id: {task_id}\n"
                                    "intent: analyze\n"
                                    "route: analyze_simple\n"
                                    "title: Analyze runtime flow\n"
                                    "---\n"
                                    "Inspect the repo and explain the main orchestration flow.\n"
                                ),
                            }
                        ),
                    ),
                    response_id="neon-task-1",
                    narration="This looks contained enough for a direct analysis.",
                )
                return
            if tool_names.count("read_file") < 1:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="neon-read-1", name="read_file", arguments='{"path":"aca/orchestration/orchestrator.py","start_line":1,"end_line":220}'),
                    response_id="neon-read-1",
                    narration="I have the task pinned down. Now I’m doing a focused orientation pass.",
                )
                return
            if "todo.md" not in written_artifacts:
                yield from self._tool_call_response(
                    request,
                    ToolCall(
                        id="neon-todo-1",
                        name="write_task_artifact",
                        arguments=json.dumps(
                            {
                                "artifact_name": "todo.md",
                                "content": "- inspect the orchestrator entrypoint\n- summarize the task and todo flow\n",
                            }
                        ),
                    ),
                    response_id="neon-todo-1",
                    narration="I have enough context to turn this into a concrete todo.",
                )
                return
            todo_payload = next(
                (payload for payload in reversed(tool_payloads) if isinstance(payload, dict) and payload.get("items")),
                None,
            )
            todo_items = todo_payload.get("items", []) if isinstance(todo_payload, dict) else []
            if "read_todo_state" not in tool_names:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="neon-todo-state-1", name="read_todo_state", arguments="{}"),
                    response_id="neon-todo-state-1",
                )
                return
            started_ids = {str(payload.get("todo_id")) for payload in tool_payloads if isinstance(payload, dict) and payload.get("status") == "in_progress"}
            completed_ids = {str(payload.get("todo_id")) for payload in tool_payloads if isinstance(payload, dict) and payload.get("status") == "completed"}
            if todo_items:
                first_id = str(todo_items[0]["todo_id"])
                if first_id not in started_ids and first_id not in completed_ids:
                    yield from self._tool_call_response(
                        request,
                        ToolCall(id="neon-start-1", name="start_todo_item", arguments=json.dumps({"todo_id": first_id})),
                        response_id="neon-start-1",
                    )
                    return
                if "search_code" not in tool_names:
                    yield from self._tool_call_response(
                        request,
                        ToolCall(id="neon-search-1", name="search_code", arguments='{"pattern":"class NeonOrchestrator","path":"aca"}'),
                        response_id="neon-search-1",
                    )
                    return
                if first_id not in completed_ids:
                    yield from self._tool_call_response(
                        request,
                        ToolCall(
                            id="neon-complete-1",
                            name="complete_todo_item",
                            arguments=json.dumps({"todo_id": first_id, "outcome": "Inspected the orchestrator entrypoint and stream loop."}),
                        ),
                        response_id="neon-complete-1",
                    )
                    return
            if len(todo_items) > 1:
                second_id = str(todo_items[1]["todo_id"])
                if second_id not in started_ids and second_id not in completed_ids:
                    yield from self._tool_call_response(
                        request,
                        ToolCall(id="neon-start-2", name="start_todo_item", arguments=json.dumps({"todo_id": second_id})),
                        response_id="neon-start-2",
                    )
                    return
                if tool_names.count("read_file") < 2:
                    yield from self._tool_call_response(
                        request,
                        ToolCall(id="neon-read-2", name="read_file", arguments='{"path":"aca/orchestration/tools.py","start_line":1,"end_line":220}'),
                        response_id="neon-read-2",
                    )
                    return
                if second_id not in completed_ids:
                    yield from self._tool_call_response(
                        request,
                        ToolCall(
                            id="neon-complete-2",
                            name="complete_todo_item",
                            arguments=json.dumps({"todo_id": second_id, "outcome": "Summarized the task and todo flow."}),
                        ),
                        response_id="neon-complete-2",
                    )
                    return
            yield from self._text_response(request, "Simple analysis summary.", response_id="neon-simple-final")
            return

        if not tool_messages:
            yield from self._tool_call_response(
                request,
                ToolCall(id="neon-repo-1", name="get_repo_summary", arguments="{}"),
                response_id="neon-repo-1",
                narration="I’m going to scout the repo before deciding how deep this needs to go.",
            )
            return
        if "list_files" not in tool_names:
            yield from self._tool_call_response(
                request,
                ToolCall(id="neon-list-2", name="list_files", arguments='{"path":".","limit":20,"max_depth":2}'),
                response_id="neon-list-2",
                narration="This looks broad, so I’m confirming the repo shape before I lock in the task.",
            )
            return
        if "task.md" not in written_artifacts:
            yield from self._tool_call_response(
                request,
                ToolCall(
                    id="neon-task-2",
                    name="write_task_artifact",
                    arguments=json.dumps(
                        {
                            "artifact_name": "task.md",
                            "content": (
                                "---\n"
                                f"task_id: {task_id}\n"
                                "intent: analyze\n"
                                "route: analyze_delegated\n"
                                "title: Scan the codebase\n"
                                "---\n"
                                "Scan the codebase and explain how the agent architecture works.\n"
                            ),
                        }
                    ),
                ),
                response_id="neon-task-2",
                narration="This is substantial enough for a delegated scan.",
            )
            return
        if tool_names.count("read_file") < 2:
            target = "aca/orchestration/orchestrator.py" if tool_names.count("read_file") == 0 else "aca/orchestration/tools.py"
            response_id = "neon-orient-1" if tool_names.count("read_file") == 0 else "neon-orient-2"
            yield from self._tool_call_response(
                request,
                ToolCall(id=response_id, name="read_file", arguments=json.dumps({"path": target, "start_line": 1, "end_line": 220})),
                response_id=response_id,
                narration="I’m doing the orientation pass that will shape the plan." if tool_names.count("read_file") == 0 else None,
            )
            return
        if "plan.md" not in written_artifacts:
            yield from self._tool_call_response(
                request,
                ToolCall(
                    id="neon-plan-1",
                    name="write_task_artifact",
                    arguments=json.dumps(
                        {
                            "artifact_name": "plan.md",
                            "content": "# Plan\n\n1. Inspect the orchestrator entrypoint and stream loop.\n2. Inspect shared tool gating and todo control.\n3. Synthesize the final agent architecture.\n",
                        }
                    ),
                ),
                response_id="neon-plan-1",
                narration="I have enough context to write the investigation plan.",
            )
            return
        if "todo.md" not in written_artifacts:
            yield from self._tool_call_response(
                request,
                ToolCall(
                    id="neon-todo-2",
                    name="write_task_artifact",
                    arguments=json.dumps(
                        {
                            "artifact_name": "todo.md",
                            "content": "- inspect the orchestrator stream path\n- inspect shared tool and todo handling\n",
                        }
                    ),
                ),
                response_id="neon-todo-2",
                narration="The plan is ready, so now I’m deriving the execution todo from it.",
            )
            return
        if "spawn_analyze_worker" in system_prompt and not spawn_scheduled:
            yield from self._tool_call_response(
                request,
                ToolCall(id="neon-spawn-1", name="spawn_analyze_worker", arguments="{}"),
                response_id="neon-spawn-1",
                narration="Everything is staged. Handing this off to the worker now.",
            )
            return
        if not spawn_scheduled:
            yield from self._tool_call_response(
                request,
                ToolCall(id="neon-spawn-1", name="spawn_analyze_worker", arguments="{}"),
                response_id="neon-spawn-1",
                narration="Everything is staged. Handing this off to the worker now.",
            )
            return
        if "findings.md" not in [payload.get("artifact_name") for payload in tool_payloads if isinstance(payload, dict)]:
            yield from self._tool_call_response(
                request,
                ToolCall(id="neon-findings-read", name="read_task_artifact", arguments='{"artifact_name":"findings.md"}'),
                response_id="neon-findings-read",
            )
            return
        if "completion.json" not in [payload.get("artifact_name") for payload in tool_payloads if isinstance(payload, dict)]:
            yield from self._tool_call_response(
                request,
                ToolCall(id="neon-completion-read", name="read_task_artifact", arguments='{"artifact_name":"completion.json"}'),
                response_id="neon-completion-read",
            )
            return
        yield from self._text_response(request, "Delegated analysis summary.", response_id="neon-delegated-final")

    def _worker_response(self, request: ProviderRequest, tool_messages: list[Message]) -> Iterator[ProviderEvent]:
        tool_names = [self._tool_name(message) for message in tool_messages]
        tool_payloads = [self._tool_payload(message) for message in tool_messages]
        if "read_task_artifact" not in tool_names:
            yield from self._tool_call_response(
                request,
                ToolCall(id="worker-task-1", name="read_task_artifact", arguments='{"artifact_name":"task.md"}'),
                response_id="worker-task-1",
                narration="I have the handoff. Reading the task artifacts first.",
            )
            return
        if tool_names.count("read_task_artifact") == 1:
            yield from self._tool_call_response(
                request,
                ToolCall(id="worker-plan-1", name="read_task_artifact", arguments='{"artifact_name":"plan.md"}'),
                response_id="worker-plan-1",
            )
            return
        if tool_names.count("read_task_artifact") == 2:
            yield from self._tool_call_response(
                request,
                ToolCall(id="worker-todo-1", name="read_task_artifact", arguments='{"artifact_name":"todo.md"}'),
                response_id="worker-todo-1",
            )
            return
        if "read_todo_state" not in tool_names:
            yield from self._tool_call_response(
                request,
                ToolCall(id="worker-todo-state-1", name="read_todo_state", arguments="{}"),
                response_id="worker-todo-state-1",
            )
            return
        todo_payload = next(
            (payload for payload in reversed(tool_payloads) if isinstance(payload, dict) and payload.get("items")),
            None,
        )
        todo_items = todo_payload.get("items", []) if isinstance(todo_payload, dict) else []
        started_ids = {str(payload.get("todo_id")) for payload in tool_payloads if isinstance(payload, dict) and payload.get("status") == "in_progress"}
        completed_ids = {str(payload.get("todo_id")) for payload in tool_payloads if isinstance(payload, dict) and payload.get("status") == "completed"}
        if todo_items:
            first_id = str(todo_items[0]["todo_id"])
            if first_id not in started_ids and first_id not in completed_ids:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="worker-start-1", name="start_todo_item", arguments=json.dumps({"todo_id": first_id})),
                    response_id="worker-start-1",
                    narration="Starting with the first delegated todo item.",
                )
                return
            if "search_code" not in tool_names:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="worker-search-1", name="search_code", arguments='{"pattern":"class NeonOrchestrator","path":"aca"}'),
                    response_id="worker-search-1",
                )
                return
            if first_id not in completed_ids:
                yield from self._tool_call_response(
                    request,
                    ToolCall(
                        id="worker-complete-1",
                        name="complete_todo_item",
                        arguments=json.dumps({"todo_id": first_id, "outcome": "Traced the orchestrator stream path."}),
                    ),
                    response_id="worker-complete-1",
                )
                return
        if len(todo_items) > 1:
            second_id = str(todo_items[1]["todo_id"])
            if second_id not in started_ids and second_id not in completed_ids:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="worker-start-2", name="start_todo_item", arguments=json.dumps({"todo_id": second_id})),
                    response_id="worker-start-2",
                )
                return
            if "read_file" not in tool_names:
                yield from self._tool_call_response(
                    request,
                    ToolCall(id="worker-read-1", name="read_file", arguments='{"path":"aca/orchestration/tools.py","start_line":1,"end_line":260}'),
                    response_id="worker-read-1",
                )
                return
            if second_id not in completed_ids:
                yield from self._tool_call_response(
                    request,
                    ToolCall(
                        id="worker-complete-2",
                        name="complete_todo_item",
                        arguments=json.dumps({"todo_id": second_id, "outcome": "Traced the shared tool and todo handling."}),
                    ),
                    response_id="worker-complete-2",
                )
                return
        if "write_task_artifact" not in tool_names:
            yield from self._tool_call_response(
                request,
                ToolCall(
                    id="worker-findings-1",
                    name="write_task_artifact",
                    arguments=json.dumps(
                        {
                            "artifact_name": "findings.md",
                            "content": "# Findings\n\nThe agent architecture centers on NeonOrchestrator, shared orchestration tools, and a delegated analyze worker.\n",
                        }
                    ),
                ),
                response_id="worker-findings-1",
                narration="The todo is complete. I’m consolidating the findings now.",
            )
            return
        yield from self._text_response(request, "Worker finished.", response_id="worker-final")

    def _route_for_request(self, full_user_text: str) -> str:
        lowered = full_user_text.lower()
        if any(word in lowered for word in ("fix ", "implement", "edit ", "refactor")):
            return "implement_disabled"
        if "scan the codebase" in lowered or "codebase" in lowered or "architecture" in lowered:
            return "analyze_delegated"
        if "analyze" in lowered or "inspect" in lowered:
            return "analyze_simple"
        return "chat"

    def _extract_task_id(self, system_prompt: str) -> str:
        marker = "use task_id `"
        if marker in system_prompt:
            tail = system_prompt.split(marker, 1)[1]
            return tail.split("`", 1)[0]
        raise AssertionError(f"Task id not found in system prompt: {system_prompt!r}")

    def _tool_payload(self, message: Message) -> dict[str, object] | None:
        if not message.content:
            return None
        try:
            parsed = json.loads(str(message.content))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _tool_name(self, message: Message) -> str:
        return str(message.metadata.get("tool_name", ""))

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

    def _tool_call_response(
        self,
        request: ProviderRequest,
        tool_call: ToolCall,
        *,
        response_id: str,
        narration: str | None = None,
    ) -> Iterator[ProviderEvent]:
        if narration:
            yield ProviderEvent(type="text.delta", delta=narration)
        yield ProviderEvent(
            type="response.completed",
            result=RunResult(
                provider=self.provider_name,
                model=request.model,
                assistant_message=Message(
                    role="assistant",
                    content=narration or "Need to use a tool first.",
                    tool_calls=[tool_call],
                    reasoning="Use the tool first.",
                    reasoning_details=[],
                ),
                reasoning="Use the tool first.",
                reasoning_details=[],
                text=narration or "Need to use a tool first.",
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


class PrematureAnswerProvider(FakeNeonProvider):
    def __init__(self) -> None:
        super().__init__()
        self.premature_answer_emitted = False

    def _neon_response(self, request: ProviderRequest, full_user_text: str, tool_messages: list[Message]) -> Iterator[ProviderEvent]:
        tool_names = [self._tool_name(message) for message in tool_messages]
        tool_payloads = [self._tool_payload(message) for message in tool_messages]
        written_artifacts = {
            payload.get("artifact_name"): payload
            for payload in tool_payloads
            if isinstance(payload, dict) and payload.get("artifact_name")
        }
        if not self.premature_answer_emitted and "task.md" not in written_artifacts and "list_files" in tool_names:
            self.premature_answer_emitted = True
            yield from self._text_response(request, "Premature architectural explanation that should never be shown.", response_id="premature-1")
            return
        yield from super()._neon_response(request, full_user_text, tool_messages)


class NeonOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "aca" / "orchestration").mkdir(parents=True, exist_ok=True)
        (self.root / "aca" / "orchestration" / "orchestrator.py").write_text("class NeonOrchestrator:\n    pass\n", encoding="utf-8")
        (self.root / "aca" / "orchestration" / "tools.py").write_text("class OrchestrationToolRegistry:\n    pass\n", encoding="utf-8")
        settings = Settings(
            sqlite_url=f"sqlite:///{self.root / 'aca.db'}",
            chroma_path=str(self.root / "chroma"),
            chroma_collection="aca-neon-test",
        )
        self.storage = initialize_storage(settings)
        self.provider = FakeNeonProvider()

        with self.storage.session_factory.begin() as session:
            session.add(
                Conversation(
                    id="conv-1",
                    user_id=None,
                    title="New chat",
                    status="active",
                    active_model="fake-model",
                    history_mode="dialogue_only",
                    thinking_enabled=False,
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

    def _orchestrator(self) -> NeonOrchestrator:
        return NeonOrchestrator(
            session_factory=self.storage.session_factory,
            provider=self.provider,
            approval_policy=AllowAllApprovalPolicy(),
            workspace_root=self.root,
            archive_root=self.root / ".archive",
        )

    def _run(self, prompt: str, *, provider: LLMProvider | None = None) -> list[object]:
        orchestrator = (
            NeonOrchestrator(
                session_factory=self.storage.session_factory,
                provider=provider,
                approval_policy=AllowAllApprovalPolicy(),
                workspace_root=self.root,
                archive_root=self.root / ".archive",
            )
            if provider is not None
            else self._orchestrator()
        )
        return list(
            orchestrator.stream_turn(
                conversation_id="conv-1",
                user_input=prompt,
                model="fake-model",
                thinking_enabled=False,
            )
        )

    def test_chat_returns_directly_without_task_workspace(self) -> None:
        events = self._run("hey")
        text_events = [event for event in events if event.type == "text.delta"]
        self.assertTrue(text_events)
        self.assertEqual("".join(event.text or "" for event in text_events), "Neon chat reply.")
        completed = next(event for event in events if event.type == "completed")
        self.assertEqual(completed.final_answer, "Neon chat reply.")
        self.assertFalse((self.root / ".aca").exists())
        with self.storage.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Task)), 0)

    def test_simple_analyze_creates_task_and_executes_todo(self) -> None:
        events = self._run("inspect the repo and explain the main flow")
        completed = next(event for event in events if event.type == "completed")
        self.assertEqual(completed.final_answer, "Simple analysis summary.")
        task_dir = next((self.root / ".aca" / "tasks").iterdir())
        self.assertTrue((task_dir / "task.md").exists())
        self.assertTrue((task_dir / "todo.md").exists())
        self.assertFalse((task_dir / "plan.md").exists())
        self.assertFalse((task_dir / "findings.md").exists())
        with self.storage.session_factory() as session:
            actions = session.scalars(select(Action).order_by(Action.started_at.asc())).all()
            action_types = [action.action_type for action in actions]
            self.assertIn("todo_item_started", action_types)
            self.assertIn("todo_item_completed", action_types)

    def test_delegated_analyze_hands_off_and_writes_findings(self) -> None:
        events = self._run("scan the codebase and tell me how the agent architecture works")
        completed = next(event for event in events if event.type == "completed")
        self.assertEqual(completed.final_answer, "Delegated analysis summary.")
        task_dir = next((self.root / ".aca" / "tasks").iterdir())
        self.assertTrue((task_dir / "task.md").exists())
        self.assertTrue((task_dir / "plan.md").exists())
        self.assertTrue((task_dir / "todo.md").exists())
        self.assertTrue((task_dir / "findings.md").exists())
        self.assertTrue((task_dir / "completion.json").exists())
        worker_events = [event for event in events if getattr(event, "agent", None) == "worker"]
        self.assertTrue(worker_events)
        with self.storage.session_factory() as session:
            worker_actions = session.scalars(
                select(Action).where(Action.agent_id == "worker").order_by(Action.started_at.asc())
            ).all()
            self.assertTrue(any(action.action_type == "todo_item_started" for action in worker_actions))
            self.assertTrue(any(action.action_type == "todo_item_completed" for action in worker_actions))

    def test_premature_no_tool_answer_is_suppressed(self) -> None:
        events = self._run(
            "scan the codebase and tell me how the agent architecture works",
            provider=PrematureAnswerProvider(),
        )
        all_text = "".join(event.text or "" for event in events if getattr(event, "text", None))
        self.assertNotIn("Premature architectural explanation that should never be shown.", all_text)
        status_messages = [event.message for event in events if event.type == "worker.status" and event.message]
        self.assertTrue(any("task" in message.lower() or "scout" in message.lower() for message in status_messages))

    def test_tool_errors_are_rendered_as_errors(self) -> None:
        orchestrator = self._orchestrator()
        summary = orchestrator._summarize_tool_result("read_file", json.dumps({"error": "blocked by guardrail"}))
        self.assertIn("hit a guardrail", summary)

    def test_blocked_pretask_read_does_not_consume_budget(self) -> None:
        (self.root / ".env").write_text("SECRET=value\n", encoding="utf-8")
        state = NeonRunState(
            conversation_id="conv-1",
            user_input="explain the repo",
            user_name="Ayush",
            planned_task_id="task-1",
            model="fake-model",
            thinking_enabled=False,
            carryover_messages=[],
        )
        context = WorkspaceToolContext(
            root=self.root,
            session_factory=self.storage.session_factory,
            task_id=None,
            agent_id="master",
            approval_policy=None,
        )
        registry = OrchestrationToolRegistry(
            orchestrator=self._orchestrator(),
            actor=ACTOR_NEON,
            route=None,
            state=state,
            workspace_manager=TaskWorkspaceManager(self.root, self.root / ".archive"),
            read_registry=WorkspaceToolRegistry(context),
            context=context,
        )

        with self.assertRaises(ValueError):
            registry.read_file(".env")

        self.assertEqual(state.pretask_read_calls, 0)

    def test_pretask_repo_reads_are_removed_from_tool_surface_after_budget(self) -> None:
        state = NeonRunState(
            conversation_id="conv-1",
            user_input="explain the repo",
            user_name="Ayush",
            planned_task_id="task-1",
            model="fake-model",
            thinking_enabled=False,
            carryover_messages=[],
            pretask_read_calls=2,
        )
        context = WorkspaceToolContext(
            root=self.root,
            session_factory=self.storage.session_factory,
            task_id=None,
            agent_id="master",
            approval_policy=None,
        )
        registry = OrchestrationToolRegistry(
            orchestrator=self._orchestrator(),
            actor=ACTOR_NEON,
            route=None,
            state=state,
            workspace_manager=TaskWorkspaceManager(self.root, self.root / ".archive"),
            read_registry=WorkspaceToolRegistry(context),
            context=context,
        )

        tool_names = [tool["function"]["name"] for tool in registry.schemas()]

        self.assertNotIn("list_files", tool_names)
        self.assertNotIn("read_file", tool_names)
        self.assertNotIn("search_code", tool_names)
        self.assertIn("write_task_artifact", tool_names)

    def test_implement_request_is_disabled_without_task(self) -> None:
        events = self._run("fix the bug in aca/orchestration/orchestrator.py")
        completed = next(event for event in events if event.type == "completed")
        self.assertIn("disabled", completed.final_answer.lower())
        self.assertFalse((self.root / ".aca").exists())
        with self.storage.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Task)), 0)

    def test_archive_sweep_moves_completed_task_out_of_repo(self) -> None:
        self._run("inspect the repo and explain the main flow")
        with self.storage.session_factory.begin() as session:
            task = session.scalars(select(Task)).first()
            assert task is not None
            task.completed_at = datetime.utcnow() - timedelta(hours=13)
            metadata = dict(task.metadata_json or {})
            metadata["archive_due_at"] = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            task.metadata_json = metadata
        orchestrator = self._orchestrator()
        orchestrator.sweep_archives()
        self.assertEqual(list((self.root / ".aca" / "tasks").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
