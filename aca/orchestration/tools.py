from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aca.core_types import TurnIntent
from aca.orchestration.protocols import OrchestratorBackend
from aca.orchestration.state import (
    ACTOR_ANALYZE_WORKER,
    ACTOR_IMPLEMENT_WORKER,
    ACTOR_NEON,
    ANALYZE_DELEGATED_ROUTE,
    ANALYZE_ROUTES,
    ANALYZE_SIMPLE_ROUTE,
    IMPLEMENT_ROUTE,
    MAX_ORIENTATION_READS,
    MAX_PRETASK_READS,
    PHASE_DELEGATED_WAIT,
    PHASE_ORIENTATION,
    PHASE_SYNTHESIZE,
    PHASE_TASK_CREATED,
    PHASE_TODO_READY,
    STEERING_ORIENTATION_DELEGATED,
    STEERING_ORIENTATION_SIMPLE,
    STEERING_PLAN_REQUIRED,
    STEERING_PRETASK_LIMIT_REACHED,
    STEERING_READ_FINDINGS,
    STEERING_READ_OUTPUT,
    STEERING_SPAWN_WORKER_REQUIRED,
    STEERING_TASK_REQUIRED_NOW,
    STEERING_TODO_ITEM_REQUIRED,
    STEERING_TODO_REQUIRED,
    STEERING_WORKER_WRITE_FINDINGS_NOW,
    STEERING_WORKER_WRITE_OUTPUT_NOW,
    AnalyzeWorkerState,
    ImplementWorkerState,
    NeonGuardrailError,
    NeonRunState,
)
from aca.orchestration.todo import TodoController, todo_in_progress, todo_is_complete
from aca.task_workspace import CANONICAL_REPO_DOCS, ParsedTaskDocument, TaskWorkspaceManager, parse_task_markdown
from aca.tools import WorkspaceToolContext, WorkspaceToolRegistry


def tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class OrchestrationToolRegistry:
    def __init__(
        self,
        *,
        orchestrator: OrchestratorBackend,
        actor: str,
        route: str | None,
        state: NeonRunState | AnalyzeWorkerState | ImplementWorkerState,
        workspace_manager: TaskWorkspaceManager,
        read_registry: WorkspaceToolRegistry,
        context: WorkspaceToolContext | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._actor = actor
        self._route = route
        self._state = state
        self._workspace_manager = workspace_manager
        self._read_registry = read_registry
        self._context = context
        self._todo = TodoController(
            actor="worker" if actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER} else "neon",
            state=state,
            route=route,
            persist=self._orchestrator._persist_todo_state,
            new_id=self._orchestrator._new_id,
        )

    def _steered_result(self, payload: dict[str, Any], steering_code: str | None) -> str:
        """Format a tool result with steering as the top-level, unmissable directive.

        When a steering code is present the result is rewritten so the very first
        fields the model sees are the human-readable instruction and the code:

            {"NEXT_STEP": "<instruction>", "steering_code": "<code>", ...rest}

        This follows the industry-standard pattern of making steering the most
        prominent part of tool output so the model cannot overlook it.
        """
        if not steering_code:
            return json.dumps(payload)
        instruction = self._next_step_instruction(steering_code)
        ordered: dict[str, Any] = {
            "NEXT_STEP": instruction,
            "steering_code": steering_code,
        }
        for key, value in payload.items():
            if key not in ordered:
                ordered[key] = value
        return json.dumps(ordered)

    def _promote_steering(self, raw_result: str) -> str:
        """Re-order an existing tool result so any steering_code becomes a top-level NEXT_STEP."""
        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            return raw_result
        if not isinstance(payload, dict):
            return raw_result
        code = payload.pop("steering_code", None)
        return self._steered_result(payload, code)

    def schemas(self) -> list[dict[str, Any]]:
        # When the worker has finished all todos but hasn't written its artifact yet,
        # lock down the tool surface to write_task_artifact only so the LLM can't go off-rail.
        if self._worker_artifact_only_mode():
            writable_artifacts = self._writable_artifact_names()
            return [
                tool_schema(
                    "write_task_artifact",
                    "Write an approved task artifact inside the active task workspace.",
                    {
                        "type": "object",
                        "properties": {
                            "artifact_name": {"type": "string", "enum": writable_artifacts},
                            "content": {"type": "string"},
                        },
                        "required": ["artifact_name", "content"],
                    },
                )
            ]

        tools: list[dict[str, Any]] = []
        if self._workspace_tools_available():
            tools.extend(self._read_registry.schemas(allow_mutation=self._workspace_mutations_available()))
        if self._actor == ACTOR_NEON and self._repo_summary_available():
            tools.append(tool_schema("get_repo_summary", "Return a compact system-generated summary of the repository shape and likely stack.", {"type": "object", "properties": {}}))
        if self._write_task_artifact_available():
            writable_artifacts = self._writable_artifact_names()
            tools.append(
                tool_schema(
                    "write_task_artifact",
                    "Write an approved task artifact inside the active task workspace.",
                    {
                        "type": "object",
                        "properties": {
                            "artifact_name": {"type": "string", "enum": writable_artifacts},
                            "content": {"type": "string"},
                        },
                        "required": ["artifact_name", "content"],
                    },
                )
            )
        if self._read_task_artifact_available():
            readable_artifacts = self._readable_artifact_names()
            tools.append(
                tool_schema(
                    "read_task_artifact",
                    "Read a task artifact from the active task workspace.",
                    {
                        "type": "object",
                        "properties": {"artifact_name": {"type": "string", "enum": readable_artifacts}},
                        "required": ["artifact_name"],
                    },
                )
            )
        if self._todo_tools_available():
            tools.extend(
                [
                    tool_schema("read_todo_state", "Read the structured todo execution state for the active task.", {"type": "object", "properties": {}}),
                    tool_schema(
                        "start_todo_item",
                        "Mark one todo item as in progress before doing deep analysis work.",
                        {
                            "type": "object",
                            "properties": {"todo_id": {"type": "string"}},
                            "required": ["todo_id"],
                        },
                    ),
                    tool_schema(
                        "complete_todo_item",
                        "Mark the current todo item as completed and record its outcome.",
                        {
                            "type": "object",
                            "properties": {
                                "todo_id": {"type": "string"},
                                "outcome": {"type": "string"},
                            },
                            "required": ["todo_id", "outcome"],
                        },
                    ),
                    tool_schema(
                        "skip_todo_item",
                        "Skip a todo item and record why it was skipped.",
                        {
                            "type": "object",
                            "properties": {
                                "todo_id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["todo_id", "reason"],
                        },
                    ),
                    tool_schema(
                        "revise_todo",
                        "Replace the todo list with a revised set of items and record why the plan changed.",
                        {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"title": {"type": "string"}},
                                        "required": ["title"],
                                    },
                                },
                                "reason": {"type": "string"},
                                "confidence_score": {"type": "number"},
                            },
                            "required": ["items", "reason", "confidence_score"],
                        },
                    ),
                ]
            )
        if self._actor == ACTOR_NEON:
            if self._spawn_worker_available():
                if self._route == IMPLEMENT_ROUTE:
                    tools.append(tool_schema("spawn_implement_worker", "Spawn the delegated implement worker after task.md, plan.md, and todo.md are ready.", {"type": "object", "properties": {}}))
                else:
                    tools.append(tool_schema("spawn_analyze_worker", "Spawn the delegated analyze worker after task.md, plan.md, and todo.md are ready.", {"type": "object", "properties": {}}))
            if self._repo_doc_available():
                tools.append(
                    tool_schema(
                        "write_repo_doc",
                        "Write one of the canonical repo-level Neon documents.",
                        {
                            "type": "object",
                            "properties": {
                                "doc_name": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["doc_name", "content"],
                        },
                    )
                )
            if self._web_search_available():
                tools.append(
                    tool_schema(
                        "web_search",
                        "Run a lightweight web search when repo analysis alone is insufficient.",
                        {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    )
                )
        return tools

    def handlers(self) -> dict[str, Any]:
        all_handlers = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "file_ops": self.file_ops,
            "run_command": self.run_command,
            "write_task_artifact": self.write_task_artifact,
            "read_task_artifact": self.read_task_artifact,
            "read_todo_state": self.read_todo_state,
            "start_todo_item": self.start_todo_item,
            "complete_todo_item": self.complete_todo_item,
            "skip_todo_item": self.skip_todo_item,
            "revise_todo": self.revise_todo,
            "get_repo_summary": self.get_repo_summary,
            "spawn_analyze_worker": self.spawn_analyze_worker,
            "spawn_implement_worker": self.spawn_implement_worker,
            "write_repo_doc": self.write_repo_doc,
            "web_search": self.web_search,
        }
        allowed_names = {tool["function"]["name"] for tool in self.schemas()}
        return {name: handler for name, handler in all_handlers.items() if name in allowed_names}

    def list_files(self, *args, **kwargs) -> str:
        self._reject_if_artifact_only("list_files")
        return self._run_repo_read(self._read_registry.list_files, *args, **kwargs)

    def read_file(self, *args, **kwargs) -> str:
        self._reject_if_artifact_only("read_file")
        return self._run_repo_read(self._read_registry.read_file, *args, **kwargs)

    def search_code(self, *args, **kwargs) -> str:
        self._reject_if_artifact_only("search_code")
        return self._run_repo_read(self._read_registry.search_code, *args, **kwargs)

    def write_file(self, *args, **kwargs) -> str:
        self._ensure_workspace_mutations_allowed()
        return self._read_registry.write_file(*args, **kwargs)

    def edit_file(self, *args, **kwargs) -> str:
        self._ensure_workspace_mutations_allowed()
        return self._read_registry.edit_file(*args, **kwargs)

    def file_ops(self, *args, **kwargs) -> str:
        self._ensure_workspace_mutations_allowed()
        return self._read_registry.file_ops(*args, **kwargs)

    def run_command(self, *args, **kwargs) -> str:
        self._ensure_workspace_mutations_allowed()
        return self._read_registry.run_command(*args, **kwargs)

    def _writable_artifact_names(self) -> list[str]:
        if self._actor == ACTOR_ANALYZE_WORKER:
            return ["findings.md"]
        if self._actor == ACTOR_IMPLEMENT_WORKER:
            return ["output.md"]
        return ["task.md", "plan.md", "todo.md"]

    def _readable_artifact_names(self) -> list[str]:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return ["task.md", "plan.md", "todo.md"]
        state = self._state
        assert isinstance(state, NeonRunState)
        names = ["task.md", "plan.md", "todo.md"]
        if state.route == ANALYZE_DELEGATED_ROUTE:
            names.extend(["findings.md", "completion.json"])
        elif state.route == IMPLEMENT_ROUTE:
            names.extend(["output.md", "completion.json"])
        return names

    def get_repo_summary(self) -> str:
        self._ensure_actor(ACTOR_NEON)
        self._ensure_not_waiting_for_worker()
        payload = self._orchestrator._repo_summary_service.get(force_refresh=False)
        if isinstance(self._state, NeonRunState):
            self._state.repo_summary_used = True
        self._orchestrator._record_runtime_action(
            task_id=self._state.task_id,
            agent_id="master",
            action_type="tool:get_repo_summary",
            target=str(self._orchestrator._workspace_root),
            input_json={},
            result_json={"top_level_count": payload.get("top_level_count", 0), "languages": payload.get("languages", [])},
        )
        return json.dumps(payload)

    def write_task_artifact(self, artifact_name: str, content: str) -> str:
        artifact_name = self._normalize_artifact_name(artifact_name)
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            expected_artifact = "findings.md" if self._actor == ACTOR_ANALYZE_WORKER else "output.md"
            if artifact_name != expected_artifact:
                raise NeonGuardrailError(STEERING_TODO_REQUIRED, f"Worker may write {expected_artifact} only.")
            path = self._workspace_manager.write_task_artifact(self._state.task_id, artifact_name, content)
            relative_path = path.relative_to(self._orchestrator._workspace_root).as_posix()
            if self._actor == ACTOR_ANALYZE_WORKER:
                self._state.findings_written = True
                self._state.findings_path = relative_path
            else:
                self._state.output_written = True
                self._state.output_path = relative_path
            self._orchestrator._record_artifact_write(
                task_id=self._state.task_id,
                agent_id="worker",
                artifact_name=artifact_name,
                relative_path=relative_path,
            )
            return json.dumps({"artifact_name": artifact_name, "path": relative_path})

        self._ensure_not_waiting_for_worker()
        if artifact_name not in {"task.md", "plan.md", "todo.md"}:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, f"You may not write {artifact_name}.")

        state = self._state
        assert isinstance(state, NeonRunState)
        if artifact_name == "task.md":
            parsed = parse_task_markdown(content)
            if parsed.task_id != state.planned_task_id:
                raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "task.md must use the runtime task id exactly.")
            if parsed.route not in ANALYZE_ROUTES | {IMPLEMENT_ROUTE}:
                raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "task.md route must be analyze_simple, analyze_delegated, or implement.")
            if state.task_written:
                if not (state.route == ANALYZE_SIMPLE_ROUTE and parsed.route == ANALYZE_DELEGATED_ROUTE and state.task_id == parsed.task_id):
                    raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "task.md may only be rewritten once to upgrade simple analysis into delegated analysis.")
                state.route = ANALYZE_DELEGATED_ROUTE
                state.intent = TurnIntent.ANALYZE
                state.task_title = parsed.title
                state.task_statement = parsed.normalized_task
                state.plan_written = False
                state.todo_written = False
                state.todo_items = []
                state.current_todo_id = None
                state.phase = PHASE_ORIENTATION
                path = self._workspace_manager.write_task_artifact(state.task_id, artifact_name, content)
                relative_path = path.relative_to(self._orchestrator._workspace_root).as_posix()
                state.artifact_paths[artifact_name] = relative_path
                self._orchestrator._record_artifact_write(task_id=state.task_id, agent_id="master", artifact_name=artifact_name, relative_path=relative_path)
                self._orchestrator._update_task_metadata_after_task_rewrite(state, parsed)
                return self._steered_result(
                    {
                        "artifact_name": artifact_name,
                        "path": relative_path,
                        "summary": "Updated task.md to delegated analysis.",
                    },
                    STEERING_ORIENTATION_DELEGATED,
                )

            state.intent = TurnIntent.IMPLEMENT if parsed.route == IMPLEMENT_ROUTE else TurnIntent.ANALYZE
            state.route = parsed.route
            state.task_title = parsed.title
            state.task_statement = parsed.normalized_task
            task_id = self._orchestrator._ensure_task_created(state, parsed)
            if self._context is not None:
                self._context.task_id = task_id
            self._workspace_manager.ensure_task_dir(task_id)
            path = self._workspace_manager.write_task_artifact(task_id, artifact_name, content)
            relative_path = path.relative_to(self._orchestrator._workspace_root).as_posix()
            state.task_written = True
            state.phase = PHASE_TASK_CREATED
            state.artifact_paths[artifact_name] = relative_path
            self._orchestrator._record_artifact_write(task_id=task_id, agent_id="master", artifact_name=artifact_name, relative_path=relative_path)
            steering_code = STEERING_PLAN_REQUIRED if parsed.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} else STEERING_TODO_REQUIRED
            return self._steered_result(
                {
                    "artifact_name": artifact_name,
                    "path": relative_path,
                    "summary": "task.md created.",
                },
                steering_code,
            )

        if not state.task_written or state.task_id is None:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "You must write task.md first.")
        if state.route == ANALYZE_SIMPLE_ROUTE:
            if artifact_name == "plan.md":
                raise NeonGuardrailError(STEERING_TODO_REQUIRED, "plan.md is not allowed for this route.")
            if artifact_name != "todo.md":
                raise NeonGuardrailError(STEERING_TODO_REQUIRED, "For this route, write todo.md after orientation.")
        elif state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE}:
            if state.worker_spawned:
                target = "output.md" if state.route == IMPLEMENT_ROUTE else "findings.md"
                read_code = STEERING_READ_OUTPUT if state.route == IMPLEMENT_ROUTE else STEERING_READ_FINDINGS
                raise NeonGuardrailError(read_code, f"The delegated worker already ran. Read {target} instead.")
            if state.todo_written:
                next_tool = "spawn_implement_worker" if state.route == IMPLEMENT_ROUTE else "spawn_analyze_worker"
                raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, f"After delegated todo.md, the next action must be {next_tool}.")
            if artifact_name == "todo.md" and not state.plan_written:
                route_label = "implement" if state.route == IMPLEMENT_ROUTE else "analyze_delegated"
                raise NeonGuardrailError(STEERING_PLAN_REQUIRED, f"For {route_label}, write plan.md before todo.md.")
            if artifact_name == "plan.md" and state.plan_written:
                raise NeonGuardrailError(STEERING_TODO_REQUIRED, "plan.md is already written. Write todo.md next.")
        if artifact_name == "todo.md":
            todo_items = self._orchestrator._initialize_todo_state(state, content)
            path = self._workspace_manager.write_todo_state(state.task_id, todo_items)
            state.todo_items = todo_items
            state.current_todo_id = None
            state.phase = PHASE_TODO_READY if state.route == ANALYZE_SIMPLE_ROUTE else PHASE_DELEGATED_WAIT
        else:
            path = self._workspace_manager.write_task_artifact(state.task_id, artifact_name, content)
        relative_path = path.relative_to(self._orchestrator._workspace_root).as_posix()
        state.artifact_paths[artifact_name] = relative_path
        if artifact_name == "plan.md":
            state.plan_written = True
        if artifact_name == "todo.md":
            state.todo_written = True
        self._orchestrator._record_artifact_write(task_id=state.task_id, agent_id="master", artifact_name=artifact_name, relative_path=relative_path)
        steering_code = None
        summary = f"wrote {artifact_name}"
        if artifact_name == "plan.md":
            steering_code = STEERING_TODO_REQUIRED
            summary = "plan.md created."
        elif artifact_name == "todo.md" and state.route == ANALYZE_SIMPLE_ROUTE:
            steering_code = STEERING_TODO_ITEM_REQUIRED
            summary = "todo.md created."
        elif artifact_name == "todo.md":
            steering_code = STEERING_SPAWN_WORKER_REQUIRED
            summary = "todo.md created."
        payload: dict[str, Any] = {"artifact_name": artifact_name, "path": relative_path, "summary": summary}
        if artifact_name == "todo.md":
            payload["items"] = state.todo_items
        return self._steered_result(payload, steering_code)

    def read_task_artifact(self, artifact_name: str) -> str:
        self._reject_if_artifact_only("read_task_artifact")
        artifact_name = self._normalize_artifact_name(artifact_name)
        if self._actor == ACTOR_NEON and self._state.worker_requested and not self._state.worker_spawned:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "The delegated worker handoff has started. Stop polling artifacts.")
        if not self._state.task_id:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "No active task workspace exists yet.")
        allowed = {"task.md", "plan.md", "todo.md"}
        if self._actor == ACTOR_NEON:
            if self._route == ANALYZE_DELEGATED_ROUTE:
                allowed |= {"findings.md", "completion.json"}
            elif self._route == IMPLEMENT_ROUTE:
                allowed |= {"output.md", "completion.json"}
            if self._route == ANALYZE_DELEGATED_ROUTE and not self._state.worker_spawned and artifact_name in {"findings.md", "completion.json"}:
                raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "The delegated worker has not completed yet.")
            if self._route == IMPLEMENT_ROUTE and not self._state.worker_spawned and artifact_name in {"output.md", "completion.json"}:
                raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "The delegated worker has not completed yet.")
        if artifact_name not in allowed:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, f"Unsupported task artifact: {artifact_name}")
        content = self._workspace_manager.read_task_artifact(self._state.task_id, artifact_name)
        path = self._workspace_manager.artifact_relative_path(self._state.task_id, artifact_name)
        self._orchestrator._record_runtime_action(
            task_id=self._state.task_id,
            agent_id="master" if self._actor == ACTOR_NEON else "worker",
            action_type="tool:read_task_artifact",
            target=path,
            input_json={"artifact_name": artifact_name},
            result_json={"path": path},
        )
        return json.dumps({"artifact_name": artifact_name, "path": path, "content": content})

    def read_todo_state(self) -> str:
        self._reject_if_artifact_only("read_todo_state")
        self._ensure_todo_exists()
        payload = json.loads(self._todo.read_state())
        self._orchestrator._record_runtime_action(
            task_id=self._state.task_id,
            agent_id="master" if self._actor == ACTOR_NEON else "worker",
            action_type="tool:read_todo_state",
            target=self._workspace_manager.artifact_relative_path(self._state.task_id, "todo.md"),
            input_json={},
            result_json={"current_todo_id": self._state.current_todo_id, "count": len(self._state.todo_items)},
        )
        return json.dumps(payload)

    def start_todo_item(self, todo_id: str) -> str:
        self._reject_if_artifact_only("start_todo_item")
        self._ensure_not_waiting_for_worker()
        return self._promote_steering(self._todo.start(todo_id))

    def complete_todo_item(self, todo_id: str, outcome: str) -> str:
        self._ensure_not_waiting_for_worker()
        return self._promote_steering(self._todo.complete(todo_id, outcome))

    def skip_todo_item(self, todo_id: str, reason: str) -> str:
        self._ensure_not_waiting_for_worker()
        return self._promote_steering(self._todo.skip(todo_id, reason))

    def revise_todo(self, items: list[dict[str, Any]], reason: str, confidence_score: float) -> str:
        self._ensure_not_waiting_for_worker()
        return self._todo.revise(items, reason, confidence_score)

    def spawn_analyze_worker(self) -> str:
        self._ensure_actor(ACTOR_NEON)
        state = self._state
        assert isinstance(state, NeonRunState)
        if state.route != ANALYZE_DELEGATED_ROUTE:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "spawn_analyze_worker is only allowed for delegated analysis.")
        if not state.task_id or not state.task_written or not state.plan_written or not state.todo_written:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "Finish task.md, plan.md, and todo.md before spawning the worker.")
        if state.worker_spawned:
            raise NeonGuardrailError(STEERING_READ_FINDINGS, "The delegated worker already ran. Read findings.md and completion.json.")
        state.worker_requested = True
        state.phase = PHASE_DELEGATED_WAIT
        self._orchestrator._record_runtime_action(
            task_id=state.task_id,
            agent_id="master",
            action_type="delegated_worker_requested",
            target=state.task_id,
            input_json={"route": state.route},
            result_json={"status": "scheduled"},
        )
        return json.dumps({"status": "scheduled", "summary": "Analyze worker scheduled.", "steering_code": STEERING_SPAWN_WORKER_REQUIRED})

    def spawn_implement_worker(self) -> str:
        self._ensure_actor(ACTOR_NEON)
        state = self._state
        assert isinstance(state, NeonRunState)
        if state.route != IMPLEMENT_ROUTE:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "spawn_implement_worker is only allowed for implement.")
        if not state.task_id or not state.task_written or not state.plan_written or not state.todo_written:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "Finish task.md, plan.md, and todo.md before spawning the implement worker.")
        if state.worker_spawned:
            raise NeonGuardrailError(STEERING_READ_OUTPUT, "The implement worker already ran. Read output.md and completion.json.")
        state.worker_requested = True
        state.phase = PHASE_DELEGATED_WAIT
        self._orchestrator._record_runtime_action(
            task_id=state.task_id,
            agent_id="master",
            action_type="delegated_worker_requested",
            target=state.task_id,
            input_json={"route": state.route},
            result_json={"status": "scheduled"},
        )
        return json.dumps({"status": "scheduled", "summary": "Implement worker scheduled.", "steering_code": STEERING_SPAWN_WORKER_REQUIRED})

    def write_repo_doc(self, doc_name: str, content: str) -> str:
        self._ensure_actor(ACTOR_NEON)
        self._ensure_not_waiting_for_worker()
        if doc_name not in CANONICAL_REPO_DOCS:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "You may only write canonical repo docs.")
        path = (self._orchestrator._workspace_root / doc_name).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        relative_path = path.relative_to(self._orchestrator._workspace_root).as_posix()
        self._orchestrator._record_runtime_action(
            task_id=self._state.task_id,
            agent_id="master",
            action_type="tool:write_repo_doc",
            target=relative_path,
            input_json={"doc_name": doc_name},
            result_json={"path": relative_path},
        )
        return json.dumps({"doc_name": doc_name, "path": relative_path})

    def web_search(self, query: str, limit: int = 5) -> str:
        self._ensure_actor(ACTOR_NEON)
        self._ensure_not_waiting_for_worker()
        state = self._state
        assert isinstance(state, NeonRunState)
        if not state.task_written:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "web_search is not allowed before task.md.")
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.todo_written and not state.worker_spawned:
            next_tool = "spawn_implement_worker" if state.route == IMPLEMENT_ROUTE else "spawn_analyze_worker"
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, f"After delegated todo.md, the next action must be {next_tool}.")
        payload = self._orchestrator._web_search(query=query, limit=limit)
        self._orchestrator._record_runtime_action(
            task_id=state.task_id,
            agent_id="master",
            action_type="tool:web_search",
            target=query,
            input_json={"query": query, "limit": limit},
            result_json={"count": payload.get("count", 0)},
        )
        return json.dumps(payload)

    def _before_repo_read(self) -> None:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return
        self._ensure_not_waiting_for_worker()
        state = self._state
        assert isinstance(state, NeonRunState)
        if not state.task_written:
            if state.pretask_read_calls >= MAX_PRETASK_READS:
                raise NeonGuardrailError(
                    STEERING_PRETASK_LIMIT_REACHED,
                    f"You cannot make another pre-task repo read now. {self._next_step_instruction(STEERING_PRETASK_LIMIT_REACHED)}",
                )
            state.pretask_read_calls += 1
            return
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.worker_spawned:
            read_code = STEERING_READ_OUTPUT if state.route == IMPLEMENT_ROUTE else STEERING_READ_FINDINGS
            raise NeonGuardrailError(read_code, self._next_step_instruction(read_code))
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.todo_written and not state.worker_spawned:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, self._next_step_instruction(STEERING_SPAWN_WORKER_REQUIRED))
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.plan_written and not state.todo_written:
            raise NeonGuardrailError(STEERING_TODO_REQUIRED, self._next_step_instruction(STEERING_TODO_REQUIRED))
        if state.route == ANALYZE_SIMPLE_ROUTE and not state.todo_written:
            if state.orientation_read_calls >= MAX_ORIENTATION_READS:
                code = STEERING_ORIENTATION_SIMPLE
                raise NeonGuardrailError(code, self._next_step_instruction(code))
            state.orientation_read_calls += 1
            state.phase = PHASE_ORIENTATION
            return
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and not state.plan_written:
            if state.orientation_read_calls >= MAX_ORIENTATION_READS:
                raise NeonGuardrailError(STEERING_ORIENTATION_DELEGATED, self._next_step_instruction(STEERING_ORIENTATION_DELEGATED))
            state.orientation_read_calls += 1
            state.phase = PHASE_ORIENTATION
            return
        if state.route == ANALYZE_SIMPLE_ROUTE:
            if todo_is_complete(state.todo_items):
                raise NeonGuardrailError(STEERING_READ_FINDINGS, self._next_step_instruction(STEERING_READ_FINDINGS))
            if todo_in_progress(state.todo_items) is None:
                raise NeonGuardrailError(STEERING_TODO_ITEM_REQUIRED, self._next_step_instruction(STEERING_TODO_ITEM_REQUIRED))

    def _run_repo_read(self, handler: Any, *args, **kwargs) -> str:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return handler(*args, **kwargs)

        state = self._state
        assert isinstance(state, NeonRunState)
        before_pretask = state.pretask_read_calls
        before_orientation = state.orientation_read_calls
        before_phase = state.phase
        self._before_repo_read()
        try:
            raw_result = handler(*args, **kwargs)
        except Exception:
            state.pretask_read_calls = before_pretask
            state.orientation_read_calls = before_orientation
            state.phase = before_phase
            raise
        return self._augment_repo_read_result(raw_result)

    def _augment_repo_read_result(self, raw_result: str) -> str:
        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            return raw_result
        if not isinstance(payload, dict):
            return raw_result
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return raw_result
        state = self._state
        assert isinstance(state, NeonRunState)
        guidance_code = self._repo_read_guidance_code()
        payload["phase"] = state.phase
        if not state.task_written:
            payload["pretask_reads_used"] = state.pretask_read_calls
        else:
            payload["orientation_reads_used"] = state.orientation_read_calls
        return self._steered_result(payload, guidance_code)

    def _repo_read_guidance_code(self) -> str | None:
        state = self._state
        assert isinstance(state, NeonRunState)
        if not state.task_written and state.pretask_read_calls >= MAX_PRETASK_READS:
            return STEERING_PRETASK_LIMIT_REACHED
        if state.task_written and state.route == ANALYZE_SIMPLE_ROUTE and not state.todo_written and state.orientation_read_calls >= MAX_ORIENTATION_READS:
            return STEERING_ORIENTATION_SIMPLE
        if state.task_written and state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and not state.plan_written and state.orientation_read_calls >= MAX_ORIENTATION_READS:
                return STEERING_ORIENTATION_DELEGATED
        return None

    def _next_step_instruction(self, steering_code: str) -> str:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            if steering_code == STEERING_TODO_ITEM_REQUIRED:
                return "Your next action must be start_todo_item for the first pending todo item, then work that item."
            if steering_code == STEERING_READ_OUTPUT:
                return "Your next action must be write_task_artifact with artifact_name='output.md'."
            if steering_code == STEERING_READ_FINDINGS:
                return "Your next action must be write_task_artifact with artifact_name='findings.md'."
            if steering_code == STEERING_WORKER_WRITE_FINDINGS_NOW:
                return (
                    "STOP all other tool calls. All todo items are complete. "
                    "Your ONLY remaining action is write_task_artifact with artifact_name='findings.md'. "
                    "Do not call any other tool. Do not produce a summary. Write findings.md now and stop."
                )
            if steering_code == STEERING_WORKER_WRITE_OUTPUT_NOW:
                return (
                    "STOP all other tool calls. All todo items are complete. "
                    "Your ONLY remaining action is write_task_artifact with artifact_name='output.md'. "
                    "Do not call any other tool. Do not produce a summary. Write output.md now and stop."
                )
            return "Follow the todo sequentially and keep the structured todo state up to date."

        state = self._state
        assert isinstance(state, NeonRunState)
        task_path = self._workspace_manager.artifact_path(state.task_id or state.planned_task_id, "task.md")
        plan_path = self._workspace_manager.artifact_path(state.task_id or state.planned_task_id, "plan.md")
        todo_path = self._workspace_manager.artifact_path(state.task_id or state.planned_task_id, "todo.md")
        if steering_code in {STEERING_PRETASK_LIMIT_REACHED, STEERING_TASK_REQUIRED_NOW}:
            return (
                f"Your next action must be write_task_artifact with artifact_name='task.md'. "
                f"Create it at {task_path}. Do not call list_files, read_file, or search_code again right now."
            )
        if steering_code == STEERING_ORIENTATION_SIMPLE:
            return f"Orientation is done. Your next action must be write_task_artifact with artifact_name='todo.md' at {todo_path}."
        if steering_code == STEERING_ORIENTATION_DELEGATED:
            return (
                f"Orientation is done. Your next action must be write_task_artifact with artifact_name='plan.md' at {plan_path}, "
                f"then write_task_artifact with artifact_name='todo.md' at {todo_path}."
            )
        if steering_code == STEERING_PLAN_REQUIRED:
            return f"Your next action must be write_task_artifact with artifact_name='plan.md' at {plan_path}."
        if steering_code == STEERING_TODO_REQUIRED:
            return f"Your next action must be write_task_artifact with artifact_name='todo.md' at {todo_path}."
        if steering_code == STEERING_TODO_ITEM_REQUIRED:
            return "Your next action must be start_todo_item for the first pending todo item."
        if steering_code == STEERING_SPAWN_WORKER_REQUIRED:
            if state.route == IMPLEMENT_ROUTE:
                return "Your next action must be spawn_implement_worker. Do not read more repo files first."
            return "Your next action must be spawn_analyze_worker. Do not read more repo files first."
        if steering_code == STEERING_READ_FINDINGS:
            return "Your next actions must be read_task_artifact('findings.md') and read_task_artifact('completion.json')."
        if steering_code == STEERING_READ_OUTPUT:
            if self._actor == ACTOR_IMPLEMENT_WORKER:
                return "Your next action must be write_task_artifact with artifact_name='output.md'."
            if state.route == IMPLEMENT_ROUTE:
                return "Your next actions must be read_task_artifact('output.md') and read_task_artifact('completion.json')."
            return "Your next actions must be read_task_artifact('findings.md') and read_task_artifact('completion.json')."
        return "Follow the workflow and use the next required tool now."

    def _repo_reads_available(self) -> bool:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return True
        state = self._state
        assert isinstance(state, NeonRunState)
        if not state.task_written:
            return state.pretask_read_calls < MAX_PRETASK_READS
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.worker_spawned:
            return False
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.todo_written and not state.worker_spawned:
            return False
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.plan_written and not state.todo_written:
            return False
        if state.route == ANALYZE_SIMPLE_ROUTE and not state.todo_written:
            return state.orientation_read_calls < MAX_ORIENTATION_READS
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and not state.plan_written:
            return state.orientation_read_calls < MAX_ORIENTATION_READS
        if state.route == ANALYZE_SIMPLE_ROUTE:
            # Reads are available throughout the todo execution window.
            # _before_repo_read enforces that an item must be in_progress when the call is actually made.
            return not todo_is_complete(state.todo_items)
        return False

    def _repo_summary_available(self) -> bool:
        if self._actor != ACTOR_NEON:
            return False
        state = self._state
        assert isinstance(state, NeonRunState)
        return not state.task_written

    def _write_task_artifact_available(self) -> bool:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return True
        state = self._state
        assert isinstance(state, NeonRunState)
        if not state.task_written:
            return True
        if state.route == ANALYZE_SIMPLE_ROUTE:
            return not state.todo_written
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE}:
            return not state.todo_written or not state.plan_written
        return False

    def _read_task_artifact_available(self) -> bool:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return True
        state = self._state
        assert isinstance(state, NeonRunState)
        return state.task_written

    def _worker_artifact_only_mode(self) -> bool:
        """True when the worker has finished all todos but hasn't written its artifact yet.

        In this state we lock the tool surface to write_task_artifact only,
        preventing the LLM from going off-rail with spurious tool calls.
        """
        if self._actor == ACTOR_ANALYZE_WORKER:
            state = self._state
            assert isinstance(state, AnalyzeWorkerState)
            return todo_is_complete(state.todo_items) and not state.findings_written
        if self._actor == ACTOR_IMPLEMENT_WORKER:
            state = self._state
            assert isinstance(state, ImplementWorkerState)
            return todo_is_complete(state.todo_items) and not state.output_written
        return False

    def _todo_tools_available(self) -> bool:
        if self._actor in {ACTOR_ANALYZE_WORKER, ACTOR_IMPLEMENT_WORKER}:
            return True
        state = self._state
        assert isinstance(state, NeonRunState)
        return state.todo_written and state.route == ANALYZE_SIMPLE_ROUTE and not todo_is_complete(state.todo_items)

    def _workspace_tools_available(self) -> bool:
        return self._repo_reads_available() or self._workspace_mutations_available()

    def _workspace_mutations_available(self) -> bool:
        if self._actor != ACTOR_IMPLEMENT_WORKER:
            return False
        state = self._state
        assert isinstance(state, ImplementWorkerState)
        return self._route == IMPLEMENT_ROUTE and bool(state.todo_items) and not todo_is_complete(state.todo_items)

    def _spawn_worker_available(self) -> bool:
        if self._actor != ACTOR_NEON:
            return False
        state = self._state
        assert isinstance(state, NeonRunState)
        return (
            state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE}
            and state.task_written
            and state.plan_written
            and state.todo_written
            and not state.worker_spawned
        )

    def _repo_doc_available(self) -> bool:
        return self._actor == ACTOR_NEON and not self._state.worker_requested

    def _web_search_available(self) -> bool:
        if self._actor != ACTOR_NEON:
            return False
        state = self._state
        assert isinstance(state, NeonRunState)
        if not state.task_written:
            return False
        if state.route in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE} and state.todo_written and not state.worker_spawned:
            return False
        return True

    def _ensure_workspace_mutations_allowed(self) -> None:
        self._ensure_actor(ACTOR_IMPLEMENT_WORKER)
        state = self._state
        assert isinstance(state, ImplementWorkerState)
        if self._route != IMPLEMENT_ROUTE:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, "Mutation tools are only allowed for implement tasks.")
        if not state.todo_items:
            raise NeonGuardrailError(STEERING_TODO_REQUIRED, "Read the implement todo before mutating files or running commands.")
        if todo_is_complete(state.todo_items):
            raise NeonGuardrailError(STEERING_READ_OUTPUT, "The implement todo is already complete. Write output.md instead.")
        if todo_in_progress(state.todo_items) is None:
            raise NeonGuardrailError(STEERING_TODO_ITEM_REQUIRED, "Start a todo item before mutating files or running commands.")

    def _ensure_not_waiting_for_worker(self) -> None:
        if self._actor == ACTOR_NEON and self._state.worker_requested and not self._state.worker_spawned:
            raise NeonGuardrailError(STEERING_SPAWN_WORKER_REQUIRED, "The delegated worker handoff is already in progress.")

    def _ensure_actor(self, actor: str) -> None:
        if self._actor != actor:
            raise NeonGuardrailError(STEERING_TASK_REQUIRED_NOW, f"{actor} tool used by the wrong actor.")

    def _reject_if_artifact_only(self, tool_name: str) -> None:
        """Raise if the worker must write its artifact and should not call any other tool."""
        if not self._worker_artifact_only_mode():
            return
        if self._actor == ACTOR_ANALYZE_WORKER:
            raise NeonGuardrailError(
                STEERING_WORKER_WRITE_FINDINGS_NOW,
                f"All todo items are done. Do not call {tool_name}. "
                "Your only remaining action is write_task_artifact with artifact_name='findings.md'.",
            )
        if self._actor == ACTOR_IMPLEMENT_WORKER:
            raise NeonGuardrailError(
                STEERING_WORKER_WRITE_OUTPUT_NOW,
                f"All todo items are done. Do not call {tool_name}. "
                "Your only remaining action is write_task_artifact with artifact_name='output.md'.",
            )

    def _ensure_todo_exists(self) -> None:
        if not self._state.task_id or not self._state.todo_items:
            raise NeonGuardrailError(STEERING_TODO_REQUIRED, "No todo state exists yet.")

    @staticmethod
    def _normalize_artifact_name(raw: str) -> str:
        """Normalize artifact names so 'task' → 'task.md', 'completion' → 'completion.json', etc."""
        cleaned = raw.strip().lower()
        KNOWN_ARTIFACTS = {
            "task": "task.md",
            "plan": "plan.md",
            "todo": "todo.md",
            "findings": "findings.md",
            "output": "output.md",
            "completion": "completion.json",
        }
        if cleaned in KNOWN_ARTIFACTS:
            return KNOWN_ARTIFACTS[cleaned]
        return raw.strip()
