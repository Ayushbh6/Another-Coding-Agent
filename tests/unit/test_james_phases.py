from __future__ import annotations

import json
from pathlib import Path

from aca.agents.james import JamesAgent, JamesPhase
from aca.agents.worker import WorkerAgent
from aca.tools import build_registry
from aca.tools.registry import PermissionMode, ToolResult


def _ok_result(tool_name: str, output=None) -> ToolResult:
    return ToolResult(
        tool_call_id=f"call-{tool_name}",
        tool_name=tool_name,
        output=output or {},
        output_json=json.dumps(output or {}),
        success=True,
        error=None,
        latency_ms=1,
        started_at=0,
    )


def _build_james(tmp_path: Path) -> JamesAgent:
    return JamesAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        repo_context="Repo path: test",
        thinking=False,
        stream=False,
    )


def test_orient_budget_switches_james_to_task_create(tmp_path: Path) -> None:
    james = _build_james(tmp_path)

    for _ in range(3):
        james._on_tool_completed("read_file", "{}", _ok_result("read_file"), routed=False)

    tools, _tool_choice, current_mode = james._extra_pre_llm_steering(
        messages=[],
        tools=[],
        tool_choice=None,
        current_mode=PermissionMode.READ,
        routing_tools_used=0,
        routed=False,
        tool_calls_this_turn=0,
    )

    assert james._phase == JamesPhase.TASK_CREATE
    assert current_mode == PermissionMode.EDIT
    tool_names = {schema["function"]["name"] for schema in tools or []}
    assert tool_names == {"create_task_workspace", "write_task_file"}


def test_prepare_tool_call_pins_task_id_and_normalizes_task_md(tmp_path: Path) -> None:
    james = _build_james(tmp_path)
    james._phase = JamesPhase.TASK_CREATE

    tc = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "write_task_file",
            "arguments": json.dumps(
                {
                    "task_id": "wrong-id",
                    "filename": "task.md",
                    "content": "type: implement\ndelegation: simple\n\n# Task: Write docs\n\n## Request\n\nWrite the docs.\n",
                }
            ),
        },
    }

    prepared = james._prepare_tool_call(tc, turn_id="turn-1", current_mode=PermissionMode.EDIT)
    args = json.loads(prepared["function"]["arguments"])

    assert args["task_id"].startswith("task-")
    assert args["task_id"] != "wrong-id"
    assert "session_id: session-123" in args["content"]
    assert "turn_id: turn-1" in args["content"]
    assert "type: implement" in args["content"]
    assert "delegation: simple" in args["content"]
    assert "# Task: Write docs" in args["content"]
    assert "## Context" in args["content"]
    assert "## Scope" in args["content"]


def test_valid_task_md_sets_route_and_phase(tmp_path: Path) -> None:
    james = _build_james(tmp_path)
    james._james_task_id = "task-001"
    james._phase = JamesPhase.TASK_CREATE

    content = (
        "task_id: task-001\n"
        "type: analysis\n"
        "delegation: delegated\n"
        "status: active\n"
        "created_at: 2026-04-13T00:00:00Z\n"
        "session_id: session-123\n"
        "turn_id: turn-1\n\n"
        "# Task: Analyze repo\n\n"
        "## Request\n\nAnalyze the repo.\n\n"
        "## Context\n\nInspected docs/ARCHITECTURE.md.\n\n"
        "## Scope\n\nIn scope: architecture review. Out of scope: implementation.\n"
    )

    james._on_tool_completed(
        "write_task_file",
        json.dumps({"task_id": "task-001", "filename": "task.md", "content": content}),
        _ok_result("write_task_file"),
        routed=False,
    )

    assert james._route == "analysis_delegated"
    assert james._james_needs_plan is True
    assert james._james_task_md_written is True
    assert james._phase == JamesPhase.TASK_ARTIFACTS


def test_invalid_task_md_sets_rewrite_steering_and_restricts_tools(tmp_path: Path) -> None:
    james = _build_james(tmp_path)
    james._james_task_id = "task-001"
    james._phase = JamesPhase.TASK_CREATE

    invalid_content = (
        "task_id: task-001\n"
        "type: analysis\n"
        "delegation: simple\n"
        "status: active\n"
        "created_at: 2026-04-13T00:00:00Z\n"
        "session_id: session-123\n"
        "turn_id: turn-1\n\n"
        "## Request\n\nMissing the title and scope.\n"
    )

    james._on_tool_completed(
        "write_task_file",
        json.dumps({"task_id": "task-001", "filename": "task.md", "content": invalid_content}),
        _ok_result("write_task_file"),
        routed=False,
    )

    tools, _tool_choice, current_mode = james._extra_pre_llm_steering(
        messages=[],
        tools=[],
        tool_choice=None,
        current_mode=PermissionMode.EDIT,
        routing_tools_used=0,
        routed=False,
        tool_calls_this_turn=0,
    )

    assert james._james_task_md_written is False
    assert current_mode == PermissionMode.EDIT
    assert {schema["function"]["name"] for schema in tools or []} == {"read_file", "write_task_file"}


def test_task_artifact_phase_restricts_reads_to_templates_or_workspace(tmp_path: Path) -> None:
    james = _build_james(tmp_path)
    james._phase = JamesPhase.TASK_ARTIFACTS
    james._james_task_id = "task-001"
    workspace = tmp_path / ".aca" / "active" / "task-001"
    workspace.mkdir(parents=True)

    err = james._validate_tool_call_args(
        "read_file",
        {"path": "aca/cli.py"},
        turn_id="turn-1",
        current_mode=PermissionMode.EDIT,
    )
    assert err is not None

    ok = james._validate_tool_call_args(
        "read_file",
        {"path": str(workspace / "task.md")},
        turn_id="turn-1",
        current_mode=PermissionMode.EDIT,
    )
    assert ok is None


def test_worker_stops_after_writing_result_file(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        thinking=False,
        stream=False,
    )

    worker._on_tool_completed(
        "write_task_file",
        json.dumps({"task_id": "task-001", "filename": "findings.md", "content": "done"}),
        _ok_result("write_task_file"),
        routed=False,
    )

    tools, tool_choice, current_mode = worker._extra_pre_llm_steering(
        messages=[],
        tools=[],
        tool_choice=None,
        current_mode=PermissionMode.FULL,
        routing_tools_used=0,
        routed=False,
        tool_calls_this_turn=3,
    )

    assert tools is None
    assert tool_choice == "none"
    assert current_mode == PermissionMode.FULL


def test_worker_passes_through_tools_before_result_file(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        current_task_id="task-001",
        thinking=False,
        stream=False,
    )

    original_tools = [{"type": "function", "function": {"name": "read_file"}}]
    tools, tool_choice, current_mode = worker._extra_pre_llm_steering(
        messages=[],
        tools=original_tools,
        tool_choice=None,
        current_mode=PermissionMode.FULL,
        routing_tools_used=0,
        routed=False,
        tool_calls_this_turn=0,
    )

    assert tools == original_tools
    assert tool_choice is None
    assert current_mode == PermissionMode.FULL


def test_worker_prepare_tool_call_pins_workspace_task_id(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        current_task_id="task-001",
        thinking=False,
        stream=False,
    )

    tc = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "advance_todo",
            "arguments": json.dumps({"task_id": "wrong-id", "item_index": 0, "action": "complete"}),
        },
    }

    prepared = worker._prepare_tool_call(tc, turn_id="turn-1", current_mode=PermissionMode.FULL)
    args = json.loads(prepared["function"]["arguments"])
    assert args["task_id"] == "task-001"


def test_worker_blocks_writing_non_result_artifacts(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        current_task_id="task-001",
        thinking=False,
        stream=False,
    )

    err = worker._validate_tool_call_args(
        "write_task_file",
        {"task_id": "task-001", "filename": "plan.md", "content": "nope"},
        turn_id="turn-1",
        current_mode=PermissionMode.FULL,
    )
    assert err is not None


def test_worker_blocks_reading_other_task_workspace(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        current_task_id="task-001",
        thinking=False,
        stream=False,
    )
    current = tmp_path / ".aca" / "active" / "task-001"
    other = tmp_path / ".aca" / "active" / "task-002"
    current.mkdir(parents=True)
    other.mkdir(parents=True)

    err = worker._validate_tool_call_args(
        "read_file",
        {"path": str(other / "task.md")},
        turn_id="turn-1",
        current_mode=PermissionMode.FULL,
    )
    assert err is not None

    ok = worker._validate_tool_call_args(
        "read_file",
        {"path": str(current / "task.md")},
        turn_id="turn-1",
        current_mode=PermissionMode.FULL,
    )
    assert ok is None


def test_worker_blocks_listing_other_task_workspace(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        current_task_id="task-001",
        thinking=False,
        stream=False,
    )
    (tmp_path / ".aca" / "active" / "task-001").mkdir(parents=True)
    other = tmp_path / ".aca" / "active" / "task-999"
    other.mkdir(parents=True)

    err = worker._validate_tool_call_args(
        "list_files",
        {"path": str(other)},
        turn_id="turn-1",
        current_mode=PermissionMode.FULL,
    )
    assert err is not None


def test_worker_blocks_workspace_tools_for_other_task_id(tmp_path: Path) -> None:
    worker = WorkerAgent(
        registry=build_registry(),
        session_id="session-123",
        repo_root=str(tmp_path),
        current_task_id="task-001",
        thinking=False,
        stream=False,
    )

    err = worker._validate_tool_call_args(
        "get_next_todo",
        {"task_id": "task-002"},
        turn_id="turn-1",
        current_mode=PermissionMode.FULL,
    )
    assert err is not None
