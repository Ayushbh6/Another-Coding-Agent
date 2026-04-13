"""
Unit tests for aca/tools/registry.py

Tests cover:
  - registration (normal, duplicate detection)
  - permission-mode schema filtering (Option A)
  - dispatch (success, permission guard, bad tool name, bad JSON args)
  - to_tool_message output shape
"""

import json
import pytest

from aca.tools.registry import (
    PermissionMode,
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tool(name: str, category: ToolCategory, return_value=None):
    """Return a ToolDefinition with a trivial callable."""
    def fn(**kwargs):
        return return_value if return_value is not None else {"called": name, "kwargs": kwargs}

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": "param x"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }
    return ToolDefinition(fn=fn, schema=schema, category=category)


def _make_tool_call(name: str, args: dict) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ── Registration ──────────────────────────────────────────────────────────────

class TestRegistration:
    def test_register_single_tool(self):
        registry = ToolRegistry()
        tool = _make_tool("my_tool", ToolCategory.READ)
        registry.register(tool)
        assert "my_tool" in registry.list_names()

    def test_register_many(self):
        registry = ToolRegistry()
        tools = [_make_tool(f"t{i}", ToolCategory.READ) for i in range(5)]
        registry.register_many(tools)
        assert len(registry.list_names()) == 5

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        tool = _make_tool("dup", ToolCategory.READ)
        registry.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_get_tool_returns_definition(self):
        registry = ToolRegistry()
        tool = _make_tool("finder", ToolCategory.READ)
        registry.register(tool)
        fetched = registry.get_tool("finder")
        assert fetched.name == "finder"

    def test_get_tool_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="unknown_tool"):
            registry.get_tool("unknown_tool")


# ── Permission-mode schema filtering (Option A) ───────────────────────────────

class TestPermissionFiltering:
    def _populated_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(_make_tool("read_t",      ToolCategory.READ))
        registry.register(_make_tool("memory_t",    ToolCategory.MEMORY))
        registry.register(_make_tool("write_t",     ToolCategory.WRITE))
        registry.register(_make_tool("workspace_t", ToolCategory.WORKSPACE))
        registry.register(_make_tool("exec_t",      ToolCategory.EXECUTION))
        return registry

    def test_read_mode_only_read_and_memory(self):
        registry = self._populated_registry()
        schemas = registry.get_schemas(PermissionMode.READ)
        names = {s["function"]["name"] for s in schemas}
        assert names == {"read_t", "memory_t"}

    def test_edit_mode_excludes_execution(self):
        registry = self._populated_registry()
        schemas = registry.get_schemas(PermissionMode.EDIT)
        names = {s["function"]["name"] for s in schemas}
        assert "exec_t" not in names
        assert {"read_t", "memory_t", "write_t", "workspace_t"} <= names

    def test_full_mode_includes_all(self):
        registry = self._populated_registry()
        schemas = registry.get_schemas(PermissionMode.FULL)
        names = {s["function"]["name"] for s in schemas}
        assert names == {"read_t", "memory_t", "write_t", "workspace_t", "exec_t"}

    def test_list_names_with_mode(self):
        registry = self._populated_registry()
        read_names = registry.list_names(PermissionMode.READ)
        assert "exec_t" not in read_names
        assert "write_t" not in read_names

    def test_get_schemas_returns_correct_schema_shape(self):
        registry = ToolRegistry()
        registry.register(_make_tool("t", ToolCategory.READ))
        schemas = registry.get_schemas(PermissionMode.READ)
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert "name" in schemas[0]["function"]


# ── Dispatch ──────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_successful_dispatch(self):
        registry = ToolRegistry()
        registry.register(_make_tool("adder", ToolCategory.READ, return_value={"sum": 42}))
        tc = _make_tool_call("adder", {"x": "hello"})
        result = registry.dispatch(tc, mode=PermissionMode.READ)
        assert result.success is True
        assert result.tool_name == "adder"
        assert json.loads(result.output_json) == {"sum": 42}
        assert result.error is None
        assert result.latency_ms >= 0

    def test_dispatch_passes_args_to_fn(self):
        captured = {}
        def fn(**kwargs):
            captured.update(kwargs)
            return {}
        schema = {
            "type": "function",
            "function": {
                "name": "capture",
                "description": "captures args",
                "parameters": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
        registry = ToolRegistry()
        registry.register(ToolDefinition(fn=fn, schema=schema, category=ToolCategory.READ))
        tc = _make_tool_call("capture", {"val": "test_value"})
        registry.dispatch(tc, mode=PermissionMode.READ)
        assert captured == {"val": "test_value"}

    def test_dispatch_unknown_tool_returns_error_result(self):
        registry = ToolRegistry()
        tc = _make_tool_call("ghost", {})
        result = registry.dispatch(tc, mode=PermissionMode.FULL)
        assert result.success is False
        assert "ghost" in result.error

    def test_dispatch_permission_denied_returns_error_result(self):
        registry = ToolRegistry()
        registry.register(_make_tool("exec_only", ToolCategory.EXECUTION))
        tc = _make_tool_call("exec_only", {})
        result = registry.dispatch(tc, mode=PermissionMode.READ)
        assert result.success is False
        assert "not permitted" in result.error

    def test_dispatch_invalid_json_args_returns_error_result(self):
        registry = ToolRegistry()
        registry.register(_make_tool("t", ToolCategory.READ))
        tc = {
            "id": "bad",
            "type": "function",
            "function": {"name": "t", "arguments": "NOT_JSON"},
        }
        result = registry.dispatch(tc, mode=PermissionMode.READ)
        assert result.success is False
        assert "JSON" in result.error

    def test_dispatch_fn_exception_returns_error_result(self):
        def boom(**kwargs):
            raise RuntimeError("boom!")
        schema = {
            "type": "function",
            "function": {
                "name": "exploder",
                "description": "always raises",
                "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            },
        }
        registry = ToolRegistry()
        registry.register(ToolDefinition(fn=boom, schema=schema, category=ToolCategory.READ))
        tc = _make_tool_call("exploder", {})
        result = registry.dispatch(tc, mode=PermissionMode.READ)
        assert result.success is False
        assert "boom!" in result.error


# ── to_tool_message ───────────────────────────────────────────────────────────

class TestToToolMessage:
    def test_success_message_shape(self):
        registry = ToolRegistry()
        registry.register(_make_tool("t", ToolCategory.READ, return_value={"ok": True}))
        tc = _make_tool_call("t", {})
        result = registry.dispatch(tc, mode=PermissionMode.READ)
        msg = registry.to_tool_message(result)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_t"
        assert json.loads(msg["content"]) == {"ok": True}

    def test_error_message_contains_error(self):
        registry = ToolRegistry()
        tc = _make_tool_call("nonexistent", {})
        result = registry.dispatch(tc, mode=PermissionMode.FULL)
        msg = registry.to_tool_message(result)
        assert msg["role"] == "tool"
        parsed = json.loads(msg["content"])
        assert "error" in parsed


# ── example_guidelines read guard (via registry.dispatch) ────────────────────

import pytest
from pathlib import Path as _Path
from aca.tools import read as read_module

_GLOBAL_EG = _Path.home() / ".aca" / "example_guidelines"


def _make_registry_with_read_file():
    """Build a minimal registry containing only the real read_file tool."""
    registry = ToolRegistry()
    read_module.register(registry)
    return registry


class TestExampleGuidelinesReadGuard:
    """
    The registry blocks read_file calls into ~/.aca/example_guidelines/ based
    on the calling agent. James: task/plan/todo. Worker: findings/output.
    All others: nothing.

    Tests use the real global path (files already live there after install).
    """

    def _dispatch_read(self, registry, filename, agent):
        path = str(_GLOBAL_EG / filename)
        tc = _make_tool_call("read_file", {"path": path, "repo_root": "."})
        return registry.dispatch(tc, mode=PermissionMode.READ, agent=agent)

    # ── James permitted files ─────────────────────────────────────────────────

    def test_james_can_read_task_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "task.md", "james")
        assert result.success, result.error

    def test_james_can_read_plan_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "plan.md", "james")
        assert result.success, result.error

    def test_james_can_read_todo_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "todo.md", "james")
        assert result.success, result.error

    # ── James blocked files ───────────────────────────────────────────────────

    def test_james_cannot_read_findings_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "findings.md", "james")
        assert not result.success
        assert "example_guidelines" in (result.error or "") or "not permitted" in (result.error or "")

    def test_james_cannot_read_output_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "output.md", "james")
        assert not result.success

    # ── Worker permitted files ────────────────────────────────────────────────

    def test_worker_can_read_findings_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "findings.md", "worker")
        assert result.success, result.error

    def test_worker_can_read_output_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "output.md", "worker")
        assert result.success, result.error

    # ── Worker blocked files ──────────────────────────────────────────────────

    def test_worker_cannot_read_task_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "task.md", "worker")
        assert not result.success

    def test_worker_cannot_read_plan_md(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "plan.md", "worker")
        assert not result.success

    # ── Unknown agents / challenger ───────────────────────────────────────────

    def test_challenger_cannot_read_any_guidelines(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "task.md", "challenger")
        assert not result.success
        assert "reserved" in (result.error or "") or "not permitted" in (result.error or "")

    def test_unknown_agent_cannot_read_guidelines(self):
        result = self._dispatch_read(_make_registry_with_read_file(), "task.md", "unknown_bot")
        assert not result.success

    def test_no_agent_cannot_read_guidelines(self):
        path = str(_GLOBAL_EG / "task.md")
        tc = _make_tool_call("read_file", {"path": path, "repo_root": "."})
        result = _make_registry_with_read_file().dispatch(tc, mode=PermissionMode.READ, agent=None)
        assert not result.success

    # ── Normal files outside guidelines are unaffected ────────────────────────

    def test_normal_read_unaffected_by_guard(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hi\n")
        tc = _make_tool_call(
            "read_file", {"path": "hello.txt", "repo_root": str(tmp_path)}
        )
        result = _make_registry_with_read_file().dispatch(tc, mode=PermissionMode.READ, agent=None)
        assert result.success
