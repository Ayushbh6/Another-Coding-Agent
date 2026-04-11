"""
Tool registry for ACA.

ToolRegistry is the single source of truth for all tools in the system.
It owns:
  - tool registration (callable + JSON schema + metadata)
  - permission-filtered schema generation (Option A: LLM only sees what it's allowed)
  - dispatch (call the tool, log to DB, return structured result)

Usage
-----
Registry is built once at startup by wiring in each tool module:

    from aca.tools.registry import ToolRegistry, PermissionMode
    from aca.tools import read, write, workspace, memory, execution

    registry = ToolRegistry()
    read.register(registry)
    write.register(registry)
    ...

BaseAgent holds the registry instance and calls:
  - registry.get_schemas(mode)   → passed to call_llm as `tools=`
  - registry.dispatch(...)       → called for each tool_call in LLM response
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────────────────

class PermissionMode(str, Enum):
    READ = "read"
    EDIT = "edit"
    FULL = "full"


class ToolCategory(str, Enum):
    READ      = "read"
    WRITE     = "write"
    WORKSPACE = "workspace"
    MEMORY    = "memory"
    EXECUTION = "execution"


# Which categories are available in each permission mode (Option A enforcement)
_MODE_ALLOWED_CATEGORIES: dict[PermissionMode, set[ToolCategory]] = {
    PermissionMode.READ: {ToolCategory.READ, ToolCategory.MEMORY},
    PermissionMode.EDIT: {ToolCategory.READ, ToolCategory.WRITE, ToolCategory.WORKSPACE, ToolCategory.MEMORY},
    PermissionMode.FULL: {ToolCategory.READ, ToolCategory.WRITE, ToolCategory.WORKSPACE, ToolCategory.MEMORY, ToolCategory.EXECUTION},
}


# ── Tool definition ───────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """
    Everything the registry needs to know about a tool.

    fn:       the Python callable that implements the tool
    schema:   the OpenAI-format function schema dict passed to the LLM
    category: used for permission-mode filtering
    """
    fn: Callable[..., Any]
    schema: dict                    # {"type": "function", "function": {...}}
    category: ToolCategory

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


# ── Dispatch result ───────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    output: Any                     # the raw return value from the tool fn
    output_json: str                # JSON-serialised output (used in messages + DB)
    success: bool
    error: str | None
    latency_ms: int
    started_at: int


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all ACA tools.

    Thread-safety: not required (single-threaded CLI process).
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, tool: ToolDefinition) -> None:
        """Add a tool to the registry. Raises if the name is already registered."""
        name = tool.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool

    def register_many(self, tools: list[ToolDefinition]) -> None:
        for tool in tools:
            self.register(tool)

    # ── Schema access (Option A enforcement) ──────────────────────────────────

    def get_schemas(self, mode: PermissionMode) -> list[dict]:
        """
        Return the list of tool schema dicts for the given permission mode.

        Only tools whose category is allowed in `mode` are included.
        The LLM never sees schemas for tools it cannot use.
        """
        allowed = _MODE_ALLOWED_CATEGORIES[mode]
        return [
            t.schema
            for t in self._tools.values()
            if t.category in allowed
        ]

    def get_tool(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Registered tools: {list(self._tools)}")
        return self._tools[name]

    def list_names(self, mode: PermissionMode | None = None) -> list[str]:
        if mode is None:
            return list(self._tools)
        allowed = _MODE_ALLOWED_CATEGORIES[mode]
        return [n for n, t in self._tools.items() if t.category in allowed]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(
        self,
        tool_call: dict,
        *,
        mode: PermissionMode,
        db: Any = None,
        llm_call_id: str | None = None,
        turn_id: str | None = None,
        session_id: str | None = None,
        agent: str | None = None,
    ) -> ToolResult:
        """
        Execute a tool call and return a ToolResult.

        tool_call must be a dict with keys:
          "id"       → the tool_call_id from the LLM response
          "function" → {"name": str, "arguments": str (JSON)}

        Permission is validated here as a secondary guard (primary is schema filtering).
        DB logging is written if db + context fields are all provided.
        """
        tool_call_id = tool_call.get("id", str(uuid.uuid4()))
        fn_block = tool_call.get("function", {})
        tool_name = fn_block.get("name", "")
        raw_args = fn_block.get("arguments", "{}")

        started_at = int(time.time() * 1000)

        # Parse arguments
        try:
            args: dict = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            return self._make_error_result(
                tool_call_id, tool_name, f"Invalid JSON arguments: {exc}", started_at, db,
                llm_call_id, turn_id, session_id, agent,
            )

        # Lookup
        try:
            tool_def = self.get_tool(tool_name)
        except KeyError as exc:
            return self._make_error_result(
                tool_call_id, tool_name, str(exc), started_at, db,
                llm_call_id, turn_id, session_id, agent,
            )

        # Permission guard (secondary — primary is schema-level)
        allowed = _MODE_ALLOWED_CATEGORIES[mode]
        if tool_def.category not in allowed:
            return self._make_error_result(
                tool_call_id, tool_name,
                f"Tool '{tool_name}' (category={tool_def.category.value}) is not permitted in '{mode.value}' mode.",
                started_at, db, llm_call_id, turn_id, session_id, agent,
            )

        # Execute
        try:
            output = tool_def.fn(**args)
            output_json = json.dumps(output, default=str)
            latency_ms = int(time.time() * 1000) - started_at
            result = ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                output=output,
                output_json=output_json,
                success=True,
                error=None,
                latency_ms=latency_ms,
                started_at=started_at,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int(time.time() * 1000) - started_at
            result = ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                output=None,
                output_json="null",
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
                started_at=started_at,
            )

        self._log_tool_call(result, db, llm_call_id, turn_id, session_id, agent, args)
        return result

    def to_tool_message(self, result: ToolResult) -> dict:
        """
        Convert a ToolResult into the message dict that must be appended to the
        conversation history so the LLM sees the tool output.
        """
        content = result.output_json if result.success else json.dumps({"error": result.error})
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": content,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_error_result(
        self,
        tool_call_id: str,
        tool_name: str,
        error: str,
        started_at: int,
        db: Any,
        llm_call_id: str | None,
        turn_id: str | None,
        session_id: str | None,
        agent: str | None,
    ) -> ToolResult:
        latency_ms = int(time.time() * 1000) - started_at
        result = ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            output=None,
            output_json="null",
            success=False,
            error=error,
            latency_ms=latency_ms,
            started_at=started_at,
        )
        self._log_tool_call(result, db, llm_call_id, turn_id, session_id, agent, {})
        return result

    def _log_tool_call(
        self,
        result: ToolResult,
        db: Any,
        llm_call_id: str | None,
        turn_id: str | None,
        session_id: str | None,
        agent: str | None,
        input_args: dict,
    ) -> None:
        if db is None or not all([llm_call_id, turn_id, session_id, agent]):
            return
        try:
            db.execute(
                """
                INSERT INTO tool_calls (
                    tool_call_id, llm_call_id, turn_id, session_id, agent,
                    tool_name, input_json, output_json, success, error,
                    latency_ms, started_at, attempt_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.tool_call_id, llm_call_id, turn_id, session_id, agent,
                    result.tool_name,
                    json.dumps(input_args, default=str),
                    result.output_json,
                    1 if result.success else 0,
                    result.error,
                    result.latency_ms,
                    result.started_at,
                    1,
                ),
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][tool_calls] DB log failed: {exc}")
