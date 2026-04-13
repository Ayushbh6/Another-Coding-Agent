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
import inspect
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from pathlib import Path
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
    SYSTEM    = "system"   # system-only tools — never exposed to agents via get_schemas()


# Which categories are available in each permission mode (Option A enforcement).
# ToolCategory.SYSTEM is intentionally excluded from ALL modes — system-managed
# tools (e.g. compact_context) are invoked by the runtime, never by agents.
_MODE_ALLOWED_CATEGORIES: dict[PermissionMode, set[ToolCategory]] = {
    PermissionMode.READ: {ToolCategory.READ, ToolCategory.MEMORY},
    PermissionMode.EDIT: {ToolCategory.READ, ToolCategory.WRITE, ToolCategory.WORKSPACE, ToolCategory.MEMORY},
    PermissionMode.FULL: {ToolCategory.READ, ToolCategory.WRITE, ToolCategory.WORKSPACE, ToolCategory.MEMORY, ToolCategory.EXECUTION},
}


# ── Example-guidelines read guard ─────────────────────────────────────────────
# ~/.aca/example_guidelines/ is a global read-only install, shared across all
# repos. Each agent may only read the subset of files relevant to their role.

_GLOBAL_ACA_GUIDELINES: Path = Path.home() / ".aca" / "example_guidelines"

# agent name → frozenset of permitted filenames within example_guidelines/
_EXAMPLE_GUIDELINES_ALLOWED: dict[str, frozenset[str]] = {
    "james":      frozenset({"task.md", "plan.md", "todo.md"}),
    "worker":     frozenset({"findings.md", "output.md"}),
    "challenger": frozenset(),   # no guidelines relevant to the challenger role
}


# ── Gitignore helper ─────────────────────────────────────────────────────────

def _is_gitignored(path: Path, repo_root: Path) -> bool:
    """
    Return True if `path` is ignored by git in `repo_root`.

    Uses ``git check-ignore -q <path>`` — exit 0 means ignored.
    Falls back to False (fail-open) if git is unavailable or repo has no .git.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "-q", str(path)],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False  # git unavailable or not a git repo — fail open


def _check_example_guidelines_read(
    path_str: str,
    repo_root_str: str,
    agent: str | None,
) -> str | None:
    """
    Return an error string if this read_file call is blocked, or None if allowed.

    Checks both absolute paths (e.g. /Users/foo/.aca/example_guidelines/task.md)
    and repo-relative paths in case an agent uses a relative reference.

    A read into ~/.aca/example_guidelines/ is allowed only when:
      1. The agent name is known and listed in _EXAMPLE_GUIDELINES_ALLOWED.
      2. The specific filename is in that agent's permitted set.
    """
    eg_dir = _GLOBAL_ACA_GUIDELINES.resolve()

    # Resolve the path: try absolute first, then relative to repo_root
    p = Path(path_str)
    if p.is_absolute():
        target = p.resolve()
    else:
        target = (Path(repo_root_str) / path_str).resolve()

    try:
        target.relative_to(eg_dir)   # raises ValueError if not inside eg_dir
    except ValueError:
        return None  # path is not under example_guidelines/ — no restriction

    filename = target.name
    agent_key = (agent or "").lower()
    allowed_files = _EXAMPLE_GUIDELINES_ALLOWED.get(agent_key, frozenset())

    if filename not in allowed_files:
        if not allowed_files:
            return (
                f"Agent '{agent}' is not permitted to read any files from "
                f"'~/.aca/example_guidelines/'. That folder is reserved for James "
                "(task.md, plan.md, todo.md) and Worker (findings.md, output.md)."
            )
        return (
            f"Agent '{agent}' is not permitted to read '{filename}' from "
            f"'~/.aca/example_guidelines/'. Allowed files for this agent: "
            f"{sorted(allowed_files)}."
        )
    return None  # allowed


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
        # Absolute path strings the user has explicitly unlocked for reading
        # (gitignored files are blocked by default; /allow <path> adds here).
        self._read_allowlist: set[str] = set()

    def add_to_read_allowlist(self, path_str: str, repo_root: str | Path = ".") -> str:
        """
        Add a specific path to the per-session read allowlist.

        Returns the resolved absolute path string so the caller can confirm
        what was actually allowed.  Path must be inside the repo or absolute.
        """
        p = Path(path_str)
        if p.is_absolute():
            resolved = str(p.resolve())
        else:
            resolved = str((Path(repo_root) / path_str).resolve())
        self._read_allowlist.add(resolved)
        return resolved

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

    def get_schemas_for_names(self, names: set[str] | list[str]) -> list[dict]:
        """Return schemas for the explicit tool names in declaration order."""
        allowed_names = set(names)
        return [
            t.schema
            for t in self._tools.values()
            if t.name in allowed_names
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
        allowed_tool_names: set[str] | None = None,
        injected_kwargs: dict[str, Any] | None = None,
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

        if allowed_tool_names is not None and tool_name not in allowed_tool_names:
            return self._make_error_result(
                tool_call_id,
                tool_name,
                f"Tool '{tool_name}' is not exposed in the current runtime phase.",
                started_at,
                db,
                llm_call_id,
                turn_id,
                session_id,
                agent,
            )

        # Example-guidelines read guard (agent-scoped, read-only reference files)
        if tool_name == "read_file":
            path_arg = args.get("path", "")
            repo_root_arg = args.get("repo_root", ".")
            eg_error = _check_example_guidelines_read(path_arg, repo_root_arg, agent)
            if eg_error:
                return self._make_error_result(
                    tool_call_id, tool_name, eg_error, started_at, db,
                    llm_call_id, turn_id, session_id, agent,
                )

        # Gitignore guard — applies to read_file and get_file_outline.
        # If the file is gitignored, block it unless the user has explicitly
        # added it to the session allowlist via /allow.
        if tool_name in ("read_file", "get_file_outline"):
            path_arg = args.get("path", "")
            repo_root_arg = args.get("repo_root", ".")
            p = Path(path_arg)
            resolved_path = (
                p.resolve() if p.is_absolute()
                else (Path(repo_root_arg) / path_arg).resolve()
            )
            if str(resolved_path) not in self._read_allowlist:
                # Fast-path: check if any path component is in the always-excluded set.
                # Import here to avoid a circular import at module level.
                from aca.tools.read import _ALWAYS_EXCLUDE_DIRS
                path_parts = set(resolved_path.parts)
                if path_parts & _ALWAYS_EXCLUDE_DIRS:
                    return self._make_error_result(
                        tool_call_id, tool_name,
                        f"'{path_arg}' is inside a directory that is always excluded "
                        f"({path_parts & _ALWAYS_EXCLUDE_DIRS}). "
                        "Use /allow <path> to explicitly unlock a specific file for this session.",
                        started_at, db, llm_call_id, turn_id, session_id, agent,
                    )
                if _is_gitignored(resolved_path, Path(repo_root_arg).resolve()):
                    return self._make_error_result(
                        tool_call_id, tool_name,
                        f"'{path_arg}' is listed in .gitignore and cannot be read. "
                        "If you need to share this file with ACA, use /allow <path> "
                        "in the terminal to explicitly unlock it for this session.",
                        started_at, db, llm_call_id, turn_id, session_id, agent,
                    )

        # Execute
        try:
            call_args = dict(args)
            if injected_kwargs:
                accepted_params = set(inspect.signature(tool_def.fn).parameters)
                for key, value in injected_kwargs.items():
                    if key in accepted_params:
                        call_args[key] = value
            output = tool_def.fn(**call_args)
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
