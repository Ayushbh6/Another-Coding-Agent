"""
Shared prompt-facing tool documentation helpers.

These helpers render model-facing tool tables from the live registered tool
schemas so prompt docs cannot drift from the actual tool contracts.
"""

from __future__ import annotations

from aca.tools.registry import ToolRegistry


_HIDDEN_PROMPT_PARAMS = {"repo_root", "timeout"}

_PROMPT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_files": "Recursive repo listing with optional glob filtering.",
    "search_repo": "ripgrep full-text search with optional file glob filter.",
    "read_files": "Read one or more file slices in one call. Use a single request for one file.",
    "get_file_outline": "AST/regex structure map for quick orientation before reading code.",
    "search_memory": "Search prior turns, task workspaces, and remembered context.",
    "edit_file": "Primary exact-edit tool. Atomic ordered replacements in one file.",
    "apply_patch": "Unified diff patch for structural or context-sensitive edits.",
    "write_file": "Create or fully replace a file.",
    "delete_file": "Delete a single file.",
    "create_task_workspace": "Create the pinned task workspace for this task.",
    "write_task_file": "Write a file into the current task workspace.",
    "get_next_todo": "Claim the next todo item and mark it in progress.",
    "advance_todo": "Complete or skip the current todo item in order.",
    "move_task_to_archive": "Archive a completed task workspace.",
    "run_command": "Run a shell command. Pipes, chaining, redirection allowed. Destructive patterns blocked.",
    "run_tests": "Run pytest for a path with optional extra arguments.",
}


def tool_signature(
    registry: ToolRegistry,
    tool_name: str,
    *,
    hidden_params: set[str] | None = None,
) -> str:
    """Render a concise prompt-facing signature from the live tool schema."""
    tool = registry.get_tool(tool_name)
    fn_schema = tool.schema["function"]
    params = fn_schema.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))
    hidden = hidden_params if hidden_params is not None else _HIDDEN_PROMPT_PARAMS

    rendered_params: list[str] = []
    for name in properties:
        if name in hidden:
            continue
        rendered_params.append(name if name in required else f"{name}?")
    return f"{fn_schema['name']}({', '.join(rendered_params)})"


def render_tool_table(
    registry: ToolRegistry,
    tool_names: list[str],
    *,
    hidden_params: set[str] | None = None,
) -> str:
    """Render a markdown table for the given tool names."""
    rows = ["| Tool | Description |", "|------|-------------|"]
    for tool_name in tool_names:
        signature = tool_signature(registry, tool_name, hidden_params=hidden_params)
        description = _PROMPT_TOOL_DESCRIPTIONS.get(
            tool_name,
            registry.get_tool(tool_name).schema["function"]["description"],
        )
        rows.append(f"| `{signature}` | {description} |")
    return "\n".join(rows)
