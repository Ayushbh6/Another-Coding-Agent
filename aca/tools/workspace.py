"""
Workspace tools for ACA.

Manage the .aca/active/<task-id>/ workspace per TOOLS_AND_PERMISSIONS.md §9.

Rules enforced here (not left to agent discretion):
  - task subfolders only created via create_task_workspace()
  - artifact writes only via write_task_file() with permitted filename validation
  - general write tools are blocked from .aca/ (enforced in write.py)
  - move_task_to_archive() is the only way to retire a workspace
  - todo step sequencing enforced by get_next_todo() / advance_todo()

Requires PermissionMode.EDIT or above.

Register into a ToolRegistry via:
    from aca.tools import workspace
    workspace.register(registry)
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from aca.tools.registry import ToolCategory, ToolDefinition, ToolRegistry


# ── Constants ─────────────────────────────────────────────────────────────────

_ACA_ACTIVE = ".aca/active"
_EXAMPLE_GUIDELINES_DIR = ".aca/example_guidelines"
_GLOBAL_GUIDELINES = Path.home() / ".aca" / "example_guidelines"

# Permitted artifact filenames per TOOLS_AND_PERMISSIONS.md §9.3
_PERMITTED_FILENAMES = {"task.md", "plan.md", "todo.md", "findings.md", "output.md"}

# Todo item state markers
_PENDING   = "[ ]"
_DONE      = "[x]"
_INPROG    = "[>]"
_SKIPPED   = "[~]"

# Regex that matches a todo list line: captures (marker, rest-of-text)
_TODO_LINE_RE = re.compile(r"^- (\[[ x>~]\])(.*)", re.MULTILINE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_dir(repo_root: Path) -> Path:
    return (repo_root / _ACA_ACTIVE).resolve()


def _task_dir(repo_root: Path, task_id: str) -> Path:
    return _active_dir(repo_root) / task_id


def _guard_task_id(task_id: str) -> None:
    """Reject task IDs that could be used for path traversal."""
    if not task_id or "/" in task_id or "\\" in task_id or task_id.startswith("."):
        raise ValueError(
            f"Invalid task_id '{task_id}'. "
            "task_id must be a simple alphanumeric slug (e.g. 'task-001')."
        )


def _guard_not_example_guidelines(target: Path, repo_root: Path) -> None:
    """Hard-block any write targeting example_guidelines/ (global or repo-local)."""
    global_eg = _GLOBAL_GUIDELINES.resolve()
    try:
        target.relative_to(global_eg)
        raise ValueError(
            f"'{target.name}' resolves into '~/.aca/example_guidelines/' which is "
            "read-only. These files are global format reference templates and must "
            "never be modified."
        )
    except ValueError as exc:
        if "read-only" in str(exc):
            raise

    repo_eg = (repo_root / _EXAMPLE_GUIDELINES_DIR).resolve()
    try:
        target.relative_to(repo_eg)
        raise ValueError(
            f"'{target.name}' is inside '.aca/example_guidelines/' which is read-only. "
            "These files are format reference templates and must never be modified."
        )
    except ValueError as exc:
        if "read-only" in str(exc):
            raise


def _guard_filename(filename: str) -> None:
    """Ensure the filename is on the permitted list."""
    if filename not in _PERMITTED_FILENAMES:
        raise ValueError(
            f"'{filename}' is not a permitted artifact filename. "
            f"Allowed: {sorted(_PERMITTED_FILENAMES)}"
        )


def _load_todo(task_path: Path) -> str:
    """Read todo.md; raise FileNotFoundError if absent."""
    todo_file = task_path / "todo.md"
    if not todo_file.exists():
        raise FileNotFoundError(
            "todo.md not found in task workspace. "
            "Write todo.md via write_task_file before calling todo tools."
        )
    return todo_file.read_text(encoding="utf-8")


def _save_todo(task_path: Path, content: str) -> None:
    (task_path / "todo.md").write_text(content, encoding="utf-8")


def _parse_items(content: str) -> list[tuple[int, str, str]]:
    """
    Return a list of (line_index, marker, text) for every todo item line.
    line_index is the 0-based index in content.splitlines().
    """
    lines = content.splitlines()
    items = []
    for i, line in enumerate(lines):
        m = _TODO_LINE_RE.match(line)
        if m:
            items.append((i, m.group(1), m.group(2).strip()))
    return items


def _update_current_step(content: str, text: str | None) -> str:
    """Replace the text under ## Current step with `text`, or 'All items complete.'"""
    replacement = text if text else "All items complete."
    # Replace everything after '## Current step' up to the next heading or EOF
    new_content = re.sub(
        r"(## Current step\s*\n)([^\#]*)",
        lambda m: m.group(1) + replacement + "\n",
        content,
        flags=re.DOTALL,
    )
    # If the section didn't exist, append it
    if "## Current step" not in new_content:
        new_content = new_content.rstrip() + f"\n\n## Current step\n{replacement}\n"
    return new_content


# ── Tool implementations ──────────────────────────────────────────────────────

def create_task_workspace(
    task_id: str | None = None,
    repo_root: str = ".",
    db: Any = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> dict:
    """
    Create the .aca/active/<task-id>/ folder for a new task.

    Also writes a DB row to the `tasks` table if db + session_id + turn_id are provided.
    Returns {"task_id": str, "workspace_path": str, "created": bool}
    """
    if not task_id:
        task_id = f"task-{uuid.uuid4().hex[:8]}"
    _guard_task_id(task_id)
    root = Path(repo_root).resolve()
    task_path = _task_dir(root, task_id)

    already_existed = task_path.exists()
    task_path.mkdir(parents=True, exist_ok=True)

    workspace_path = str(task_path.relative_to(root))

    if db is not None and session_id and turn_id and not already_existed:
        try:
            db.execute(
                """
                INSERT INTO tasks (
                    task_id, session_id, turn_id, task_type, delegation,
                    title, status, created_at, workspace_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, session_id, turn_id,
                    "unknown", "unknown",   # caller should update after creating task.md
                    task_id, "active",
                    int(time.time() * 1000),
                    workspace_path,
                ),
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][workspace] DB log failed for create_task_workspace: {exc}")

    return {
        "task_id": task_id,
        "workspace_path": workspace_path,
        "created": not already_existed,
    }


def write_task_file(
    task_id: str,
    filename: str,
    content: str,
    repo_root: str = ".",
) -> dict:
    """
    Write or overwrite an artifact file inside .aca/active/<task-id>/.

    filename must be one of: task.md, plan.md, todo.md, findings.md, output.md
    The task workspace must already exist (call create_task_workspace first).

    Returns {"task_id": str, "filename": str, "bytes_written": int, "action": str}
    """
    _guard_task_id(task_id)
    _guard_filename(filename)

    root = Path(repo_root).resolve()
    task_path = _task_dir(root, task_id)

    if not task_path.exists():
        raise FileNotFoundError(
            f"Task workspace '{task_id}' does not exist. "
            "Call create_task_workspace first."
        )

    target = task_path / filename
    # Belt-and-suspenders: reject if resolved path lands in example_guidelines/
    _guard_not_example_guidelines(target, root)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")

    return {
        "task_id": task_id,
        "filename": filename,
        "path": str(target.relative_to(root)),
        "bytes_written": len(content.encode("utf-8")),
        "action": "overwritten" if existed else "created",
    }


def get_next_todo(task_id: str, repo_root: str = ".") -> dict:
    """
    Get the next pending todo item and mark it as in-progress [>].

    Reads todo.md, finds the first pending item (- [ ]), marks it [>],
    updates ## Current step, and returns the item text and metadata.

    If a [>] item already exists (resumed turn), returns it without re-marking.
    If all items are done or skipped, returns {"all_done": true}.

    The agent must complete or skip the returned item via advance_todo
    before calling get_next_todo again.
    """
    _guard_task_id(task_id)
    root = Path(repo_root).resolve()
    task_path = _task_dir(root, task_id)
    content = _load_todo(task_path)
    items = _parse_items(content)

    # If there's already an in-progress item, return it (idempotent resume)
    for idx, (line_i, marker, text) in enumerate(items):
        if marker == _INPROG:
            pending_count = sum(1 for _, m, _ in items if m == _PENDING)
            return {
                "item": text,
                "index": idx,
                "status": "resumed",
                "remaining": pending_count,
                "all_done": False,
            }

    # Find the first pending item
    first_pending = next(((idx, li, t) for idx, (li, m, t) in enumerate(items) if m == _PENDING), None)

    if first_pending is None:
        return {"all_done": True, "item": None, "index": None, "remaining": 0}

    item_idx, line_i, item_text = first_pending

    # Mark it [>] in-progress
    lines = content.splitlines(keepends=True)
    lines[line_i] = lines[line_i].replace("- [ ]", "- [>]", 1)
    new_content = "".join(lines)
    new_content = _update_current_step(new_content, item_text)
    _save_todo(task_path, new_content)

    pending_count = sum(1 for _, m, _ in items if m == _PENDING) - 1  # we just claimed one

    return {
        "item": item_text,
        "index": item_idx,
        "status": "started",
        "remaining": pending_count,
        "all_done": False,
    }


def advance_todo(
    task_id: str,
    item_index: int,
    action: str,
    skip_reason: str = "",
    repo_root: str = ".",
) -> dict:
    """
    Mark the current in-progress todo item as complete [x] or skipped [~],
    then automatically start the next pending item [>].

    action must be "complete" or "skip".
    skip_reason is REQUIRED when action="skip" — must explain specifically why
    the item was already handled. Vague reasons are not accepted.

    Gate: item_index must be the current [>] item. Attempting to advance any
    other index returns an error — no jumping ahead.

    Returns the next item info (or all_done=true when the list is finished).
    """
    if action not in ("complete", "skip"):
        raise ValueError(f"action must be 'complete' or 'skip', got '{action}'.")

    if action == "skip":
        if not skip_reason or not skip_reason.strip():
            raise ValueError(
                "skip_reason is required when action='skip'. "
                "Provide a specific explanation of why this item was already handled. "
                "Do not skip items without a concrete reason."
            )
        if len(skip_reason.strip()) < 20:
            raise ValueError(
                f"skip_reason is too vague ('{skip_reason}'). "
                "Write a specific explanation: what work was already done, where, and when."
            )

    _guard_task_id(task_id)
    root = Path(repo_root).resolve()
    task_path = _task_dir(root, task_id)
    content = _load_todo(task_path)
    items = _parse_items(content)

    if item_index < 0 or item_index >= len(items):
        raise ValueError(
            f"item_index {item_index} is out of range. "
            f"Todo has {len(items)} items (0-based)."
        )

    line_i, marker, item_text = items[item_index]

    # Gate: must be the [>] in-progress item
    if marker != _INPROG:
        # Give a clear, actionable error
        inprog = next(((i, t) for i, (_, m, t) in enumerate(items) if m == _INPROG), None)
        pending = next(((i, t) for i, (_, m, t) in enumerate(items) if m == _PENDING), None)
        if inprog:
            raise ValueError(
                f"Cannot advance item {item_index} ('{item_text[:60]}') — "
                f"it is not in progress. "
                f"The current in-progress item is index {inprog[0]}: '{inprog[1][:60]}'. "
                "Complete or skip that item first."
            )
        elif pending:
            raise ValueError(
                f"Cannot advance item {item_index} — no item is currently in progress. "
                "Call get_next_todo first to start an item."
            )
        else:
            raise ValueError(
                f"Cannot advance item {item_index} — all items are already done or skipped."
            )

    lines = content.splitlines(keepends=True)

    # Apply the action
    if action == "complete":
        lines[line_i] = lines[line_i].replace("- [>]", "- [x]", 1)
    else:  # skip
        reason_clean = skip_reason.strip()
        # Replace the line entirely to embed the reason
        base = lines[line_i].rstrip("\n").replace("- [>]", "- [~]", 1)
        lines[line_i] = f"{base} — SKIPPED: {reason_clean}\n"

    new_content = "".join(lines)

    # Auto-start the next pending item
    updated_items = _parse_items(new_content)
    next_pending = next(
        ((i, li, t) for i, (li, m, t) in enumerate(updated_items) if m == _PENDING),
        None,
    )

    if next_pending is not None:
        next_idx, next_line_i, next_text = next_pending
        next_lines = new_content.splitlines(keepends=True)
        next_lines[next_line_i] = next_lines[next_line_i].replace("- [ ]", "- [>]", 1)
        new_content = "".join(next_lines)
        new_content = _update_current_step(new_content, next_text)
        all_done = False
    else:
        new_content = _update_current_step(new_content, None)
        next_idx = None
        next_text = None
        all_done = True

    _save_todo(task_path, new_content)

    remaining = sum(1 for _, m, _ in _parse_items(new_content) if m == _PENDING)

    return {
        "action": action,
        "completed_item": item_text,
        "advanced_to": next_text,
        "next_index": next_idx,
        "remaining": remaining,
        "all_done": all_done,
    }


def move_task_to_archive(
    task_id: str,
    archive_base: str,
    repo_root: str = ".",
    db: Any = None,
) -> dict:
    """
    Move a completed task workspace from .aca/active/<task-id>/
    to <archive_base>/<task-id>/ (user-level storage outside the repo).

    Updates the `tasks` DB row (status=archived, archive_path, archived_at)
    if db is provided.

    Returns {"task_id": str, "archived_to": str}
    """
    _guard_task_id(task_id)

    root = Path(repo_root).resolve()
    task_path = _task_dir(root, task_id)

    if not task_path.exists():
        raise FileNotFoundError(
            f"Task workspace '{task_id}' not found at '{task_path}'."
        )

    archive_dir = Path(archive_base).resolve() / task_id
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_path), str(archive_dir))

    archived_at = int(time.time() * 1000)

    if db is not None:
        try:
            db.execute(
                """
                UPDATE tasks
                SET status = 'archived', archive_path = ?, archived_at = ?
                WHERE task_id = ?
                """,
                (str(archive_dir), archived_at, task_id),
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][workspace] DB log failed for move_task_to_archive: {exc}")

    return {
        "task_id": task_id,
        "archived_to": str(archive_dir),
    }


# ── Schema definitions ────────────────────────────────────────────────────────

_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_task_workspace",
            "description": (
                "Create the task workspace folder .aca/active/<task-id>/ for a new task. "
                "Must be called before any artifact files can be written for this task. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional task identifier slug. Runtime may ignore this and pin its own task id.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_task_file",
            "description": (
                "Write or overwrite an artifact file inside the task workspace .aca/active/<task-id>/. "
                "filename must be one of: task.md, plan.md, todo.md, findings.md, output.md. "
                "The workspace must already exist (call create_task_workspace first). "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task identifier, matching the workspace folder name.",
                    },
                    "filename": {
                        "type": "string",
                        "enum": ["task.md", "plan.md", "todo.md", "findings.md", "output.md"],
                        "description": "The artifact file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full markdown content to write to the file.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["task_id", "filename", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_todo",
            "description": (
                "Get the next pending todo item for a task and mark it as in-progress [>]. "
                "Always call this before starting work on any todo item. "
                "Returns one item at a time — work only that item before calling advance_todo. "
                "If an item is already in-progress [>] (resumed turn), returns it without re-marking. "
                "Returns all_done=true when no pending items remain. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task identifier.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_todo",
            "description": (
                "Mark the current in-progress todo item as complete [x] or skipped [~], "
                "and automatically start the next pending item [>]. "
                "GATE: item_index must be the current [>] item — attempting to advance any "
                "other item returns an error. "
                "skip_reason is REQUIRED for action='skip' and must be specific: "
                "state exactly what work was already done, where, and when. "
                "Only skip if you are 100% certain the item was handled as a side-effect "
                "of other work — do not skip to avoid doing work. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task identifier.",
                    },
                    "item_index": {
                        "type": "integer",
                        "description": "0-based index of the item to advance. Must match the current [>] item.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["complete", "skip"],
                        "description": "'complete' if the item is done. 'skip' only if the item was provably handled already.",
                    },
                    "skip_reason": {
                        "type": "string",
                        "description": (
                            "Required when action='skip'. "
                            "Be specific: 'Already implemented in item 0 — method added at registry.py:123.' "
                            "Vague reasons (e.g. 'not needed') are rejected."
                        ),
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["task_id", "item_index", "action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_task_to_archive",
            "description": (
                "Move a completed task workspace from .aca/active/<task-id>/ to a user-level archive directory. "
                "Use after a task is finished and the 12-hour TTL has elapsed, or when cleaning up. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task identifier to archive.",
                    },
                    "archive_base": {
                        "type": "string",
                        "description": "Absolute path to the user-level archive base directory.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["task_id", "archive_base"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Registration ──────────────────────────────────────────────────────────────

_FN_MAP = {
    "create_task_workspace": create_task_workspace,
    "write_task_file": write_task_file,
    "get_next_todo": get_next_todo,
    "advance_todo": advance_todo,
    "move_task_to_archive": move_task_to_archive,
}


def register(registry: ToolRegistry) -> None:
    """Register all workspace tools into the given ToolRegistry."""
    for schema in _SCHEMAS:
        name = schema["function"]["name"]
        registry.register(ToolDefinition(
            fn=_FN_MAP[name],
            schema=schema,
            category=ToolCategory.WORKSPACE,
        ))
