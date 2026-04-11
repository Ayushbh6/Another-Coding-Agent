"""
Workspace tools for ACA.

Manage the .aca/active/<task-id>/ workspace per TOOLS_AND_PERMISSIONS.md §9.

Rules enforced here (not left to agent discretion):
  - task subfolders only created via create_task_workspace()
  - artifact writes only via write_task_file() with permitted filename validation
  - general write tools are blocked from .aca/ (enforced in write.py)
  - move_task_to_archive() is the only way to retire a workspace

Requires PermissionMode.EDIT or above.

Register into a ToolRegistry via:
    from aca.tools import workspace
    workspace.register(registry)
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from aca.tools.registry import ToolCategory, ToolDefinition, ToolRegistry


# ── Constants ─────────────────────────────────────────────────────────────────

_ACA_ACTIVE = ".aca/active"

# Permitted artifact filenames per TOOLS_AND_PERMISSIONS.md §9.3
_PERMITTED_FILENAMES = {"task.md", "plan.md", "todo.md", "findings.md", "output.md"}


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


def _guard_filename(filename: str) -> None:
    """Ensure the filename is on the permitted list."""
    if filename not in _PERMITTED_FILENAMES:
        raise ValueError(
            f"'{filename}' is not a permitted artifact filename. "
            f"Allowed: {sorted(_PERMITTED_FILENAMES)}"
        )


# ── Tool implementations ──────────────────────────────────────────────────────

def create_task_workspace(
    task_id: str,
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
    existed = target.exists()
    target.write_text(content, encoding="utf-8")

    return {
        "task_id": task_id,
        "filename": filename,
        "path": str(target.relative_to(root)),
        "bytes_written": len(content.encode("utf-8")),
        "action": "overwritten" if existed else "created",
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
                        "description": "Unique task identifier slug, e.g. 'task-001' or 'task-abc123'. No slashes or dots.",
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
