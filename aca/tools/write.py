"""
Write tools for ACA.

All tools here require PermissionMode.EDIT or above.
They enforce:
  - repo-root boundary (no writes outside the repo)
  - blocked from writing into .aca/ (workspace tools own that path)
  - no path traversal
  - no overwrites of sensitive files

Register into a ToolRegistry via:
    from aca.tools import write
    write.register(registry)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import whatthepatch

from aca.tools.registry import ToolCategory, ToolDefinition, ToolRegistry
from aca.tools.read import _global_aca_dir, _is_sensitive_path, _resolve_and_guard


# ── Safety helpers ────────────────────────────────────────────────────────────

_ACA_WORKSPACE_DIR = ".aca"
_EXAMPLE_GUIDELINES_DIR = ".aca/example_guidelines"


def _guard_write_path(path_str: str, repo_root: Path) -> Path:
    """
    Resolve and validate a write target.

    Raises ValueError if:
      - path resolves into ~/.aca/example_guidelines/ (global read-only templates)
      - path is inside repo-local .aca/example_guidelines/
      - path is inside repo .aca/ (workspace tools own that)
      - path escapes repo root
      - path matches a sensitive file pattern
    """
    target = _resolve_and_guard(path_str, repo_root)

    # Block global ~/.aca/example_guidelines/ — checked first so the error is specific
    global_eg = (_global_aca_dir() / "example_guidelines").resolve()
    try:
        target.relative_to(global_eg)
        raise ValueError(
            f"Path '{path_str}' resolves into '~/.aca/example_guidelines/' which is "
            "read-only. These files are global format reference templates and must "
            "never be modified."
        )
    except ValueError as exc:
        if "read-only" in str(exc):
            raise

    # Hard-block repo-local example_guidelines/ (belt-and-suspenders)
    eg_dir = (repo_root / _EXAMPLE_GUIDELINES_DIR).resolve()
    try:
        target.relative_to(eg_dir)
        raise ValueError(
            f"Path '{path_str}' is inside '.aca/example_guidelines/' which is read-only. "
            "These files are format reference templates and must never be modified."
        )
    except ValueError as exc:
        if "read-only" in str(exc):
            raise

    aca_dir = (repo_root / _ACA_WORKSPACE_DIR).resolve()
    try:
        target.relative_to(aca_dir)
        raise ValueError(
            f"Writes into '{_ACA_WORKSPACE_DIR}/' are not allowed via general write tools. "
            "Use write_task_file for workspace artifact files."
        )
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise

    if _is_sensitive_path(target):
        raise ValueError(
            f"Path '{path_str}' matches a sensitive file pattern. Write blocked."
        )
    return target


# ── Tool implementations ──────────────────────────────────────────────────────

def write_file(path: str, content: str, repo_root: str = ".", overwrite: bool = True) -> dict:
    """
    Write content to a file inside the repo.

    - overwrite=True (default): create or overwrite the file unconditionally.
    - overwrite=False: create the file; raise FileExistsError if it already exists.

    Creates parent directories as needed.

    Returns {"path": str, "bytes_written": int, "action": "created"|"overwritten"}
    """
    root = Path(repo_root).resolve()
    target = _guard_write_path(path, root)

    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError(
            f"File '{path}' already exists. Set overwrite=true to replace it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return {
        "path": str(target.relative_to(root)),
        "bytes_written": len(content.encode("utf-8")),
        "action": "overwritten" if existed else "created",
    }


def create_file(path: str, content: str, repo_root: str = ".") -> dict:
    """Alias kept for backwards-compat with existing tests; delegates to write_file."""
    return write_file(path=path, content=content, repo_root=repo_root, overwrite=False)


def update_file(path: str, old_string: str, new_string: str, repo_root: str = ".") -> dict:
    """
    Replace an exact string occurrence in a file.

    `old_string` must appear exactly once in the file — this prevents silent
    multi-site edits. If it appears zero or multiple times the tool raises an
    error with a count so the agent can adjust.

    For multiple replacements in one call, use multi_update_file instead.

    Returns {"path": str, "replaced": 1}
    """
    root = Path(repo_root).resolve()
    target = _guard_write_path(path, root)

    if not target.exists():
        raise FileNotFoundError(f"File not found: '{path}'")

    original = target.read_text(encoding="utf-8")
    count = original.count(old_string)

    if count == 0:
        raise ValueError(
            f"old_string not found in '{path}'. "
            "Check for whitespace/indentation differences or use search_repo to locate the exact text."
        )
    if count > 1:
        raise ValueError(
            f"old_string appears {count} times in '{path}'. "
            "Provide more surrounding context lines to make it unique."
        )

    updated = original.replace(old_string, new_string, 1)
    target.write_text(updated, encoding="utf-8")

    return {
        "path": str(target.relative_to(root)),
        "replaced": 1,
    }


def multi_update_file(
    path: str,
    edits: list[dict],
    repo_root: str = ".",
) -> dict:
    """
    Apply multiple exact-string replacements to a file atomically.

    `edits` is an ordered list of {"old_string": str, "new_string": str} dicts.
    Rules:
      - Each old_string must appear exactly once in the *current* file content.
      - Edits are applied in the order given; each edit sees the result of prior edits.
      - If any edit fails (not found, or found multiple times), the entire operation
        is rolled back — the file is left unchanged.

    Returns:
      {"path": str, "edits_applied": int, "bytes_written": int}
    """
    if not edits:
        raise ValueError("edits list is empty — nothing to do.")

    root = Path(repo_root).resolve()
    target = _guard_write_path(path, root)

    if not target.exists():
        raise FileNotFoundError(f"File not found: '{path}'")

    original = target.read_text(encoding="utf-8")
    current = original

    for i, edit in enumerate(edits):
        old = edit.get("old_string", "")
        new = edit.get("new_string", "")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError(f"Edit #{i+1}: old_string and new_string must be strings.")
        count = current.count(old)
        if count == 0:
            raise ValueError(
                f"Edit #{i+1}: old_string not found in the file at this point. "
                "Remember edits are applied in order — the file content changes after each step."
            )
        if count > 1:
            raise ValueError(
                f"Edit #{i+1}: old_string appears {count} times. "
                "Provide more surrounding context to make it unique."
            )
        current = current.replace(old, new, 1)

    # All edits validated — write once
    target.write_text(current, encoding="utf-8")

    return {
        "path": str(target.relative_to(root)),
        "edits_applied": len(edits),
        "bytes_written": len(current.encode("utf-8")),
    }


def apply_patch(path: str, patch: str, repo_root: str = ".") -> dict:
    """
    Apply a unified diff patch to a file using pure Python (whatthepatch).

    `patch` must be a valid unified diff string (output of `git diff` or `diff -u`).
    No system `patch` binary required — works cross-platform.

    Returns {"path": str, "success": True, "hunks_applied": int}
    """
    root = Path(repo_root).resolve()
    target = _guard_write_path(path, root)

    if not target.exists():
        raise FileNotFoundError(f"File not found: '{path}'")

    original = target.read_text(encoding="utf-8")

    # Parse the patch
    diffs = list(whatthepatch.parse_patch(patch))
    if not diffs:
        raise ValueError("No valid diff hunks found in the provided patch string.")

    # Use the first diff that matches our file (by path fragment or positionally)
    diff = None
    rel_str = str(target.relative_to(root))
    for d in diffs:
        old_p = (d.header.old_path or "").lstrip("ab/")
        new_p = (d.header.new_path or "").lstrip("ab/")
        if rel_str in (old_p, new_p) or old_p.endswith(target.name) or new_p.endswith(target.name):
            diff = d
            break
    if diff is None:
        diff = diffs[0]  # fallback: apply first diff

    # Apply
    result_lines = whatthepatch.apply_diff(diff, original)
    if result_lines is None:
        raise RuntimeError(
            "Patch could not be applied cleanly. The file content may not match the patch context. "
            "Read the current file and regenerate the patch."
        )

    # whatthepatch returns lines without trailing newline on the last line sometimes
    result_text = "\n".join(result_lines)
    # Preserve original trailing newline behaviour
    if original.endswith("\n") and not result_text.endswith("\n"):
        result_text += "\n"

    target.write_text(result_text, encoding="utf-8")

    hunks = len([c for c in diff.changes if c.old is None or c.new is None])

    return {
        "path": str(target.relative_to(root)),
        "success": True,
        "hunks_applied": hunks,
    }


def delete_file(path: str, repo_root: str = ".") -> dict:
    """
    Delete a file inside the repo.

    Blocked for:
      - paths outside the repo root
      - sensitive files (.env, credentials, etc.)
      - paths inside .aca/ (workspace lifecycle is managed by workspace tools)
      - directories (use this only for single files)

    Returns {"path": str, "deleted": True}
    """
    root = Path(repo_root).resolve()
    target = _guard_write_path(path, root)

    if not target.exists():
        raise FileNotFoundError(f"File not found: '{path}'")
    if not target.is_file():
        raise ValueError(
            f"'{path}' is a directory. delete_file only removes single files."
        )

    target.unlink()

    return {
        "path": str(target.relative_to(root)),
        "deleted": True,
    }


# ── Schema definitions ────────────────────────────────────────────────────────

_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file inside the repo. "
                "Set overwrite=false to create a new file and fail if it already exists. "
                "Set overwrite=true (default) to create or unconditionally replace the file. "
                "For targeted edits to existing files, prefer update_file or multi_update_file. "
                "Cannot write into .aca/. Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from repo root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "If true (default), create or overwrite. "
                            "If false, fail with an error if the file already exists."
                        ),
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_file",
            "description": (
                "Replace a single exact string in a file with a new string. "
                "old_string must appear exactly once — include enough surrounding lines to make it unique. "
                "For multiple replacements in one call, use multi_update_file. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from repo root.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace. Must appear exactly once.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_update_file",
            "description": (
                "Apply multiple exact-string replacements to a file in a single atomic operation. "
                "Use this for multi-hunk edits to avoid making many sequential tool calls. "
                "Edits are applied in order — each edit sees the result of the previous ones. "
                "All edits succeed or the file is left completely unchanged (all-or-nothing). "
                "Each old_string must appear exactly once at the point it is applied. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from repo root.",
                    },
                    "edits": {
                        "type": "array",
                        "description": "Ordered list of replacements to apply.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {
                                    "type": "string",
                                    "description": "Exact string to replace (must be unique at time of application).",
                                },
                                "new_string": {
                                    "type": "string",
                                    "description": "Replacement string.",
                                },
                            },
                            "required": ["old_string", "new_string"],
                            "additionalProperties": False,
                        },
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["path", "edits"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff patch to a file using pure Python — no system patch binary needed. "
                "patch must be a valid unified diff string (e.g. from `git diff` or `diff -u`). "
                "If the patch does not apply cleanly, read the current file and regenerate the patch. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file to patch.",
                    },
                    "patch": {
                        "type": "string",
                        "description": "The unified diff patch string to apply.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["path", "patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Delete a single file inside the repo. "
                "Blocked for sensitive files, paths outside the repo, and .aca/ workspace files. "
                "Does not delete directories. "
                "Requires EDIT or FULL permission mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file to delete.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Registration ──────────────────────────────────────────────────────────────

_FN_MAP = {
    "write_file": write_file,
    "update_file": update_file,
    "multi_update_file": multi_update_file,
    "apply_patch": apply_patch,
    "delete_file": delete_file,
}


def register(registry: ToolRegistry) -> None:
    """Register all write tools into the given ToolRegistry."""
    for schema in _SCHEMAS:
        name = schema["function"]["name"]
        registry.register(ToolDefinition(
            fn=_FN_MAP[name],
            schema=schema,
            category=ToolCategory.WRITE,
        ))
