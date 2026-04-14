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

from aca.tools.registry import ToolCategory, ToolDefinition, ToolExecutionError, ToolRegistry
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


def _raise_edit_failure(
    *,
    path: str,
    failed_edit_index: int,
    reason: str,
    detail: str,
    suggested_next_tool: list[str],
) -> None:
    raise ToolExecutionError(
        detail,
        payload={
            "path": path,
            "failed_edit_index": failed_edit_index,
            "reason": reason,
            "rolled_back": True,
            "suggested_next_tool": suggested_next_tool,
        },
    )


def _apply_exact_edits(path: str, edits: list[dict], repo_root: str = ".") -> tuple[dict, str]:
    """Shared backend for exact-string edit tools."""
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
            _raise_edit_failure(
                path=path,
                failed_edit_index=i,
                reason="old_string_not_found",
                detail=(
                    f"Edit #{i+1}: old_string not found in '{path}' at the current file state. "
                    "Edits are applied in order, and this operation is atomic: if any edit fails, "
                    "the whole batch is rolled back and the file is left unchanged. "
                    "Do not blindly retry another exact-edit batch. Re-read the file, then use "
                    "edit_file for isolated exact replacements or apply_patch for larger, overlapping, "
                    "or context-sensitive edits."
                ),
                suggested_next_tool=["read_file", "edit_file"],
            )
        if count > 1:
            _raise_edit_failure(
                path=path,
                failed_edit_index=i,
                reason="old_string_not_unique",
                detail=(
                    f"Edit #{i+1}: old_string appears {count} times in '{path}'. "
                    "Edits are order-sensitive and atomic, so the file has been rolled back unchanged. "
                    "Re-read a larger slice to make the target unique, then use edit_file again or "
                    "switch to apply_patch if the change is structural."
                ),
                suggested_next_tool=["read_files", "apply_patch"],
            )
        current = current.replace(old, new, 1)

    # All edits validated — write once
    target.write_text(current, encoding="utf-8")

    result = {
        "path": str(target.relative_to(root)),
        "edits_applied": len(edits),
        "bytes_written": len(current.encode("utf-8")),
    }
    return result, str(target.relative_to(root))


def edit_file(path: str, edits: list[dict], repo_root: str = ".") -> dict:
    """
    Primary exact-edit tool for ordered atomic replacements in one file.

    Each edit item must be {"old_string": str, "new_string": str}. Every
    old_string must match exactly once at the moment it is applied. All edits
    succeed or the file is rolled back unchanged.
    """
    result, _rel_path = _apply_exact_edits(path=path, edits=edits, repo_root=repo_root)
    return result


def update_file(path: str, old_string: str, new_string: str, repo_root: str = ".") -> dict:
    """
    Legacy exact-edit alias for a single replacement.

    Prefer edit_file for all new prompt guidance.
    """
    result, rel_path = _apply_exact_edits(
        path=path,
        edits=[{"old_string": old_string, "new_string": new_string}],
        repo_root=repo_root,
    )
    return {
        "path": rel_path,
        "replaced": result["edits_applied"],
    }


def multi_update_file(
    path: str,
    edits: list[dict],
    repo_root: str = ".",
) -> dict:
    """
    Legacy exact-edit alias for multiple ordered atomic replacements.

    Prefer edit_file for all new prompt guidance.
    """
    result, _rel_path = _apply_exact_edits(path=path, edits=edits, repo_root=repo_root)
    return result


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
                "For targeted edits to existing files, prefer edit_file or apply_patch. "
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
            "name": "edit_file",
            "description": (
                "Apply one or more exact-string replacements to a file in one atomic operation. "
                "This is the preferred exact-edit tool. Edits are applied in order, and each old_string "
                "must appear exactly once at the point it is applied. If any edit fails, the whole file "
                "is rolled back unchanged. Requires EDIT or FULL permission mode."
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
                        "description": "Ordered list of exact replacements to apply atomically.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {
                                    "type": "string",
                                    "description": "Exact string to replace (must be unique when this edit runs).",
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
            "name": "update_file",
            "description": (
                "Legacy exact-edit alias for a single replacement. "
                "Prefer edit_file for all new calls. old_string must appear exactly once. "
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
                "Legacy exact-edit alias for ordered atomic replacements. "
                "Prefer edit_file for all new calls. Edits are applied in order and "
                "all edits succeed or the file is left completely unchanged. "
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
    "edit_file": edit_file,
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
            expose_to_agent=(name not in {"update_file", "multi_update_file"}),
        ))
