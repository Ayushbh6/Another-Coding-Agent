"""
Read tools for ACA.

All tools here are read-only and safe for PermissionMode.READ and above.
They operate within a repo root boundary and block access to sensitive paths.

Register into a ToolRegistry via:
    from aca.tools import read
    read.register(registry)
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

from aca.tools.registry import ToolCategory, ToolDefinition, ToolRegistry, _is_internal_repo_aca_path


# ── Hard-excluded directory names ────────────────────────────────────────────────────
#
# These directories are ALWAYS excluded from list_files, regardless of
# .gitignore.  They are never useful for an LLM to traverse and are often
# enormous (tokens, binaries, compiled artefacts).
#
# Inspired by OpenCode (ignoreNested set) and Claude Code (static exclusion).
_ALWAYS_EXCLUDE_DIRS: frozenset[str] = frozenset({
    # Python
    ".venv", "venv", "env", ".env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", "*.egg-info", ".eggs",
    "htmlcov", ".coverage",
    # Node / JS
    "node_modules", ".pnp", ".yarn",
    "dist", "build", ".next", ".nuxt", ".svelte-kit",
    ".cache", ".parcel-cache", ".turbo",
    # Rust
    "target",
    # Java / JVM
    "out", ".gradle", ".mvn",
    # Version control / IDE
    ".git", ".hg", ".svn",
    ".idea", ".vscode", ".vs",
    # Misc build/vendor
    "vendor", "_build", "buck-out",
    # OS
    ".DS_Store",
})


# ── Safety helpers ────────────────────────────────────────────────────────────

_BLOCKED_PATH_FRAGMENTS = [
    ".ssh", ".aws", ".gnupg", ".gpg",
    "id_rsa", "id_ed25519", "id_ecdsa",
    ".env",
]

_BLOCKED_FILENAME_PATTERNS = [
    ".env", ".env.local", ".env.production", ".env.staging",
    "credentials", "secrets", "token", "private_key",
]

_READ_FILE_DEFAULT_MAX_LINES = 2000
_READ_FILES_DEFAULT_MAX_TOTAL_LINES = 4000


def _is_sensitive_path(path: Path) -> bool:
    """Return True if the path looks like a secrets / credentials file."""
    parts_lower = [p.lower() for p in path.parts]
    name_lower = path.name.lower()
    for fragment in _BLOCKED_PATH_FRAGMENTS:
        if any(fragment in part for part in parts_lower):
            return True
    for pattern in _BLOCKED_FILENAME_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def _global_aca_dir() -> Path:
    """Return the resolved path to ~/.aca/."""
    return Path.home() / ".aca"


def _is_global_guidelines_path(target: Path) -> bool:
    """Return True if target resolves inside ~/.aca/example_guidelines/."""
    eg_dir = (_global_aca_dir() / "example_guidelines").resolve()
    try:
        target.resolve().relative_to(eg_dir)
        return True
    except ValueError:
        return False


def _resolve_and_guard(path_str: str, repo_root: Path) -> Path:
    """
    Resolve a path relative to repo_root, enforce repo boundary,
    and reject sensitive paths.

    Narrow exception: absolute paths that resolve inside
    ~/.aca/example_guidelines/ are allowed through — they are the global
    read-only format reference files, accessible from any repo.

    Raises ValueError for any violation so the LLM receives a clear error.
    """
    # Allow absolute paths directly (e.g. /Users/foo/.aca/example_guidelines/task.md)
    p = Path(path_str)
    if p.is_absolute():
        target = p.resolve()
    else:
        target = (repo_root / path_str).resolve()

    # Narrow exception: global ~/.aca/example_guidelines/ bypasses repo boundary
    if _is_global_guidelines_path(target):
        # Still block sensitive patterns just in case
        if _is_sensitive_path(target):
            raise ValueError(
                f"Path '{path_str}' matches a sensitive file pattern. "
                "Reading secrets/credentials is blocked."
            )
        return target

    try:
        target.relative_to(repo_root.resolve())
    except ValueError:
        raise ValueError(
            f"Path '{path_str}' escapes the repo root. "
            "ACA only reads files inside the current repository "
            "or from ~/.aca/example_guidelines/."
        )
    if _is_sensitive_path(target):
        raise ValueError(
            f"Path '{path_str}' matches a sensitive file pattern. "
            "Reading secrets/credentials is blocked."
        )
    return target


# ── Tool implementations ──────────────────────────────────────────────────────

def read_file(
    path: str,
    repo_root: str = ".",
    start_line: int | None = None,
    end_line: int | None = None,
    max_lines: int | None = None,
) -> dict:
    """
    Read a file (or a slice of it) inside the repo.

    Line numbers are 1-based and inclusive.
    - start_line only  → read from that line to EOF (or max_lines)
    - end_line only    → read from line 1 to end_line
    - both             → read that range
    - max_lines        → cap the number of lines returned from start_line
    - none             → full file (capped at 2000 lines; use start_line/end_line for larger files)

    Returns:
      {
        "path": str,
        "content": str,           # the returned lines joined
        "start_line": int,        # actual first line returned (1-based)
        "end_line": int,          # actual last line returned (1-based)
        "lines_returned": int,
        "total_lines": int,       # total lines in the file
        "truncated": bool,        # True if file has more lines beyond end_line
      }
    """
    root = Path(repo_root).resolve()
    target = _resolve_and_guard(path, root)
    if not target.exists():
        raise FileNotFoundError(f"File not found: '{path}'")
    if not target.is_file():
        raise ValueError(f"'{path}' is a directory, not a file.")

    all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    total_lines = len(all_lines)

    # Resolve window bounds (convert to 0-based indices)
    s = (start_line - 1) if start_line is not None else 0
    e = end_line if end_line is not None else total_lines   # end_line is inclusive

    # Clamp
    s = max(0, min(s, total_lines))
    e = max(s, min(e, total_lines))

    # Apply max_lines cap
    if max_lines is not None:
        e = min(e, s + max_lines)

    # Apply default cap when no bounds given
    if start_line is None and end_line is None and max_lines is None:
        e = min(e, s + _READ_FILE_DEFAULT_MAX_LINES)

    slice_lines = all_lines[s:e]
    content = "".join(slice_lines)

    # For files outside the repo root (e.g. global ~/.aca/example_guidelines/),
    # relative_to would fail — use the absolute path string instead.
    try:
        display_path = str(target.relative_to(root))
    except ValueError:
        display_path = str(target)

    return {
        "path": display_path,
        "content": content,
        "start_line": s + 1,
        "end_line": s + len(slice_lines),
        "lines_returned": len(slice_lines),
        "total_lines": total_lines,
        "truncated": (s + len(slice_lines)) < total_lines,
    }


def read_files(
    requests: list[dict],
    repo_root: str = ".",
    max_total_lines: int = _READ_FILES_DEFAULT_MAX_TOTAL_LINES,
) -> dict:
    """
    Read one or more file slices in one call, preserving request order.

    Each request item is:
      {"path": str, "start_line": int?, "end_line": int?}

    Repeated paths are allowed so callers can request multiple disjoint slices
    from the same file. The global max_total_lines budget is enforced across the
    entire batch. Once that budget is exhausted, later requests are omitted.
    """
    if not requests:
        raise ValueError("requests list is empty — nothing to do.")
    if max_total_lines <= 0:
        raise ValueError("max_total_lines must be a positive integer.")

    results: list[dict] = []
    remaining_lines = max_total_lines
    omitted_count = 0

    for idx, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"Request #{idx + 1} must be an object.")

        path = request.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Request #{idx + 1} must include a non-empty path.")

        start_line = request.get("start_line")
        end_line = request.get("end_line")
        if start_line is not None and not isinstance(start_line, int):
            raise ValueError(f"Request #{idx + 1}: start_line must be an integer.")
        if end_line is not None and not isinstance(end_line, int):
            raise ValueError(f"Request #{idx + 1}: end_line must be an integer.")

        if remaining_lines <= 0:
            omitted_count += len(requests) - idx
            break

        # Bound each individual read by the remaining global budget.
        bounded_end_line = end_line
        bounded_max_lines: int | None = None
        if start_line is not None and end_line is not None:
            span = (end_line - start_line) + 1
            if span > remaining_lines:
                bounded_end_line = start_line + remaining_lines - 1
        elif start_line is not None:
            bounded_max_lines = remaining_lines
        elif end_line is not None:
            bounded_end_line = min(end_line, remaining_lines)
        else:
            bounded_max_lines = min(remaining_lines, _READ_FILE_DEFAULT_MAX_LINES)

        result = read_file(
            path=path,
            repo_root=repo_root,
            start_line=start_line,
            end_line=bounded_end_line,
            max_lines=bounded_max_lines,
        )
        results.append(result)
        remaining_lines -= result["lines_returned"]

    omitted_reason = None
    if omitted_count:
        omitted_reason = (
            f"Stopped after reaching max_total_lines={max_total_lines}; "
            f"omitted {omitted_count} later request(s)."
        )

    return {
        "results": results,
        "total_slices_read": len(results),
        "total_lines_returned": sum(item["lines_returned"] for item in results),
        "max_total_lines": max_total_lines,
        "omitted_count": omitted_count,
        "omitted_reason": omitted_reason,
    }


def read_many_files(
    requests: list[dict],
    repo_root: str = ".",
    max_total_lines: int = _READ_FILES_DEFAULT_MAX_TOTAL_LINES,
) -> dict:
    """Legacy alias kept for compatibility."""
    return read_files(requests=requests, repo_root=repo_root, max_total_lines=max_total_lines)


def _batch_gitignored(paths: list[Path], repo_root: Path) -> set[str]:
    """
    Ask git which of the given paths are gitignored, in a single subprocess call.

    Uses ``git check-ignore --stdin`` for efficiency (one process, N paths).
    Returns a set of absolute path strings that are gitignored.
    Falls back to empty set if git is unavailable or repo has no .git.
    """
    if not paths:
        return set()
    try:
        stdin_data = "\n".join(str(p) for p in paths)
        result = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "--stdin"],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # check-ignore outputs the paths that ARE ignored (one per line)
        ignored = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                ignored.add(str(Path(line).resolve()))
        return ignored
    except Exception:  # noqa: BLE001
        return set()  # fail open


def list_files(
    path: str = ".",
    pattern: str | None = None,
    repo_root: str = ".",
    max_depth: int = 3,
    include_hidden: bool = False,
) -> dict:
    """
    List files and directories under `path` up to max_depth levels deep.
    Optionally filter by glob pattern (e.g. "*.py").

    Returns:
      {"path": str, "pattern": str|None, "entries": [{"name": str, "type": str, "size_bytes": int|None}]}
    """
    root = Path(repo_root).resolve()
    target = _resolve_and_guard(path, root)
    if not target.exists():
        raise FileNotFoundError(f"Directory not found: '{path}'")
    if not target.is_dir():
        raise ValueError(f"'{path}' is a file, not a directory.")

    # Collect all candidate paths first so we can batch-check gitignore.
    candidates: list[Path] = []

    def _collect(directory: Path, current_depth: int) -> None:
        try:
            children = sorted(directory.iterdir())
        except PermissionError:
            return
        for item in children:
            # Always-exclude: junk dirs that are never useful to an LLM
            if item.is_dir() and item.name in _ALWAYS_EXCLUDE_DIRS:
                continue
            # Hidden files/dirs (unless the caller explicitly asked for them)
            if not include_hidden and item.name.startswith("."):
                continue
            # Safety: skip paths that escape repo root or match sensitive patterns
            try:
                _resolve_and_guard(str(item.relative_to(root)), root)
            except ValueError:
                continue
            candidates.append(item)
            if item.is_dir() and current_depth < max_depth:
                _collect(item, current_depth + 1)

    _collect(target, 1)

    # Batch gitignore check — one subprocess call for all candidates
    gitignored = _batch_gitignored(candidates, root)

    entries = []
    for item in candidates:
        abs_str = str(item.resolve())
        if abs_str in gitignored and not _is_internal_repo_aca_path(item, root):
            continue  # silently skip gitignored entries
        # Pattern filter — only applied to files
        if pattern and item.is_file() and not item.match(pattern):
            continue
        entries.append({
            "name": str(item.relative_to(root)),
            "type": "file" if item.is_file() else "dir",
            "size_bytes": item.stat().st_size if item.is_file() else None,
        })

    return {
        "path": str(target.relative_to(root)),
        "pattern": pattern,
        "entries": entries,
    }


def search_repo(
    query: str,
    file_pattern: str | None = None,
    repo_root: str = ".",
    max_results: int = 50,
    context_lines: int = 2,
    case_insensitive: bool = False,
) -> dict:
    """
    Search the repo for a literal string or regex pattern using ripgrep.
    Returns matching file paths, line numbers, matched text, and surrounding
    context lines.

    Returns:
      {
        "query": str,
        "match_count": int,
        "matches": [
          {
            "file": str,
            "line": int,
            "text": str,           # the matching line
            "context_before": [str],
            "context_after": [str],
          }
        ]
      }
    """
    root = Path(repo_root).resolve()

    cmd = ["rg", "--json", f"--context={context_lines}", "--max-count=200", query, str(root)]
    if file_pattern:
        cmd.extend(["--glob", file_pattern])
    if case_insensitive:
        cmd.append("--ignore-case")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ripgrep (rg) is not installed or not on PATH. "
            "Install it with: brew install ripgrep"
        )

    # Parse the NDJSON output from rg --json.
    # Message types: "begin", "match", "context", "end", "summary"
    # We collect context lines per file+match-block, then attach them to the match.
    matches: list[dict] = []
    pending_context_before: list[str] = []

    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")

        if msg_type == "begin":
            pending_context_before = []

        elif msg_type == "context":
            data = obj["data"]
            pending_context_before.append(data["lines"]["text"].rstrip("\n"))

        elif msg_type == "match":
            data = obj["data"]
            file_path = data["path"]["text"]
            try:
                rel = str(Path(file_path).relative_to(root))
            except ValueError:
                rel = file_path

            match_text = data["lines"]["text"].rstrip("\n")
            match_line = data["line_number"]

            matches.append({
                "file": rel,
                "line": match_line,
                "text": match_text,
                "context_before": list(pending_context_before),
                "context_after": [],       # filled in by subsequent context messages
                "_collecting_after": True, # internal flag, stripped below
            })
            pending_context_before = []    # reset for next match in same block

        elif msg_type == "end":
            # Any pending context lines after the last match in this block
            # belong to context_after of the last match in this file
            if matches and matches[-1].get("_collecting_after"):
                matches[-1]["context_after"] = list(pending_context_before)
            pending_context_before = []

        if len(matches) >= max_results:
            break

    # Clean up internal flag
    for m in matches:
        m.pop("_collecting_after", None)

    return {
        "query": query,
        "file_pattern": file_pattern,
        "case_insensitive": case_insensitive,
        "match_count": len(matches),
        "matches": matches,
    }


def get_file_outline(path: str, repo_root: str = ".") -> dict:
    """
    Return a structural outline of a file: class names, function/method names,
    and their starting line numbers — without reading the full content.

    For Python files, uses the AST for accuracy.
    For all other files, uses regex heuristics to find function/class-like
    declarations (works for JS/TS, Go, Rust, Java, etc.).

    Use this to orient inside a large file before deciding which lines to read.

    Returns:
      {
        "path": str,
        "language": str,       # "python" | "generic"
        "total_lines": int,
        "outline": [
          {"kind": "class"|"function"|"method", "name": str, "line": int, "parent": str|None}
        ]
      }
    """
    root = Path(repo_root).resolve()
    target = _resolve_and_guard(path, root)
    if not target.exists():
        raise FileNotFoundError(f"File not found: '{path}'")
    if not target.is_file():
        raise ValueError(f"'{path}' is a directory, not a file.")

    source = target.read_text(encoding="utf-8", errors="replace")
    all_lines = source.splitlines()
    total_lines = len(all_lines)

    # ── Python: use AST ───────────────────────────────────────────────────────
    if target.suffix.lower() == ".py":
        outline: list[dict] = []
        try:
            tree = ast.parse(source, filename=str(target))
        except SyntaxError as exc:
            return {
                "path": str(target.relative_to(root)),
                "language": "python",
                "total_lines": total_lines,
                "outline": [],
                "parse_error": str(exc),
            }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                outline.append({
                    "kind": "class",
                    "name": node.name,
                    "line": node.lineno,
                    "parent": None,
                })
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        outline.append({
                            "kind": "method",
                            "name": child.name,
                            "line": child.lineno,
                            "parent": node.name,
                        })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Top-level functions only (skip methods already captured above)
                parent_is_class = any(
                    isinstance(parent_node, ast.ClassDef)
                    and any(
                        getattr(c, "lineno", None) == node.lineno
                        for c in ast.iter_child_nodes(parent_node)
                    )
                    for parent_node in ast.walk(tree)
                    if isinstance(parent_node, ast.ClassDef)
                )
                if not parent_is_class:
                    outline.append({
                        "kind": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "parent": None,
                    })

        outline.sort(key=lambda x: x["line"])
        return {
            "path": str(target.relative_to(root)),
            "language": "python",
            "total_lines": total_lines,
            "outline": outline,
        }

    # ── Generic: regex heuristics ─────────────────────────────────────────────
    # Covers JS/TS, Go, Rust, Java, C/C++, Ruby, etc.
    patterns: list[tuple[str, str]] = [
        # JS/TS: class Foo, class Foo extends Bar
        (r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", "class"),
        # JS/TS: function foo, async function foo, export function foo
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[\(\<]", "function"),
        # JS/TS: arrow / method: foo = () =>, foo: () =>
        (r"^\s*(?:(?:public|private|protected|static|async)\s+)*(\w+)\s*[=:]\s*(?:async\s*)?\(", "function"),
        # Go: func (recv) Name(, func Name(
        (r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", "function"),
        # Rust: fn name, pub fn name, async fn name
        (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[\(\<]", "function"),
        # Rust: struct / enum / impl / trait
        (r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)", "class"),
        # Java/C#: class/interface/enum
        (r"^\s*(?:public|private|protected|abstract|static)?\s*(?:class|interface|enum)\s+(\w+)", "class"),
        # Java/C#: method declarations
        (r"^\s*(?:public|private|protected|static|final|override|virtual|abstract)[\w\s<>\[\]]*\s+(\w+)\s*\(", "function"),
        # Ruby: def name, def self.name
        (r"^\s*def\s+(?:self\.)?(\w+)", "function"),
        # Ruby: class / module
        (r"^\s*(?:class|module)\s+(\w+)", "class"),
    ]

    outline_generic: list[dict] = []
    seen: set[tuple[int, str]] = set()

    for lineno, line in enumerate(all_lines, start=1):
        for pat, kind in patterns:
            m = re.match(pat, line)
            if m:
                name = m.group(1)
                key = (lineno, name)
                if key not in seen:
                    seen.add(key)
                    outline_generic.append({
                        "kind": kind,
                        "name": name,
                        "line": lineno,
                        "parent": None,
                    })
                break  # first matching pattern wins for this line

    return {
        "path": str(target.relative_to(root)),
        "language": "generic",
        "total_lines": total_lines,
        "outline": outline_generic,
    }


# ── Schema definitions ────────────────────────────────────────────────────────

_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file (or a line-range slice of it) inside the repo. "
                "Use start_line/end_line to zoom into large files — always check total_lines first. "
                "Without bounds, returns up to 2000 lines from the start. "
                "Blocked for secrets/credential files and paths outside the repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from repo root. E.g. 'src/main.py'.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root. Defaults to cwd.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-based line number to start reading from (inclusive).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based line number to stop reading at (inclusive).",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to return from start_line. Overrides end_line.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories under a path in the repo. "
                "Optionally filter by a glob pattern (e.g. '*.py'). "
                "Max depth 3 by default; set max_depth=1 for a flat listing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the directory to list. Defaults to repo root.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob to filter entries, e.g. '*.py', '*.md'.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to traverse. Defaults to 3.",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files/dirs (starting with '.'). Defaults to false.",
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
            "name": "read_files",
            "description": (
                "Read one or more files or file slices in one call. "
                "Use a single request for one file, or multiple requests for 2+ files, task artifact bundles, "
                "or multiple disjoint slices from the same file. "
                "Requests are processed in order and later requests may be omitted once the global line budget is exhausted. "
                "Blocked for secrets/credential files and paths outside the repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "description": "Ordered list of file read requests.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to the file from repo root.",
                                },
                                "start_line": {
                                    "type": "integer",
                                    "description": "1-based line number to start reading from (inclusive).",
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": "1-based line number to stop reading at (inclusive).",
                                },
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root. Defaults to cwd.",
                    },
                    "max_total_lines": {
                        "type": "integer",
                        "description": f"Global line budget across the whole batch. Defaults to {_READ_FILES_DEFAULT_MAX_TOTAL_LINES}.",
                    },
                },
                "required": ["requests"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": (
                "Search the repo for a literal string or regex using ripgrep. "
                "Returns matching file, line number, matched text, and context lines before/after. "
                "Supports regex patterns, file glob filtering, and case-insensitive matching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search string or regex pattern.",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob to restrict search, e.g. '*.py', '*.ts'.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max match entries to return. Defaults to 50.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context before/after each match. Defaults to 2.",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive matching. Defaults to false.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_outline",
            "description": (
                "Return a structural outline of a file — class names, function/method names, "
                "and their starting line numbers — without reading the full content. "
                "Use this to orient inside a large file before deciding which lines to read with read_file. "
                "Python files use the AST for accuracy. Other languages use regex heuristics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from repo root.",
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
    "read_file": read_file,
    "read_files": read_files,
    "list_files": list_files,
    "search_repo": search_repo,
    "get_file_outline": get_file_outline,
}


def register(registry: ToolRegistry) -> None:
    """Register all read tools into the given ToolRegistry."""
    for schema in _SCHEMAS:
        name = schema["function"]["name"]
        registry.register(ToolDefinition(
            fn=_FN_MAP[name],
            schema=schema,
            category=ToolCategory.READ,
            expose_to_agent=(name != "read_file"),
        ))
