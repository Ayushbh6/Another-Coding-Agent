from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from aca.approval import ApprovalPolicy, ApprovalRequest
from aca.storage.models import Action


ToolHandler = Callable[..., str]

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
}

SENSITIVE_FILE_NAMES = {
    ".env",
}

SENSITIVE_FILE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)

DEFAULT_READ_MAX_LINES = 200
DEFAULT_READ_MAX_CHARS = 4000


class ToolPermissionError(RuntimeError):
    pass


@dataclass(slots=True)
class WorkspaceToolContext:
    root: Path
    session_factory: sessionmaker[Session]
    task_id: str | None
    agent_id: str
    approval_policy: ApprovalPolicy | None


class WorkspaceToolRegistry:
    def __init__(self, context: WorkspaceToolContext) -> None:
        self._context = context
        self._ignore_patterns = self._load_ignore_patterns()

    def schemas(self, *, allow_mutation: bool) -> list[dict[str, Any]]:
        tools = [
            _tool_schema(
                "list_files",
                "List files under a path in the workspace.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative path."},
                        "limit": {"type": "integer", "description": "Maximum files to return."},
                        "path_glob": {"type": "string", "description": "Optional glob filter for workspace-relative paths."},
                        "max_depth": {"type": "integer", "description": "Optional maximum depth under the requested path."},
                    },
                },
            ),
            _tool_schema(
                "read_file",
                "Read a file from the workspace.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative file path."},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "max_chars": {"type": "integer", "description": "Optional maximum characters to return from the selected excerpt."},
                        "tail_chars": {"type": "integer", "description": "Optional number of trailing characters to return from the selected excerpt."},
                    },
                    "required": ["path"],
                },
            ),
            _tool_schema(
                "search_code",
                "Search text in workspace files.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "context_lines": {"type": "integer", "description": "Number of surrounding lines to include for each match."},
                        "max_chars_per_match": {"type": "integer", "description": "Optional maximum characters to return for each contextual snippet."},
                        "path_glob": {"type": "string", "description": "Optional glob filter for workspace-relative paths."},
                    },
                    "required": ["pattern"],
                },
            ),
        ]

        if allow_mutation:
            tools.extend(
                [
                    _tool_schema(
                        "run_command",
                        "Run a shell command in the workspace.",
                        {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"},
                                "cwd": {"type": "string"},
                                "timeout_sec": {"type": "integer"},
                            },
                            "required": ["command"],
                        },
                    ),
                    _tool_schema(
                        "write_file",
                        "Write full file contents.",
                        {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    ),
                    _tool_schema(
                        "apply_patch",
                        "Replace text in a file with an exact-match patch.",
                        {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"},
                                "replace_all": {"type": "boolean"},
                            },
                            "required": ["path", "old_text", "new_text"],
                        },
                    ),
                ]
            )

        return tools

    def handlers(self, *, allow_mutation: bool) -> dict[str, ToolHandler]:
        handlers: dict[str, ToolHandler] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
        }
        if allow_mutation:
            handlers.update(
                {
                    "run_command": self.run_command,
                    "write_file": self.write_file,
                    "apply_patch": self.apply_patch,
                }
            )
        return handlers

    def list_files(
        self,
        path: str = ".",
        limit: int = 200,
        path_glob: str | None = None,
        max_depth: int | None = None,
    ) -> str:
        root = self._resolve_dir(path)
        files: list[str] = []
        for candidate in self._iter_files(root, path_glob=path_glob, max_depth=max_depth):
            files.append(candidate.relative_to(self._context.root.resolve()).as_posix())
            if len(files) >= limit:
                break
        result = {
            "path": root.relative_to(self._context.root.resolve()).as_posix() if root != self._context.root.resolve() else ".",
            "files": files,
            "count": len(files),
            "path_glob": path_glob,
            "max_depth": max_depth,
        }
        self._log_tool(
            "list_files",
            target=str(root),
            input_json={"path": path, "limit": limit, "path_glob": path_glob, "max_depth": max_depth},
            result_json={"count": len(files)},
        )
        return json.dumps(result)

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int | None = None,
        tail_chars: int | None = None,
    ) -> str:
        resolved = self._resolve_file(path)
        self._ensure_safe_to_read(resolved)
        text = resolved.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = max(1, start_line)
        requested_stop = end_line
        if requested_stop is None:
            requested_stop = min(len(lines), start + DEFAULT_READ_MAX_LINES - 1)
        stop = min(len(lines), requested_stop)
        excerpt = "\n".join(f"{index + 1}: {line}" for index, line in enumerate(lines[start - 1:stop], start=start - 1))
        truncated = False
        excerpt_mode = "lines"

        if end_line is None and stop < len(lines):
            truncated = True
            excerpt_mode = "lines"

        if tail_chars is not None and tail_chars >= 0 and len(excerpt) > tail_chars:
            excerpt = excerpt[-tail_chars:]
            truncated = True
            excerpt_mode = "tail_chars"
        elif max_chars is not None and max_chars >= 0 and len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars]
            truncated = True
            excerpt_mode = "max_chars"
        elif max_chars is None and tail_chars is None and len(excerpt) > DEFAULT_READ_MAX_CHARS:
            excerpt = excerpt[:DEFAULT_READ_MAX_CHARS]
            truncated = True
            excerpt_mode = "max_chars"

        result = {
            "path": resolved.relative_to(self._context.root.resolve()).as_posix(),
            "content": excerpt,
            "start_line": start,
            "end_line": stop if stop >= start else start,
            "truncated": truncated,
            "excerpt_mode": excerpt_mode,
        }
        self._log_tool(
            "read_file",
            target=str(resolved),
            input_json={
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "max_chars": max_chars,
                "tail_chars": tail_chars,
            },
            result_json={"lines": stop - start + 1 if stop >= start else 0, "truncated": truncated},
        )
        return json.dumps(result)

    def search_code(
        self,
        pattern: str,
        path: str = ".",
        limit: int = 50,
        context_lines: int = 0,
        max_chars_per_match: int | None = None,
        path_glob: str | None = None,
    ) -> str:
        root = self._resolve_dir(path)
        matches: list[dict[str, Any]] = []
        for candidate in self._iter_files(root, path_glob=path_glob):
            try:
                self._ensure_safe_to_read(candidate)
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except ValueError:
                continue
            lines = text.splitlines()
            for index, line in enumerate(lines, start=1):
                if pattern in line:
                    snippet_start = max(1, index - max(0, context_lines))
                    snippet_end = min(len(lines), index + max(0, context_lines))
                    snippet = "\n".join(
                        f"{line_no}: {snippet_line}"
                        for line_no, snippet_line in enumerate(lines[snippet_start - 1:snippet_end], start=snippet_start)
                    )
                    truncated = False
                    if max_chars_per_match is not None and max_chars_per_match >= 0 and len(snippet) > max_chars_per_match:
                        snippet = snippet[:max_chars_per_match]
                        truncated = True
                    matches.append(
                        {
                            "path": candidate.relative_to(self._context.root.resolve()).as_posix(),
                            "line": index,
                            "text": line.strip(),
                            "start_line": snippet_start,
                            "end_line": snippet_end,
                            "snippet": snippet,
                            "truncated": truncated,
                        }
                    )
                    if len(matches) >= limit:
                        result = {"matches": matches, "count": len(matches)}
                        self._log_tool(
                            "search_code",
                            target=str(root),
                            input_json={
                                "pattern": pattern,
                                "path": path,
                                "limit": limit,
                                "context_lines": context_lines,
                                "max_chars_per_match": max_chars_per_match,
                                "path_glob": path_glob,
                            },
                            result_json={"count": len(matches)},
                        )
                        return json.dumps(result)
        result = {"matches": matches, "count": len(matches)}
        self._log_tool(
            "search_code",
            target=str(root),
            input_json={
                "pattern": pattern,
                "path": path,
                "limit": limit,
                "context_lines": context_lines,
                "max_chars_per_match": max_chars_per_match,
                "path_glob": path_glob,
            },
            result_json={"count": len(matches)},
        )
        return json.dumps(result)

    def run_command(self, command: str, cwd: str = ".", timeout_sec: int = 60) -> str:
        self._require_approval("run_command", {"command": command, "cwd": cwd}, preview=command)
        working_dir = self._resolve_dir(cwd)
        completed = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        result = {
            "command": command,
            "cwd": str(working_dir.relative_to(self._context.root.resolve())),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        self._log_tool("run_command", target=str(working_dir), input_json={"command": command, "cwd": cwd}, result_json={"exit_code": completed.returncode})
        return json.dumps(result)

    def write_file(self, path: str, content: str) -> str:
        self._require_approval("write_file", {"path": path}, preview=path)
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        result = {"path": str(resolved.relative_to(self._context.root.resolve())), "bytes_written": len(content.encode("utf-8"))}
        self._log_tool("write_file", target=str(resolved), input_json={"path": path}, result_json=result)
        return json.dumps(result)

    def apply_patch(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        self._require_approval("apply_patch", {"path": path}, preview=path)
        resolved = self._resolve_file(path)
        text = resolved.read_text(encoding="utf-8")
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise ValueError("old_text was not found in the target file.")
        if occurrences > 1 and not replace_all:
            raise ValueError("old_text matched multiple locations. Set replace_all=true to replace all matches.")
        updated = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        resolved.write_text(updated, encoding="utf-8")
        result = {
            "path": str(resolved.relative_to(self._context.root.resolve())),
            "replacements": occurrences if replace_all else 1,
        }
        self._log_tool("apply_patch", target=str(resolved), input_json={"path": path, "replace_all": replace_all}, result_json=result)
        return json.dumps(result)

    def _require_approval(self, tool_name: str, payload: dict[str, Any], *, preview: str) -> None:
        if self._context.approval_policy is None:
            raise ToolPermissionError(f"{tool_name} requires an approval policy.")
        approved = self._context.approval_policy.request(
            ApprovalRequest(
                agent_id=self._context.agent_id,
                tool_name=tool_name,
                payload=payload,
                preview=preview,
            )
        )
        if not approved:
            self._log_tool(tool_name, target=None, input_json=payload, result_json={}, status="denied", error_text="Approval denied")
            raise ToolPermissionError(f"Approval denied for {tool_name}.")

    def _resolve_path(self, path: str) -> Path:
        root = self._context.root.resolve()
        candidate = (root / path).resolve()
        if root not in [candidate, *candidate.parents]:
            raise ValueError(f"Path escapes workspace root: {path}")
        return candidate

    def _resolve_file(self, path: str) -> Path:
        candidate = self._resolve_path(path)
        if not candidate.exists():
            raise FileNotFoundError(path)
        if not candidate.is_file():
            raise ValueError(f"Not a file: {path}")
        return candidate

    def _resolve_dir(self, path: str) -> Path:
        candidate = self._resolve_path(path)
        if not candidate.exists():
            raise FileNotFoundError(path)
        if not candidate.is_dir():
            raise ValueError(f"Not a directory: {path}")
        return candidate

    def _iter_files(self, root: Path, *, path_glob: str | None = None, max_depth: int | None = None):
        for candidate in sorted(root.rglob("*")):
            if candidate.is_dir():
                continue
            relative = candidate.relative_to(self._context.root.resolve())
            if self._is_ignored_path(relative):
                continue
            if path_glob and not fnmatch(relative.as_posix(), path_glob):
                continue
            if max_depth is not None and len(candidate.relative_to(root).parts) - 1 > max_depth:
                continue
            yield candidate

    def _is_ignored_path(self, relative: Path) -> bool:
        for part in relative.parts:
            if part in IGNORED_DIRECTORY_NAMES:
                return True
        relative_posix = relative.as_posix()
        ignored = False
        for pattern in self._ignore_patterns:
            is_negated = pattern.startswith("!")
            candidate = pattern[1:] if is_negated else pattern
            if self._matches_ignore_pattern(relative_posix, candidate):
                ignored = not is_negated
        if ignored:
            return True
        return self._is_sensitive_path(relative)

    def _is_sensitive_path(self, relative: Path) -> bool:
        name = relative.name
        if name in SENSITIVE_FILE_NAMES:
            return True
        if name.startswith(".env."):
            return True
        return name.endswith(SENSITIVE_FILE_SUFFIXES)

    def _ensure_safe_to_read(self, resolved: Path) -> None:
        relative = resolved.relative_to(self._context.root.resolve())
        if relative.parts and relative.parts[0] == ".aca":
            raise ValueError(
                f"Cannot read .aca internal path via read_file: {relative}. "
                "Use read_task_artifact to read task workspace files."
            )
        if self._is_ignored_path(relative):
            raise ValueError(f"Access denied to sensitive or ignored path: {relative}")

    def _load_ignore_patterns(self) -> tuple[str, ...]:
        patterns: list[str] = []
        for filename in (".gitignore", ".ignore"):
            ignore_file = self._context.root / filename
            if not ignore_file.exists() or not ignore_file.is_file():
                continue
            try:
                content = ignore_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for raw_line in content.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
        return tuple(patterns)

    def _matches_ignore_pattern(self, relative_path: str, pattern: str) -> bool:
        normalized = pattern.rstrip("/")
        if not normalized:
            return False
        if "/" not in normalized:
            return any(part == normalized or fnmatch(part, normalized) for part in Path(relative_path).parts)
        anchored = normalized.lstrip("/")
        return fnmatch(relative_path, anchored) or relative_path.startswith(f"{anchored}/")

    def _log_tool(
        self,
        tool_name: str,
        *,
        target: str | None,
        input_json: dict[str, Any],
        result_json: dict[str, Any],
        status: str = "success",
        error_text: str | None = None,
    ) -> None:
        with self._context.session_factory.begin() as session:
            session.add(
                Action(
                    id=f"act-{uuid.uuid4().hex}",
                    task_id=self._context.task_id,
                    agent_id=self._context.agent_id,
                    action_type=f"tool:{tool_name}",
                    target=target,
                    input_json=input_json,
                    result_json=result_json,
                    status=status,
                    error_text=error_text,
                    metadata_json={},
                )
            )


def _tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
