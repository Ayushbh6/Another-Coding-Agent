from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from aca.approval import ApprovalPolicy, ApprovalRequest
from aca.storage.models import Action
from aca.tools.access import PathAccessManager
from aca.tools.commands import classify_command


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
DEFAULT_COMMAND_MAX_CHARS = 4000
SYMBOL_PREFIXES = (
    "class",
    "def",
    "async def",
    "function",
    "const",
    "let",
    "var",
    "interface",
    "type",
    "enum",
    "struct",
    "trait",
    "impl",
    "mod",
)


class ToolPermissionError(RuntimeError):
    pass


@dataclass(slots=True)
class WorkspaceToolContext:
    root: Path
    session_factory: sessionmaker[Session]
    task_id: str | None
    agent_id: str
    approval_policy: ApprovalPolicy | None
    path_access_manager: PathAccessManager | None = None
    auto_approve_tools: frozenset[str] = frozenset()


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
                "Search text or declarations in workspace files.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "context_lines": {"type": "integer", "description": "Number of surrounding lines to include for each match."},
                        "max_chars_per_match": {"type": "integer", "description": "Optional maximum characters to return for each contextual snippet."},
                        "path_glob": {"type": "string", "description": "Optional glob filter for workspace-relative paths."},
                        "mode": {"type": "string", "enum": ["substring", "regex", "symbol"]},
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
                        "Create a new file or replace an entire file.",
                        {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "overwrite": {"type": "boolean"},
                            },
                            "required": ["path", "content"],
                        },
                    ),
                    _tool_schema(
                        "edit_file",
                        "Apply precise sequential edits to an existing file.",
                        {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "edits": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "op": {"type": "string", "enum": ["replace_exact", "replace_range", "insert_before", "insert_after"]},
                                            "old_text": {"type": "string"},
                                            "new_text": {"type": "string"},
                                            "occurrence": {"type": ["integer", "string"]},
                                            "start_line": {"type": "integer"},
                                            "end_line": {"type": "integer"},
                                            "anchor": {"type": "string"},
                                        },
                                        "required": ["op", "new_text"],
                                    },
                                },
                            },
                            "required": ["path", "edits"],
                        },
                    ),
                    _tool_schema(
                        "file_ops",
                        "Apply bounded file-system operations inside the workspace.",
                        {
                            "type": "object",
                            "properties": {
                                "operations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "op": {"type": "string", "enum": ["mkdir", "move", "copy", "delete"]},
                                            "path": {"type": "string"},
                                            "destination": {"type": "string"},
                                            "overwrite": {"type": "boolean"},
                                            "recursive": {"type": "boolean"},
                                        },
                                        "required": ["op", "path"],
                                    },
                                },
                            },
                            "required": ["operations"],
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
                    "edit_file": self.edit_file,
                    "file_ops": self.file_ops,
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
        mode: str = "substring",
    ) -> str:
        root = self._resolve_dir(path)
        if mode not in {"substring", "regex", "symbol"}:
            raise ValueError("mode must be one of: substring, regex, symbol")
        compiled_regex: re.Pattern[str] | None = None
        if mode == "regex":
            try:
                compiled_regex = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern: {exc}") from exc
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
                if not self._line_matches(line, pattern, mode=mode, compiled_regex=compiled_regex):
                    continue
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
                        "mode": mode,
                    }
                )
                if len(matches) >= limit:
                    return self._search_result(root, pattern, path, limit, context_lines, max_chars_per_match, path_glob, mode, matches)
        return self._search_result(root, pattern, path, limit, context_lines, max_chars_per_match, path_glob, mode, matches)

    def run_command(self, command: str, cwd: str = ".", timeout_sec: int = 60) -> str:
        risk = classify_command(command)
        preview = f"[{risk.value}] {command}"
        self._require_approval("run_command", {"command": command, "cwd": cwd, "risk": risk.value}, preview=preview)
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
            "risk": risk.value,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-DEFAULT_COMMAND_MAX_CHARS:],
            "stderr": completed.stderr[-DEFAULT_COMMAND_MAX_CHARS:],
        }
        self._log_tool(
            "run_command",
            target=str(working_dir),
            input_json={"command": command, "cwd": cwd, "risk": risk.value},
            result_json={"exit_code": completed.returncode, "risk": risk.value},
        )
        return json.dumps(result)

    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        self._require_approval("write_file", {"path": path, "overwrite": overwrite}, preview=path)
        resolved = self._resolve_path(path)
        existed_before = resolved.exists()
        if resolved.exists() and not overwrite:
            raise ValueError("Target file already exists. Set overwrite=true to replace it.")
        if resolved.exists() and resolved.is_dir():
            raise ValueError(f"Cannot write file over directory: {path}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        result = {
            "path": str(resolved.relative_to(self._context.root.resolve())),
            "bytes_written": len(content.encode("utf-8")),
            "overwrote_existing": existed_before,
        }
        self._log_tool("write_file", target=str(resolved), input_json={"path": path, "overwrite": overwrite}, result_json=result)
        return json.dumps(result)

    def edit_file(self, path: str, edits: list[dict[str, Any]]) -> str:
        self._require_approval("edit_file", {"path": path, "edit_count": len(edits)}, preview=path)
        resolved = self._resolve_file(path)
        text = resolved.read_text(encoding="utf-8")
        operations_applied: list[dict[str, Any]] = []
        for index, edit in enumerate(edits, start=1):
            op = str(edit.get("op", "")).strip()
            if op == "replace_exact":
                old_text = str(edit.get("old_text", ""))
                new_text = str(edit.get("new_text", ""))
                occurrence = edit.get("occurrence", 1)
                text, replacements = self._replace_exact(text, old_text, new_text, occurrence)
                operations_applied.append({"index": index, "op": op, "replacements": replacements})
                continue
            if op == "replace_range":
                start_line = int(edit.get("start_line", 0) or 0)
                end_line = int(edit.get("end_line", 0) or 0)
                new_text = str(edit.get("new_text", ""))
                text = self._replace_range(text, start_line, end_line, new_text)
                operations_applied.append({"index": index, "op": op, "start_line": start_line, "end_line": end_line})
                continue
            if op in {"insert_before", "insert_after"}:
                anchor = str(edit.get("anchor", ""))
                new_text = str(edit.get("new_text", ""))
                occurrence = edit.get("occurrence", 1)
                text = self._insert_relative(text, anchor, new_text, occurrence, before=(op == "insert_before"))
                operations_applied.append({"index": index, "op": op, "occurrence": occurrence})
                continue
            raise ValueError(f"Unsupported edit operation at index {index}: {op}")
        resolved.write_text(text, encoding="utf-8")
        result = {
            "path": str(resolved.relative_to(self._context.root.resolve())),
            "operations_applied": operations_applied,
        }
        self._log_tool("edit_file", target=str(resolved), input_json={"path": path, "edits": edits}, result_json={"operation_count": len(operations_applied)})
        return json.dumps(result)

    def file_ops(self, operations: list[dict[str, Any]]) -> str:
        self._require_approval("file_ops", {"operation_count": len(operations)}, preview=f"{len(operations)} file operation(s)")
        applied: list[dict[str, Any]] = []
        for index, operation in enumerate(operations, start=1):
            op = str(operation.get("op", "")).strip()
            path = str(operation.get("path", "")).strip()
            if not path:
                raise ValueError(f"Operation {index} is missing path.")
            if op == "mkdir":
                target = self._resolve_path(path)
                target.mkdir(parents=True, exist_ok=True)
                applied.append({"index": index, "op": op, "path": target.relative_to(self._context.root.resolve()).as_posix()})
                continue
            if op == "delete":
                target = self._resolve_path(path)
                recursive = bool(operation.get("recursive", False))
                if not target.exists():
                    raise FileNotFoundError(path)
                if target.is_dir():
                    if recursive:
                        shutil.rmtree(target)
                    else:
                        target.rmdir()
                else:
                    target.unlink()
                applied.append({"index": index, "op": op, "path": path, "recursive": recursive})
                continue
            if op in {"move", "copy"}:
                destination = str(operation.get("destination", "")).strip()
                if not destination:
                    raise ValueError(f"Operation {index} requires destination.")
                source = self._resolve_path(path)
                if not source.exists():
                    raise FileNotFoundError(path)
                destination_path = self._resolve_path(destination)
                overwrite = bool(operation.get("overwrite", False))
                if destination_path.exists():
                    if not overwrite:
                        raise ValueError(f"Destination already exists: {destination}")
                    if destination_path.is_dir() and not source.is_dir():
                        raise ValueError(f"Destination is a directory: {destination}")
                    if destination_path.is_dir():
                        shutil.rmtree(destination_path)
                    else:
                        destination_path.unlink()
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                if op == "move":
                    shutil.move(str(source), str(destination_path))
                elif source.is_dir():
                    shutil.copytree(source, destination_path)
                else:
                    shutil.copy2(source, destination_path)
                applied.append({"index": index, "op": op, "path": path, "destination": destination, "overwrite": overwrite})
                continue
            raise ValueError(f"Unsupported file operation at index {index}: {op}")
        self._log_tool("file_ops", target=str(self._context.root.resolve()), input_json={"operations": operations}, result_json={"operation_count": len(applied)})
        return json.dumps({"operations_applied": applied})

    def _search_result(
        self,
        root: Path,
        pattern: str,
        path: str,
        limit: int,
        context_lines: int,
        max_chars_per_match: int | None,
        path_glob: str | None,
        mode: str,
        matches: list[dict[str, Any]],
    ) -> str:
        result = {"matches": matches, "count": len(matches), "mode": mode}
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
                "mode": mode,
            },
            result_json={"count": len(matches), "mode": mode},
        )
        return json.dumps(result)

    def _replace_exact(self, text: str, old_text: str, new_text: str, occurrence: Any) -> tuple[str, int]:
        if not old_text:
            raise ValueError("replace_exact requires old_text.")
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise ValueError("replace_exact target was not found.")
        if occurrence == "all":
            return text.replace(old_text, new_text), occurrences
        index = int(occurrence or 1)
        if index < 1:
            raise ValueError("replace_exact occurrence must be >= 1 or 'all'.")
        if occurrences < index:
            raise ValueError(f"replace_exact occurrence {index} was not found.")
        cursor = 0
        matched = 0
        while True:
            found = text.find(old_text, cursor)
            if found < 0:
                break
            matched += 1
            if matched == index:
                updated = text[:found] + new_text + text[found + len(old_text) :]
                return updated, 1
            cursor = found + len(old_text)
        raise ValueError(f"replace_exact occurrence {index} was not found.")

    def _replace_range(self, text: str, start_line: int, end_line: int, new_text: str) -> str:
        if start_line < 1 or end_line < start_line:
            raise ValueError("replace_range requires valid 1-based start_line/end_line.")
        lines = text.splitlines(keepends=True)
        if end_line > len(lines):
            raise ValueError("replace_range end_line is beyond the file.")
        replacement_lines = new_text.splitlines(keepends=True)
        if new_text and not replacement_lines:
            replacement_lines = [new_text]
        updated_lines = lines[: start_line - 1] + replacement_lines + lines[end_line:]
        return "".join(updated_lines)

    def _insert_relative(self, text: str, anchor: str, new_text: str, occurrence: Any, *, before: bool) -> str:
        if not anchor:
            raise ValueError("insert operation requires anchor.")
        occurrences = text.count(anchor)
        if occurrences == 0:
            raise ValueError("Anchor text was not found.")
        if occurrence == "all":
            raise ValueError("insert operations require a single occurrence index.")
        index = int(occurrence or 1)
        if index < 1:
            raise ValueError("Insert occurrence must be >= 1.")
        if occurrences < index:
            raise ValueError(f"Anchor occurrence {index} was not found.")
        cursor = 0
        matched = 0
        while True:
            found = text.find(anchor, cursor)
            if found < 0:
                break
            matched += 1
            if matched == index:
                insert_at = found if before else found + len(anchor)
                return text[:insert_at] + new_text + text[insert_at:]
            cursor = found + len(anchor)
        raise ValueError(f"Anchor occurrence {index} was not found.")

    def _line_matches(self, line: str, pattern: str, *, mode: str, compiled_regex: re.Pattern[str] | None) -> bool:
        if mode == "substring":
            return pattern in line
        if mode == "regex":
            assert compiled_regex is not None
            return compiled_regex.search(line) is not None
        normalized = pattern.strip()
        if not normalized:
            return False
        for prefix in SYMBOL_PREFIXES:
            if re.search(rf"^\s*{re.escape(prefix)}\s+{re.escape(normalized)}\b", line):
                return True
        return re.search(rf"\b{re.escape(normalized)}\b", line) is not None and any(prefix in line for prefix in SYMBOL_PREFIXES)

    def _require_approval(self, tool_name: str, payload: dict[str, Any], *, preview: str) -> None:
        if tool_name in self._context.auto_approve_tools:
            return
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
        if self._is_sensitive_path(relative):
            if self._context.path_access_manager and self._context.path_access_manager.can_read(relative, sensitive=True):
                return
            raise ValueError(f"Access denied to sensitive or ignored path: {relative}")
        if self._is_ignored_path(relative):
            if self._context.path_access_manager and self._context.path_access_manager.can_read(relative, sensitive=False):
                return
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
