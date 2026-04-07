from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVE_TASKS_ROOT = Path(".aca/tasks")
CANONICAL_REPO_DOCS = {"MEMORY.md", "patterns.md", "repo-structure.md"}
TASK_ARTIFACTS = {"task.md", "plan.md", "todo.md", "findings.md", "output.md", "completion.json"}
ROUTE_VALUES = {"analyze_simple", "analyze_delegated"}
IMPLEMENT_ROUTE_VALUE = "implement"
ROUTE_VALUES = ROUTE_VALUES | {IMPLEMENT_ROUTE_VALUE}
TODO_STATUSES = {"pending", "in_progress", "completed", "skipped"}


@dataclass(frozen=True, slots=True)
class ParsedTaskDocument:
    task_id: str
    intent: str
    route: str
    title: str
    normalized_task: str


class TaskWorkspaceManager:
    def __init__(self, workspace_root: Path, archive_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()
        self._archive_root = archive_root.expanduser().resolve()

    @property
    def active_tasks_root(self) -> Path:
        return (self._workspace_root / ACTIVE_TASKS_ROOT).resolve()

    def ensure_active_root(self) -> Path:
        root = self.active_tasks_root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def ensure_task_dir(self, task_id: str) -> Path:
        path = self.active_tasks_root / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_dir(self, task_id: str) -> Path:
        return (self.active_tasks_root / task_id).resolve()

    def artifact_path(self, task_id: str, artifact_name: str) -> Path:
        if artifact_name not in TASK_ARTIFACTS:
            raise ValueError(f"Unsupported task artifact: {artifact_name}")
        return (self.task_dir(task_id) / artifact_name).resolve()

    def artifact_relative_path(self, task_id: str, artifact_name: str) -> str:
        return self.artifact_path(task_id, artifact_name).relative_to(self._workspace_root).as_posix()

    def write_task_artifact(self, task_id: str, artifact_name: str, content: str) -> Path:
        path = self.artifact_path(task_id, artifact_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read_task_artifact(self, task_id: str, artifact_name: str) -> str:
        path = self.artifact_path(task_id, artifact_name)
        if not path.exists():
            raise FileNotFoundError(artifact_name)
        return path.read_text(encoding="utf-8")

    def write_completion(self, task_id: str, payload: dict[str, object]) -> Path:
        path = self.artifact_path(task_id, "completion.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def archive_destination(self, task_id: str) -> Path:
        project_hash = hashlib.sha1(str(self._workspace_root).encode("utf-8")).hexdigest()[:12]
        destination = self._archive_root / f"{self._workspace_root.name}-{project_hash}" / task_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def archive_task(self, task_id: str) -> tuple[Path, Path]:
        source = self.task_dir(task_id)
        if not source.exists():
            raise FileNotFoundError(str(source))
        destination = self.archive_destination(task_id)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return source, destination

    def write_todo_state(self, task_id: str, items: list[dict[str, Any]]) -> Path:
        return self.write_task_artifact(task_id, "todo.md", render_todo_markdown(items))


def parse_task_markdown(content: str) -> ParsedTaskDocument:
    lines = content.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError("task.md must start with YAML-style frontmatter.")

    frontmatter: dict[str, str] = {}
    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped == "---":
            closing_index = index
            break
        if ":" not in line:
            raise ValueError("Invalid task.md frontmatter line.")
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip()

    if closing_index is None:
        raise ValueError("task.md frontmatter is missing a closing delimiter.")

    body = "\n".join(lines[closing_index + 1 :]).strip()
    required_keys = ("task_id", "intent", "route", "title")
    missing = [key for key in required_keys if not frontmatter.get(key)]
    if missing:
        raise ValueError(f"task.md frontmatter is missing required keys: {', '.join(missing)}")

    route = frontmatter["route"]
    if route not in ROUTE_VALUES:
        raise ValueError(f"Unsupported task route: {route}")

    return ParsedTaskDocument(
        task_id=frontmatter["task_id"],
        intent=frontmatter["intent"],
        route=route,
        title=frontmatter["title"],
        normalized_task=body,
    )


def normalize_todo_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        title = str(item.get("title", "")).strip()
        if not title:
            raise ValueError(f"Todo item {index} is missing a title.")
        status = str(item.get("status", "pending")).strip() or "pending"
        if status not in TODO_STATUSES:
            raise ValueError(f"Todo item {index} has unsupported status: {status}")
        todo_id = str(item.get("todo_id", "")).strip()
        if not todo_id:
            raise ValueError(f"Todo item {index} is missing todo_id.")
        normalized_item = {"todo_id": todo_id, "title": title, "status": status}
        note = str(item.get("note", "")).strip()
        if note:
            normalized_item["note"] = note
        outcome = str(item.get("outcome", "")).strip()
        if outcome:
            normalized_item["outcome"] = outcome
        revision_reason = str(item.get("revision_reason", "")).strip()
        if revision_reason:
            normalized_item["revision_reason"] = revision_reason
        revision_confidence = item.get("revision_confidence")
        if revision_confidence is not None and revision_confidence != "":
            normalized_item["revision_confidence"] = float(revision_confidence)
        normalized.append(normalized_item)
    return normalized


def parse_todo_markdown(content: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- [") and "] " in stripped:
            _, remainder = stripped.split("] ", 1)
            status_token = stripped[3:stripped.index("]")]
            title = remainder.strip()
            status_map = {
                "pending": "pending",
                "in_progress": "in_progress",
                "completed": "completed",
                "skipped": "skipped",
            }
            status = status_map.get(status_token, "pending")
            items.append({"title": title, "status": status})
            continue
        if stripped.startswith("- "):
            items.append({"title": stripped[2:].strip(), "status": "pending"})
            continue
        if stripped[0].isdigit() and ". " in stripped:
            _, remainder = stripped.split(". ", 1)
            items.append({"title": remainder.strip(), "status": "pending"})

    if not items:
        raise ValueError("todo.md must contain at least one bullet or numbered todo item.")
    return items


def render_todo_markdown(items: list[dict[str, Any]]) -> str:
    normalized = normalize_todo_items(items)
    lines = ["# TODO", ""]
    for item in normalized:
        title = item["title"]
        if item["status"] in {"completed", "skipped"}:
            title = f"~~{title}~~"
        lines.append(f"- [{item['status']}] {title}")
        if item.get("note"):
            lines.append(f"  - note: {item['note']}")
        if item.get("outcome"):
            lines.append(f"  - outcome: {item['outcome']}")
        if item.get("revision_reason"):
            lines.append(f"  - revision_reason: {item['revision_reason']}")
        if item.get("revision_confidence") is not None:
            lines.append(f"  - revision_confidence: {item['revision_confidence']}")
    return "\n".join(lines).rstrip() + "\n"
