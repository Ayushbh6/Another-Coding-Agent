from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from aca.orchestration.state import (
    ANALYZE_DELEGATED_ROUTE,
    ANALYZE_SIMPLE_ROUTE,
    IMPLEMENT_ROUTE,
    PHASE_SYNTHESIZE,
    PHASE_TODO_READY,
    STEERING_TODO_REVIEW,
    AnalyzeWorkerState,
    ImplementWorkerState,
    NeonGuardrailError,
    NeonRunState,
)


TodoState = NeonRunState | AnalyzeWorkerState | ImplementWorkerState
PersistTodoCallback = Callable[..., None]
IdFactory = Callable[[str], str]


def todo_is_terminal(status: str) -> bool:
    return status in {"completed", "skipped"}


def todo_in_progress(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in items:
        if item.get("status") == "in_progress":
            return item
    return None


def todo_is_complete(items: list[dict[str, Any]]) -> bool:
    return bool(items) and all(todo_is_terminal(str(item.get("status", "pending"))) for item in items)


class TodoController:
    def __init__(
        self,
        *,
        actor: str,
        state: TodoState,
        route: str | None,
        persist: PersistTodoCallback,
        new_id: IdFactory,
    ) -> None:
        self._actor = actor
        self._state = state
        self._route = route
        self._persist = persist
        self._new_id = new_id

    def read_state(self) -> str:
        return json.dumps(
            {
                "task_id": self._state.task_id,
                "items": self._state.todo_items,
                "current_todo_id": self._state.current_todo_id,
                "all_complete": todo_is_complete(self._state.todo_items),
            }
        )

    def start(self, todo_id: str) -> str:
        self._ensure_route_allows_todo_execution()
        current = todo_in_progress(self._state.todo_items)
        if current is not None and current.get("todo_id") != todo_id:
            raise NeonGuardrailError("todo_item_required", "Another todo item is already in progress.")
        matched = next((item for item in self._state.todo_items if item["todo_id"] == todo_id), None)
        if matched is None:
            raise NeonGuardrailError("todo_item_required", f"Unknown todo_id: {todo_id}")
        if matched["status"] in {"completed", "skipped"}:
            raise NeonGuardrailError("todo_item_required", "That todo item is already finished.")
        matched["status"] = "in_progress"
        matched.pop("note", None)
        self._state.current_todo_id = todo_id
        self._state.phase = PHASE_TODO_READY
        self._persist(
            self._state,
            agent_id=self._agent_id(),
            action_type="todo_item_started",
            todo_id=todo_id,
            note=None,
            extra=None,
            todo_items=None,
        )
        label = "delegated " if self._actor == "worker" else ""
        return json.dumps({"todo_id": todo_id, "status": "in_progress", "summary": f"Started {label}todo item {todo_id}."})

    def complete(self, todo_id: str, outcome: str) -> str:
        self._ensure_todo_exists()
        matched = next((item for item in self._state.todo_items if item["todo_id"] == todo_id), None)
        if matched is None:
            raise NeonGuardrailError("todo_item_required", f"Unknown todo_id: {todo_id}")
        if matched["status"] != "in_progress":
            raise NeonGuardrailError("todo_item_required", "You may only complete the todo item that is currently in progress.")
        matched["status"] = "completed"
        matched["outcome"] = outcome.strip()
        self._state.current_todo_id = None
        self._state.phase = PHASE_SYNTHESIZE if todo_is_complete(self._state.todo_items) else PHASE_TODO_READY
        self._persist(
            self._state,
            agent_id=self._agent_id(),
            action_type="todo_item_completed",
            todo_id=todo_id,
            note=outcome.strip(),
            extra=None,
            todo_items=None,
        )
        summary = (
            "Completed the final delegated todo item. Write findings.md now."
            if self._actor == "worker" and self._route == ANALYZE_DELEGATED_ROUTE and todo_is_complete(self._state.todo_items)
            else "Completed the final delegated implement todo item. Write output.md now."
            if self._actor == "worker" and self._route == IMPLEMENT_ROUTE and todo_is_complete(self._state.todo_items)
            else "Completed the final todo item. You may now synthesize the answer."
            if todo_is_complete(self._state.todo_items)
            else "Todo item completed. Review the remaining items. Either start the next one or revise the todo with a reason and confidence score."
        )
        if self._actor == "worker" and not todo_is_complete(self._state.todo_items):
            summary = "Delegated todo item completed. Review the remaining items. Either start the next one or revise the todo with a reason and confidence score."
        return json.dumps(
            {
                "todo_id": todo_id,
                "status": "completed",
                "summary": summary,
                "remaining_items": self._remaining_items(),
                "steering_code": STEERING_TODO_REVIEW if not todo_is_complete(self._state.todo_items) else None,
            }
        )

    def skip(self, todo_id: str, reason: str) -> str:
        self._ensure_todo_exists()
        matched = next((item for item in self._state.todo_items if item["todo_id"] == todo_id), None)
        if matched is None:
            raise NeonGuardrailError("todo_item_required", f"Unknown todo_id: {todo_id}")
        current = todo_in_progress(self._state.todo_items)
        if current is not None and current.get("todo_id") not in {None, todo_id}:
            raise NeonGuardrailError("todo_item_required", "Another todo item is currently in progress.")
        matched["status"] = "skipped"
        matched["note"] = reason.strip()
        if self._state.current_todo_id == todo_id:
            self._state.current_todo_id = None
        self._state.phase = PHASE_SYNTHESIZE if todo_is_complete(self._state.todo_items) else PHASE_TODO_READY
        self._persist(
            self._state,
            agent_id=self._agent_id(),
            action_type="todo_item_skipped",
            todo_id=todo_id,
            note=reason.strip(),
            extra=None,
            todo_items=None,
        )
        summary = (
            "Skipped the final delegated todo item. Write findings.md now."
            if self._actor == "worker" and self._route == ANALYZE_DELEGATED_ROUTE and todo_is_complete(self._state.todo_items)
            else "Skipped the final delegated implement todo item. Write output.md now."
            if self._actor == "worker" and self._route == IMPLEMENT_ROUTE and todo_is_complete(self._state.todo_items)
            else "Skipped the final todo item. You may now synthesize the answer."
            if todo_is_complete(self._state.todo_items)
            else "Todo item skipped. Review the remaining items. Either start the next one or revise the todo with a reason and confidence score."
        )
        if self._actor == "worker" and not todo_is_complete(self._state.todo_items):
            summary = "Delegated todo item skipped. Review the remaining items. Either start the next one or revise the todo with a reason and confidence score."
        return json.dumps(
            {
                "todo_id": todo_id,
                "status": "skipped",
                "summary": summary,
                "remaining_items": self._remaining_items(),
                "steering_code": STEERING_TODO_REVIEW if not todo_is_complete(self._state.todo_items) else None,
            }
        )

    def revise(self, items: list[dict[str, Any]], reason: str, confidence_score: float) -> str:
        self._ensure_todo_exists()
        if todo_in_progress(self._state.todo_items) is not None:
            raise NeonGuardrailError("todo_item_required", "Finish or skip the current todo item before revising the todo.")
        if confidence_score < 0 or confidence_score > 1:
            raise NeonGuardrailError("todo_item_required", "confidence_score must be between 0 and 1.")
        revised_items: list[dict[str, Any]] = []
        for raw_item in items:
            title = str(raw_item.get("title", "")).strip()
            if not title:
                raise NeonGuardrailError("todo_item_required", "Each revised todo item must include a title.")
            revised_items.append(
                {
                    "todo_id": self._new_id("todo"),
                    "title": title,
                    "status": "pending",
                    "revision_reason": reason.strip(),
                    "revision_confidence": confidence_score,
                }
            )
        if not revised_items:
            raise NeonGuardrailError("todo_item_required", "Revised todo must contain at least one item.")
        self._state.todo_items = revised_items
        self._state.current_todo_id = None
        self._state.phase = PHASE_TODO_READY
        self._persist(
            self._state,
            agent_id=self._agent_id(),
            action_type="todo_revised",
            todo_id=None,
            note=reason.strip(),
            extra={"confidence_score": confidence_score},
            todo_items=revised_items,
        )
        label = "Delegated " if self._actor == "worker" else ""
        return json.dumps(
            {
                "summary": f"{label}todo revised. Review the new sequential items, then start the next todo item.",
                "items": revised_items,
                "reason": reason.strip(),
                "confidence_score": confidence_score,
            }
        )

    def _ensure_route_allows_todo_execution(self) -> None:
        self._ensure_todo_exists()
        if self._actor == "neon" and self._route != ANALYZE_SIMPLE_ROUTE:
            raise NeonGuardrailError("todo_item_required", "Neon may only execute todo items directly for analyze_simple.")
        if self._actor == "worker" and self._route not in {ANALYZE_DELEGATED_ROUTE, IMPLEMENT_ROUTE}:
            raise NeonGuardrailError("todo_item_required", "Worker todo execution is only available for delegated analysis or implement.")

    def _ensure_todo_exists(self) -> None:
        if not self._state.task_id or not self._state.todo_items:
            raise NeonGuardrailError("todo_item_required", "No todo state exists yet.")

    def _remaining_items(self) -> list[dict[str, Any]]:
        return [
            {"todo_id": item["todo_id"], "title": item["title"], "status": item["status"]}
            for item in self._state.todo_items
            if item["status"] in {"pending", "in_progress"}
        ]

    def _agent_id(self) -> str:
        return "worker" if self._actor == "worker" else "master"
