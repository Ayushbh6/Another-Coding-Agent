from __future__ import annotations

from aca.orchestration.state import (
    STEERING_IMPLEMENT_WORKER_CONSOLIDATING,
    STEERING_ORIENTATION_DELEGATED,
    STEERING_ORIENTATION_SIMPLE,
    STEERING_PLAN_REQUIRED,
    STEERING_PRETASK_LIMIT_REACHED,
    STEERING_READ_FINDINGS,
    STEERING_READ_OUTPUT,
    STEERING_ROUTE_RECOVERY,
    STEERING_SPAWN_WORKER_REQUIRED,
    STEERING_TASK_REQUIRED_NOW,
    STEERING_TODO_ITEM_REQUIRED,
    STEERING_TODO_REQUIRED,
    STEERING_TODO_REVIEW,
    STEERING_WORKER_CONSOLIDATING,
)


FRONTEND_STEERING_MESSAGES = {
    STEERING_PRETASK_LIMIT_REACHED: "That was the quick scout pass. Time to lock the task in with `task.md` or answer directly.",
    STEERING_TASK_REQUIRED_NOW: "Neon has enough context to stop wandering. Next move: write `task.md`.",
    STEERING_ORIENTATION_SIMPLE: "Orientation is complete. Turn that into a clean `todo.md`, then work the list one item at a time.",
    STEERING_ORIENTATION_DELEGATED: "Orientation is complete. Write `plan.md`, derive `todo.md`, then hand the deeper scan to the worker.",
    STEERING_PLAN_REQUIRED: "The route is delegated, so the plan comes first. Write `plan.md` before the todo.",
    STEERING_TODO_REQUIRED: "The next durable step is the todo. Write `todo.md` so the work has a real execution contract.",
    STEERING_TODO_ITEM_REQUIRED: "The todo is ready. Pick exactly one item, mark it in progress, and work that item only.",
    STEERING_TODO_REVIEW: "Nice, one item down. Review the remaining list and either continue with the next item or revise the plan with a reason and confidence score.",
    STEERING_SPAWN_WORKER_REQUIRED: "Everything is staged. Next move is the worker handoff.",
    STEERING_READ_FINDINGS: "The worker has landed. Read `findings.md` and `completion.json`, then synthesize the answer.",
    STEERING_READ_OUTPUT: "The implement worker has landed. Read `output.md` and `completion.json`, then synthesize the answer.",
    STEERING_WORKER_CONSOLIDATING: "The worker has enough evidence and is now stitching it into `findings.md`.",
    STEERING_IMPLEMENT_WORKER_CONSOLIDATING: "The implement worker has finished the todo and is now stitching the execution summary into `output.md`.",
    STEERING_ROUTE_RECOVERY: "The current route is invalid for the task state. Re-anchor on the allowed workflow tools.",
}


def steering_message(code: str, fallback: str | None = None) -> str:
    return FRONTEND_STEERING_MESSAGES.get(code, fallback or code)
