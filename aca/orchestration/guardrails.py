from __future__ import annotations

from aca.orchestration.state import (
    ANALYZE_DELEGATED_ROUTE,
    ANALYZE_SIMPLE_ROUTE,
    IMPLEMENT_ROUTE,
    PHASE_SYNTHESIZE,
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
    NeonRunState,
)
from aca.orchestration.steering import steering_message
from aca.orchestration.todo import todo_in_progress, todo_is_complete


class GuardrailMixin:
    """Guardrail feedback, recovery, and steering instruction helpers.

    The host class must provide ``_workspace_root``.
    """

    def _should_stream_live_text(self, state: NeonRunState) -> bool:
        if state.task_id is None:
            return True
        if state.phase == PHASE_SYNTHESIZE:
            return True
        if state.route == ANALYZE_SIMPLE_ROUTE and state.todo_written and todo_is_complete(state.todo_items):
            return True
        return False

    def _guardrail_feedback_code(self, state: NeonRunState) -> str | None:
        if state.task_id is None:
            return None

        if state.route == ANALYZE_SIMPLE_ROUTE:
            if not state.todo_written:
                return STEERING_TODO_REQUIRED
            if todo_in_progress(state.todo_items) is None and not todo_is_complete(state.todo_items):
                return STEERING_TODO_ITEM_REQUIRED
            if not todo_is_complete(state.todo_items):
                return STEERING_TODO_REVIEW
            return None

        if state.route == ANALYZE_DELEGATED_ROUTE:
            if not state.plan_written:
                return STEERING_PLAN_REQUIRED
            if not state.todo_written:
                return STEERING_TODO_REQUIRED
            if not state.worker_spawned:
                return STEERING_SPAWN_WORKER_REQUIRED
            return None

        if state.route == IMPLEMENT_ROUTE:
            if not state.plan_written:
                return STEERING_PLAN_REQUIRED
            if not state.todo_written:
                return STEERING_TODO_REQUIRED
            if not state.worker_spawned:
                return STEERING_SPAWN_WORKER_REQUIRED
            return None

        return STEERING_ROUTE_RECOVERY

    def _guardrail_recovery_instruction(self, state: NeonRunState, feedback_code: str, failure_count: int) -> str:
        base = self._run_instructions(state)
        if feedback_code in {STEERING_PRETASK_LIMIT_REACHED, STEERING_TASK_REQUIRED_NOW}:
            if failure_count <= 1:
                return (
                    f"{base}\n\n"
                    "The scout pass is done. Either answer the user directly as chat, or "
                    "write task.md to begin analysis. The choice is yours."
                )
            return (
                f"{base}\n\n"
                "You must either answer the user directly or call write_task_artifact(task.md).\n"
                "If you choose a task route, use this task.md frontmatter shape exactly:\n"
                f"---\n"
                f"task_id: {state.planned_task_id}\n"
                "intent: <analyze | implement>\n"
                "route: <analyze_simple | analyze_delegated | implement>\n"
                "title: <short task title>\n"
                "---\n"
                "<normalized task statement>"
            )
        if feedback_code == STEERING_PLAN_REQUIRED:
            return f"{base}\n\nYour next action must be write_task_artifact(plan.md). Do not read artifacts or answer yet."
        if feedback_code == STEERING_TODO_REQUIRED:
            return f"{base}\n\nYour next action must be write_task_artifact(todo.md). Do not answer yet."
        if feedback_code == STEERING_TODO_ITEM_REQUIRED:
            return f"{base}\n\nYour next action must be start_todo_item for the first pending item."
        if feedback_code == STEERING_SPAWN_WORKER_REQUIRED:
            if state.route == IMPLEMENT_ROUTE:
                return f"{base}\n\nYour next action must be spawn_implement_worker. Do not answer yet."
            return f"{base}\n\nYour next action must be spawn_analyze_worker. Do not answer yet."
        if feedback_code == STEERING_READ_FINDINGS:
            return f"{base}\n\nYour next actions must be read_task_artifact(findings.md) and read_task_artifact(completion.json), then synthesize."
        if feedback_code == STEERING_READ_OUTPUT:
            return f"{base}\n\nYour next actions must be read_task_artifact(output.md) and read_task_artifact(completion.json), then synthesize."
        return f"{base}\n\nDo not abandon the task. Correct the route or next action now using the allowed tools."

    def _steering_instructions(self, state: NeonRunState) -> str:
        base = (
            f"If you take a non-chat route, use task_id `{state.planned_task_id}` in task.md frontmatter exactly. "
            "Do not invent another task id."
        )
        if not state.task_written:
            return (
                f"{base}\n"
                "You may use get_repo_summary and up to 2 pre-task repo reads before you must either answer as chat or write task.md."
            )
        if state.route == ANALYZE_SIMPLE_ROUTE:
            if not state.todo_written:
                return f"{base}\nTask created. You may now do up to 2 orientation reads, then write todo.md."
            if todo_is_complete(state.todo_items):
                return f"{base}\nTodo execution is complete. Synthesize the final answer now."
            current = todo_in_progress(state.todo_items)
            if current is None:
                return f"{base}\nTodo is ready. Start one todo item before continuing the analysis."
            return f"{base}\nCurrent todo item in progress: {current['title']}. Continue or finish it before moving on."
        if state.route == ANALYZE_DELEGATED_ROUTE:
            if not state.plan_written:
                return f"{base}\nTask created. You may now do up to 2 orientation reads, then write plan.md."
            if not state.todo_written:
                return f"{base}\nPlan is ready. Write todo.md next."
            if not state.worker_spawned:
                return f"{base}\nTodo is ready. Your next action must be spawn_analyze_worker."
            return f"{base}\nThe worker has finished. Read findings.md and completion.json, then synthesize the final answer."
        if state.route == IMPLEMENT_ROUTE:
            if not state.plan_written:
                return f"{base}\nTask created. You may now do up to 2 orientation reads, then write plan.md."
            if not state.todo_written:
                return f"{base}\nPlan is ready. Write todo.md next."
            if not state.worker_spawned:
                return f"{base}\nTodo is ready. Your next action must be spawn_implement_worker."
            return f"{base}\nThe worker has finished. Read output.md and completion.json, then synthesize the final answer."
        return f"{base}\nUse the allowed workflow tools to recover the route."

    def _user_identity_instruction(self, state: NeonRunState) -> str:
        if state.user_name:
            return f"You are currently assisting the user named {state.user_name}."
        return "You are currently assisting the active CLI user for this conversation."

    def _run_instructions(self, state: NeonRunState) -> str:
        return f"{self._user_identity_instruction(state)}\n\n{self._steering_instructions(state)}"

    def _phase_restart_instruction(self, state: NeonRunState, steering_code: str | None) -> str:
        base = self._run_instructions(state)
        if steering_code == STEERING_PRETASK_LIMIT_REACHED:
            return (
                f"{base}\n\n"
                "The scout pass is complete. Do not call list_files, read_file, or search_code now. "
                "Your next step must be either a direct chat answer or write_task_artifact(task.md)."
            )
        if steering_code == STEERING_ORIENTATION_SIMPLE:
            return (
                f"{base}\n\n"
                "Orientation is complete. Do not call repo-read tools now. "
                "Your next step must be write_task_artifact(todo.md)."
            )
        if steering_code == STEERING_ORIENTATION_DELEGATED:
            return (
                f"{base}\n\n"
                "Orientation is complete. Do not call repo-read tools now. "
                "Your next step must be write_task_artifact(plan.md), then write_task_artifact(todo.md)."
            )
        if steering_code == STEERING_PLAN_REQUIRED:
            return f"{base}\n\nYour next step must be write_task_artifact(plan.md)."
        if steering_code == STEERING_TODO_REQUIRED:
            return f"{base}\n\nYour next step must be write_task_artifact(todo.md)."
        if steering_code == STEERING_TODO_ITEM_REQUIRED:
            return f"{base}\n\nYour next step must be read_todo_state or start_todo_item for the first pending item."
        if steering_code == STEERING_SPAWN_WORKER_REQUIRED:
            if state.route == IMPLEMENT_ROUTE:
                return f"{base}\n\nYour next step must be spawn_implement_worker."
            return f"{base}\n\nYour next step must be spawn_analyze_worker."
        if steering_code == STEERING_READ_FINDINGS:
            return f"{base}\n\nYour next step must be read_task_artifact(findings.md) and read_task_artifact(completion.json)."
        if steering_code == STEERING_READ_OUTPUT:
            return f"{base}\n\nYour next step must be read_task_artifact(output.md) and read_task_artifact(completion.json)."
        return base
