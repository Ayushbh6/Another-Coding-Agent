from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from aca.orchestration.state import AnalyzeWorkerState, ImplementWorkerState, NeonRunState


class OrchestratorBackend(Protocol):
    """Interface that ``OrchestrationToolRegistry`` depends on.

    NeonOrchestrator satisfies this protocol via its mixin methods.
    Defining the contract explicitly removes the need for ``Any`` typing
    and documents exactly which orchestrator capabilities tools require.
    """

    _workspace_root: Path
    _repo_summary_service: Any

    def _persist_todo_state(
        self,
        state: NeonRunState | AnalyzeWorkerState | ImplementWorkerState,
        *,
        agent_id: str,
        action_type: str,
        todo_id: str | None,
        note: str | None,
        todo_items: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None: ...

    def _new_id(self, prefix: str) -> str: ...

    def _record_runtime_action(
        self,
        *,
        task_id: str | None,
        agent_id: str,
        action_type: str,
        target: str | None,
        input_json: dict[str, Any],
        result_json: dict[str, Any],
        status: str,
        error_text: str | None,
    ) -> None: ...

    def _record_artifact_write(
        self,
        *,
        task_id: str,
        agent_id: str,
        artifact_name: str,
        relative_path: str,
    ) -> None: ...

    def _update_task_metadata_after_task_rewrite(
        self,
        state: NeonRunState,
        parsed: Any,
    ) -> None: ...

    def _ensure_task_created(
        self,
        state: NeonRunState,
        parsed: Any,
    ) -> str: ...

    def _initialize_todo_state(
        self,
        state: NeonRunState,
        raw_content: str,
    ) -> list[dict[str, Any]]: ...

    def _web_search(self, *, query: str, limit: int) -> dict[str, Any]: ...
