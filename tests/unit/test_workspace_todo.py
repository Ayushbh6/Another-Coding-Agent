"""
Unit tests for get_next_todo and advance_todo in aca/tools/workspace.py.

All tests use a real tmp_path with a real todo.md — no mocks.
"""

from __future__ import annotations

import pytest
from aca.tools.workspace import (
    advance_todo,
    create_task_workspace,
    get_next_todo,
    write_task_file,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def task(tmp_path):
    """Create a workspace with a 3-item todo.md and return (tmp_path, task_id)."""
    task_id = "test-task-001"
    create_task_workspace(task_id, repo_root=str(tmp_path))
    content = (
        f"task_id: {task_id}\n\n"
        "# Todo: Test task\n\n"
        "## Items\n\n"
        "- [ ] First item\n"
        "- [ ] Second item\n"
        "- [ ] Third item\n\n"
        "## Current step\n\n"
        "(not started)\n"
    )
    write_task_file(task_id, "todo.md", content, repo_root=str(tmp_path))
    return tmp_path, task_id


# ── get_next_todo ─────────────────────────────────────────────────────────────

class TestGetNextTodo:
    def test_returns_first_pending_item(self, task):
        root, tid = task
        result = get_next_todo(tid, repo_root=str(root))
        assert result["all_done"] is False
        assert result["item"] == "First item"
        assert result["index"] == 0
        assert result["status"] == "started"
        assert result["remaining"] == 2  # 2 still pending after claiming first

    def test_marks_item_in_progress(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        todo_content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert "- [>] First item" in todo_content
        assert "- [ ] Second item" in todo_content

    def test_updates_current_step(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        todo_content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert "First item" in todo_content.split("## Current step")[-1]

    def test_idempotent_when_item_already_in_progress(self, task):
        root, tid = task
        r1 = get_next_todo(tid, repo_root=str(root))
        r2 = get_next_todo(tid, repo_root=str(root))
        assert r2["item"] == r1["item"]
        assert r2["index"] == r1["index"]
        assert r2["status"] == "resumed"
        # File should not double-mark
        todo_content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert todo_content.count("- [>]") == 1

    def test_all_done_when_no_pending(self, task):
        root, tid = task
        # Manually write a todo with all items done
        content = (
            f"task_id: {tid}\n\n"
            "# Todo: Test\n\n"
            "## Items\n\n"
            "- [x] Done one\n"
            "- [x] Done two\n\n"
            "## Current step\n\nAll items complete.\n"
        )
        write_task_file(tid, "todo.md", content, repo_root=str(root))
        result = get_next_todo(tid, repo_root=str(root))
        assert result["all_done"] is True
        assert result["item"] is None

    def test_raises_if_no_todo_md(self, tmp_path):
        task_id = "empty-task"
        create_task_workspace(task_id, repo_root=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="todo.md"):
            get_next_todo(task_id, repo_root=str(tmp_path))


# ── advance_todo — complete ───────────────────────────────────────────────────

class TestAdvanceTodoComplete:
    def test_completes_current_item_and_starts_next(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))  # starts item 0
        result = advance_todo(tid, 0, "complete", repo_root=str(root))
        assert result["action"] == "complete"
        assert result["completed_item"] == "First item"
        assert result["advanced_to"] == "Second item"
        assert result["next_index"] == 1
        assert result["remaining"] == 1
        assert result["all_done"] is False

    def test_marks_completed_and_next_in_progress(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        advance_todo(tid, 0, "complete", repo_root=str(root))
        content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert "- [x] First item" in content
        assert "- [>] Second item" in content
        assert "- [ ] Third item" in content

    def test_all_done_after_last_item(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        advance_todo(tid, 0, "complete", repo_root=str(root))
        advance_todo(tid, 1, "complete", repo_root=str(root))
        result = advance_todo(tid, 2, "complete", repo_root=str(root))
        assert result["all_done"] is True
        assert result["advanced_to"] is None
        content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert "All items complete." in content

    def test_updates_current_step_after_advance(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        advance_todo(tid, 0, "complete", repo_root=str(root))
        content = (root / ".aca/active" / tid / "todo.md").read_text()
        after_current = content.split("## Current step")[-1]
        assert "Second item" in after_current


# ── advance_todo — skip ───────────────────────────────────────────────────────

class TestAdvanceTodoSkip:
    def test_skip_with_reason_marks_tilde(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        reason = "Already done as part of setup — file created at src/foo.py line 10."
        advance_todo(tid, 0, "skip", skip_reason=reason, repo_root=str(root))
        content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert "- [~] First item" in content
        assert "SKIPPED:" in content
        assert reason in content

    def test_skip_auto_starts_next(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        reason = "Already done as part of setup — method exists at utils.py:45."
        result = advance_todo(tid, 0, "skip", skip_reason=reason, repo_root=str(root))
        assert result["advanced_to"] == "Second item"
        content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert "- [>] Second item" in content

    def test_skip_requires_reason(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        with pytest.raises(ValueError, match="skip_reason is required"):
            advance_todo(tid, 0, "skip", repo_root=str(root))

    def test_skip_rejects_empty_reason(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        with pytest.raises(ValueError, match="skip_reason is required"):
            advance_todo(tid, 0, "skip", skip_reason="   ", repo_root=str(root))

    def test_skip_rejects_vague_reason(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        with pytest.raises(ValueError, match="too vague"):
            advance_todo(tid, 0, "skip", skip_reason="not needed", repo_root=str(root))


# ── advance_todo — gate enforcement ──────────────────────────────────────────

class TestAdvanceTodoGate:
    def test_cannot_advance_pending_item(self, task):
        """Item 1 is still [ ] — cannot advance it before starting it."""
        root, tid = task
        get_next_todo(tid, repo_root=str(root))  # starts item 0
        with pytest.raises(ValueError, match="not in progress"):
            advance_todo(tid, 1, "complete", repo_root=str(root))

    def test_cannot_skip_pending_item(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))  # starts item 0
        reason = "Already handled — written in previous pass at file.py:10."
        with pytest.raises(ValueError, match="not in progress"):
            advance_todo(tid, 1, "skip", skip_reason=reason, repo_root=str(root))

    def test_cannot_advance_already_completed_item(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        advance_todo(tid, 0, "complete", repo_root=str(root))
        # item 0 is now [x] — try to re-advance it
        with pytest.raises(ValueError):
            advance_todo(tid, 0, "complete", repo_root=str(root))

    def test_error_when_no_item_in_progress(self, task):
        """Advance without calling get_next_todo first."""
        root, tid = task
        with pytest.raises(ValueError, match="not in progress|no item is currently in progress"):
            advance_todo(tid, 0, "complete", repo_root=str(root))

    def test_invalid_action_raises(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        with pytest.raises(ValueError, match="action must be"):
            advance_todo(tid, 0, "done", repo_root=str(root))

    def test_out_of_range_index_raises(self, task):
        root, tid = task
        get_next_todo(tid, repo_root=str(root))
        with pytest.raises(ValueError, match="out of range"):
            advance_todo(tid, 99, "complete", repo_root=str(root))

    def test_full_sequential_run(self, task):
        """Complete all 3 items in order and verify final state."""
        root, tid = task
        for expected_item in ("First item", "Second item", "Third item"):
            r = get_next_todo(tid, repo_root=str(root))
            assert r["item"] == expected_item
            result = advance_todo(tid, r["index"], "complete", repo_root=str(root))
        assert result["all_done"] is True
        content = (root / ".aca/active" / tid / "todo.md").read_text()
        assert content.count("- [x]") == 3
        assert "- [ ]" not in content
        assert "- [>]" not in content
