"""
Unit tests for aca/tools/write.py

All tests use the `repo` fixture (a fresh git-initialised tmp dir).
No mocks, no network — every assertion exercises the real tool logic.
"""

import pytest

from aca.tools.write import (
    apply_patch,
    create_file,
    delete_file,
    multi_update_file,
    update_file,
    write_file,
)


# ── write_file ────────────────────────────────────────────────────────────────

class TestWriteFile:
    def test_creates_new_file(self, repo):
        result = write_file("new.py", "print('hello')\n", repo_root=str(repo))
        assert result["action"] == "created"
        assert (repo / "new.py").read_text() == "print('hello')\n"

    def test_overwrites_existing_file(self, repo):
        (repo / "existing.py").write_text("old content\n")
        result = write_file("existing.py", "new content\n", repo_root=str(repo))
        assert result["action"] == "overwritten"
        assert (repo / "existing.py").read_text() == "new content\n"

    def test_creates_parent_dirs(self, repo):
        result = write_file("deep/nested/file.py", "x\n", repo_root=str(repo))
        assert (repo / "deep" / "nested" / "file.py").exists()
        assert result["action"] == "created"

    def test_returns_bytes_written(self, repo):
        content = "hello world\n"
        result = write_file("f.txt", content, repo_root=str(repo))
        assert result["bytes_written"] == len(content.encode("utf-8"))

    def test_blocks_path_traversal(self, repo):
        with pytest.raises(ValueError, match="escapes the repo root"):
            write_file("../../evil.py", "x", repo_root=str(repo))

    def test_blocks_aca_directory(self, repo):
        (repo / ".aca" / "active").mkdir(parents=True)
        with pytest.raises(ValueError, match=".aca"):
            write_file(".aca/active/task.md", "x", repo_root=str(repo))

    def test_blocks_sensitive_file(self, repo):
        with pytest.raises(ValueError, match="sensitive"):
            write_file(".env", "SECRET=123", repo_root=str(repo))

    def test_overwrite_false_fails_if_exists(self, repo):
        (repo / "existing.py").write_text("old\n")
        with pytest.raises(FileExistsError, match="already exists"):
            write_file("existing.py", "new\n", repo_root=str(repo), overwrite=False)

    def test_overwrite_false_creates_new_file(self, repo):
        result = write_file("brand_new.py", "# new\n", repo_root=str(repo), overwrite=False)
        assert result["action"] == "created"
        assert (repo / "brand_new.py").read_text() == "# new\n"


# ── create_file (alias for write_file overwrite=False) ────────────────────────

class TestCreateFile:
    def test_creates_file(self, repo):
        result = create_file("brand_new.py", "# new\n", repo_root=str(repo))
        assert (repo / "brand_new.py").read_text() == "# new\n"
        assert result["bytes_written"] > 0
        assert result["action"] == "created"

    def test_fails_if_file_exists(self, repo):
        (repo / "existing.py").write_text("old\n")
        with pytest.raises(FileExistsError, match="already exists"):
            create_file("existing.py", "new\n", repo_root=str(repo))

    def test_creates_parent_dirs(self, repo):
        create_file("sub/dir/file.py", "x\n", repo_root=str(repo))
        assert (repo / "sub" / "dir" / "file.py").exists()


# ── update_file ───────────────────────────────────────────────────────────────

class TestUpdateFile:
    def test_replaces_unique_string(self, repo):
        (repo / "code.py").write_text("def old_name():\n    pass\n")
        result = update_file(
            "code.py",
            old_string="def old_name():",
            new_string="def new_name():",
            repo_root=str(repo),
        )
        assert result["replaced"] == 1
        assert "new_name" in (repo / "code.py").read_text()

    def test_fails_if_not_found(self, repo):
        (repo / "code.py").write_text("hello\n")
        with pytest.raises(ValueError, match="not found"):
            update_file("code.py", old_string="zzz", new_string="aaa", repo_root=str(repo))

    def test_fails_if_not_unique(self, repo):
        (repo / "code.py").write_text("x = 1\nx = 1\n")
        with pytest.raises(ValueError, match="appears 2 times"):
            update_file("code.py", old_string="x = 1", new_string="x = 2", repo_root=str(repo))

    def test_preserves_surrounding_content(self, repo):
        content = "line1\ntarget_line\nline3\n"
        (repo / "f.py").write_text(content)
        update_file("f.py", old_string="target_line", new_string="replaced_line", repo_root=str(repo))
        result = (repo / "f.py").read_text()
        assert "line1" in result
        assert "replaced_line" in result
        assert "line3" in result
        assert "target_line" not in result

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            update_file("missing.py", "x", "y", repo_root=str(repo))


# ── multi_update_file ─────────────────────────────────────────────────────────

class TestMultiUpdateFile:
    def test_applies_multiple_edits(self, repo):
        content = "alpha\nbeta\ngamma\n"
        (repo / "f.py").write_text(content)
        result = multi_update_file(
            "f.py",
            edits=[
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "beta",  "new_string": "BETA"},
            ],
            repo_root=str(repo),
        )
        assert result["edits_applied"] == 2
        text = (repo / "f.py").read_text()
        assert "ALPHA" in text
        assert "BETA" in text
        assert "gamma" in text  # untouched

    def test_edits_are_ordered_sequentially(self, repo):
        # Second edit targets a string introduced by the first edit
        (repo / "f.py").write_text("hello world\n")
        multi_update_file(
            "f.py",
            edits=[
                {"old_string": "hello", "new_string": "goodbye"},
                {"old_string": "goodbye world", "new_string": "goodbye universe"},
            ],
            repo_root=str(repo),
        )
        assert (repo / "f.py").read_text() == "goodbye universe\n"

    def test_rolls_back_on_failure(self, repo):
        original = "alpha\nbeta\ngamma\n"
        (repo / "f.py").write_text(original)
        with pytest.raises(ValueError):
            multi_update_file(
                "f.py",
                edits=[
                    {"old_string": "alpha", "new_string": "ALPHA"},
                    {"old_string": "ZZZMISSING", "new_string": "X"},  # will fail
                ],
                repo_root=str(repo),
            )
        # File must be untouched
        assert (repo / "f.py").read_text() == original

    def test_fails_on_non_unique_old_string(self, repo):
        (repo / "f.py").write_text("x\nx\n")
        with pytest.raises(ValueError, match="appears 2 times"):
            multi_update_file("f.py", edits=[{"old_string": "x", "new_string": "y"}], repo_root=str(repo))

    def test_empty_edits_raises(self, repo):
        (repo / "f.py").write_text("x\n")
        with pytest.raises(ValueError, match="empty"):
            multi_update_file("f.py", edits=[], repo_root=str(repo))

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            multi_update_file("missing.py", edits=[{"old_string": "x", "new_string": "y"}], repo_root=str(repo))


# ── apply_patch ───────────────────────────────────────────────────────────────

class TestApplyPatch:
    def _make_patch(self, old_lines: list[str], new_lines: list[str], fname: str = "f.py") -> str:
        """Build a minimal unified diff string."""
        import difflib
        return "".join(difflib.unified_diff(
            [l + "\n" for l in old_lines],
            [l + "\n" for l in new_lines],
            fromfile=f"a/{fname}",
            tofile=f"b/{fname}",
        ))

    def test_applies_simple_change(self, repo):
        original = ["line1", "line2", "line3"]
        updated = ["line1", "LINE2", "line3"]
        (repo / "f.py").write_text("\n".join(original) + "\n")
        patch = self._make_patch(original, updated)
        result = apply_patch("f.py", patch=patch, repo_root=str(repo))
        assert result["success"] is True
        assert "LINE2" in (repo / "f.py").read_text()
        assert "line2" not in (repo / "f.py").read_text()

    def test_applies_multiline_change(self, repo):
        original = ["a", "b", "c", "d", "e"]
        updated =  ["a", "B", "C", "d", "e"]
        (repo / "f.py").write_text("\n".join(original) + "\n")
        patch = self._make_patch(original, updated)
        apply_patch("f.py", patch=patch, repo_root=str(repo))
        text = (repo / "f.py").read_text()
        assert "B" in text and "C" in text
        assert "b\n" not in text and "c\n" not in text

    def test_empty_patch_raises(self, repo):
        (repo / "f.py").write_text("x\n")
        with pytest.raises(ValueError, match="No valid diff"):
            apply_patch("f.py", patch="", repo_root=str(repo))

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            apply_patch("missing.py", patch="--- a\n+++ b\n", repo_root=str(repo))


# ── delete_file ───────────────────────────────────────────────────────────────

class TestDeleteFile:
    def test_deletes_existing_file(self, repo):
        f = repo / "todelete.py"
        f.write_text("x\n")
        result = delete_file("todelete.py", repo_root=str(repo))
        assert result["deleted"] is True
        assert not f.exists()

    def test_fails_if_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            delete_file("missing.py", repo_root=str(repo))

    def test_fails_for_directory(self, repo):
        (repo / "mydir").mkdir()
        with pytest.raises(ValueError, match="directory"):
            delete_file("mydir", repo_root=str(repo))

    def test_blocks_path_traversal(self, repo):
        with pytest.raises(ValueError, match="escapes the repo root"):
            delete_file("../../important.txt", repo_root=str(repo))

    def test_blocks_sensitive_file(self, repo):
        (repo / ".env").write_text("KEY=val")
        with pytest.raises(ValueError, match="sensitive"):
            delete_file(".env", repo_root=str(repo))

    def test_blocks_aca_directory(self, repo):
        aca = repo / ".aca" / "active" / "task-001"
        aca.mkdir(parents=True)
        (aca / "task.md").write_text("# Task")
        with pytest.raises(ValueError, match=".aca"):
            delete_file(".aca/active/task-001/task.md", repo_root=str(repo))


# ── example_guidelines write guard ───────────────────────────────────────────

from pathlib import Path as _Path

_GLOBAL_EG = str(_Path.home() / ".aca" / "example_guidelines")


class TestExampleGuidelinesWriteGuard:
    """
    All write tools must hard-block any path inside ~/.aca/example_guidelines/.
    Tests use absolute paths since that's the real global location now.
    """

    def test_write_file_blocks_global_example_guidelines(self, repo):
        with pytest.raises(ValueError, match="example_guidelines"):
            write_file(
                f"{_GLOBAL_EG}/task.md",
                "hacked",
                repo_root=str(repo),
            )

    def test_write_file_blocks_new_file_in_global_guidelines(self, repo):
        with pytest.raises(ValueError, match="example_guidelines"):
            write_file(
                f"{_GLOBAL_EG}/new.md",
                "# new",
                repo_root=str(repo),
            )

    def test_update_file_blocks_global_example_guidelines(self, repo):
        with pytest.raises(ValueError, match="example_guidelines"):
            update_file(
                f"{_GLOBAL_EG}/task.md",
                old_string="# Example",
                new_string="# hacked",
                repo_root=str(repo),
            )

    def test_delete_file_blocks_global_example_guidelines(self, repo):
        with pytest.raises(ValueError, match="example_guidelines"):
            delete_file(
                f"{_GLOBAL_EG}/task.md",
                repo_root=str(repo),
            )
