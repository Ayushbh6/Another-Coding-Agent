from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aca.approval import AllowAllApprovalPolicy, DenyAllApprovalPolicy
from aca.config import Settings
from aca.storage import initialize_storage
from aca.workspace_tools import ToolPermissionError, WorkspaceToolContext, WorkspaceToolRegistry


class WorkspaceToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
        (self.root / ".venv").mkdir(parents=True, exist_ok=True)
        (self.root / ".venv" / "junk.py").write_text("print('ignore me')\n", encoding="utf-8")
        (self.root / ".env").write_text("SECRET=value\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("ignored_dir/\n*.log\n", encoding="utf-8")
        (self.root / "ignored_dir").mkdir(parents=True, exist_ok=True)
        (self.root / "ignored_dir" / "hidden.py").write_text("print('ignored')\n", encoding="utf-8")
        (self.root / "debug.log").write_text("debug output\n", encoding="utf-8")
        settings = Settings(
            sqlite_url=f"sqlite:///{self.root / 'aca.db'}",
            chroma_path=str(self.root / "chroma"),
            chroma_collection="aca-tools-test",
        )
        self.storage = initialize_storage(settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_read_only_registry_excludes_mutating_tools(self) -> None:
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=DenyAllApprovalPolicy(),
            )
        )

        names = [tool["function"]["name"] for tool in registry.schemas(allow_mutation=False)]
        self.assertEqual(names, ["list_files", "read_file", "search_code"])

    def test_mutating_tools_require_approval(self) -> None:
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=DenyAllApprovalPolicy(),
            )
        )

        with self.assertRaises(ToolPermissionError):
            registry.write_file("notes.txt", "blocked")

    def test_mutating_tools_work_when_approved(self) -> None:
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        result = json.loads(registry.write_file("notes.txt", "created"))
        patched = json.loads(registry.apply_patch("notes.txt", "created", "updated"))

        self.assertEqual(result["path"], "notes.txt")
        self.assertEqual(patched["replacements"], 1)
        self.assertEqual((self.root / "notes.txt").read_text(encoding="utf-8"), "updated")

    def test_list_files_ignores_generated_and_gitignored_paths(self) -> None:
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        result = json.loads(registry.list_files(".", limit=50))

        self.assertIn("src/app.py", result["files"])
        self.assertNotIn(".venv/junk.py", result["files"])
        self.assertNotIn(".env", result["files"])
        self.assertNotIn("ignored_dir/hidden.py", result["files"])
        self.assertNotIn("debug.log", result["files"])

    def test_read_file_blocks_sensitive_files(self) -> None:
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        with self.assertRaises(ValueError):
            registry.read_file(".env")

    def test_read_file_supports_partial_ranges_and_char_truncation(self) -> None:
        (self.root / "src" / "multi.py").write_text(
            "line one\nline two\nline three\nline four\n",
            encoding="utf-8",
        )
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        result = json.loads(registry.read_file("src/multi.py", start_line=2, end_line=4, max_chars=18))

        self.assertEqual(result["path"], "src/multi.py")
        self.assertEqual(result["start_line"], 2)
        self.assertEqual(result["end_line"], 4)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["excerpt_mode"], "max_chars")
        self.assertLessEqual(len(result["content"]), 18)

    def test_search_code_supports_context_and_glob_filter(self) -> None:
        (self.root / "src" / "feature.py").write_text(
            "alpha\nneedle here\nomega\n",
            encoding="utf-8",
        )
        (self.root / "notes.txt").write_text("needle elsewhere\n", encoding="utf-8")
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        result = json.loads(
            registry.search_code(
                "needle",
                path=".",
                context_lines=1,
                max_chars_per_match=40,
                path_glob="src/*.py",
            )
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["matches"][0]["path"], "src/feature.py")
        self.assertEqual(result["matches"][0]["start_line"], 1)
        self.assertEqual(result["matches"][0]["end_line"], 3)
        self.assertIn("needle here", result["matches"][0]["snippet"])

    def test_list_files_supports_max_depth(self) -> None:
        (self.root / "src" / "nested").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "nested" / "deep.py").write_text("print('deep')\n", encoding="utf-8")
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        result = json.loads(registry.list_files("src", limit=50, max_depth=0))

        self.assertIn("src/app.py", result["files"])
        self.assertNotIn("src/nested/deep.py", result["files"])

    def test_read_file_defaults_to_excerpt_for_large_files(self) -> None:
        large_lines = "".join(f"line {index}\n" for index in range(1, 260))
        (self.root / "src" / "large.py").write_text(large_lines, encoding="utf-8")
        registry = WorkspaceToolRegistry(
            WorkspaceToolContext(
                root=self.root,
                session_factory=self.storage.session_factory,
                task_id=None,
                agent_id="worker",
                approval_policy=AllowAllApprovalPolicy(),
            )
        )

        result = json.loads(registry.read_file("src/large.py"))

        self.assertEqual(result["start_line"], 1)
        self.assertEqual(result["end_line"], 200)
        self.assertTrue(result["truncated"])
