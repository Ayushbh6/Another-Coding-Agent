"""
Unit tests for aca/tools/read.py

All tests use the `repo` fixture (a fresh git-initialised tmp dir).
No mocks, no network — every assertion exercises the real tool logic.
"""

import pytest

from aca.tools.read import (
    get_file_outline,
    list_files,
    read_file,
    read_files,
    search_repo,
)


# ── read_file ─────────────────────────────────────────────────────────────────

class TestReadFile:
    def test_reads_full_file(self, repo):
        f = repo / "hello.py"
        f.write_text("line1\nline2\nline3\n")
        result = read_file("hello.py", repo_root=str(repo))
        assert result["content"] == "line1\nline2\nline3\n"
        assert result["total_lines"] == 3
        assert result["lines_returned"] == 3
        assert result["truncated"] is False

    def test_start_line_only(self, repo):
        f = repo / "f.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
        result = read_file("f.txt", repo_root=str(repo), start_line=5)
        assert result["start_line"] == 5
        assert result["content"].startswith("L5")
        assert result["total_lines"] == 10

    def test_end_line_only(self, repo):
        f = repo / "f.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
        result = read_file("f.txt", repo_root=str(repo), end_line=3)
        assert result["end_line"] == 3
        assert result["lines_returned"] == 3
        assert "L4" not in result["content"]

    def test_start_and_end_line(self, repo):
        f = repo / "f.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
        result = read_file("f.txt", repo_root=str(repo), start_line=3, end_line=5)
        assert result["lines_returned"] == 3
        assert "L3" in result["content"]
        assert "L5" in result["content"]
        assert "L6" not in result["content"]

    def test_max_lines(self, repo):
        f = repo / "f.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 21)) + "\n")
        result = read_file("f.txt", repo_root=str(repo), start_line=1, max_lines=5)
        assert result["lines_returned"] == 5
        assert result["truncated"] is True

    def test_default_cap_2000_lines(self, repo):
        f = repo / "big.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 2501)) + "\n")
        result = read_file("big.txt", repo_root=str(repo))
        assert result["lines_returned"] == 2000
        assert result["total_lines"] == 2500
        assert result["truncated"] is True

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            read_file("missing.txt", repo_root=str(repo))

    def test_blocks_path_traversal(self, repo):
        with pytest.raises(ValueError, match="escapes the repo root"):
            read_file("../../etc/passwd", repo_root=str(repo))

    def test_blocks_sensitive_env_file(self, repo):
        (repo / ".env").write_text("SECRET=abc")
        with pytest.raises(ValueError, match="sensitive"):
            read_file(".env", repo_root=str(repo))

    def test_returns_total_lines_for_empty_file(self, repo):
        (repo / "empty.txt").write_text("")
        result = read_file("empty.txt", repo_root=str(repo))
        assert result["total_lines"] == 0
        assert result["lines_returned"] == 0
        assert result["content"] == ""


class TestReadFiles:
    def test_reads_single_file_via_requests(self, repo):
        (repo / "hello.py").write_text("line1\nline2\n")
        result = read_files(
            requests=[{"path": "hello.py"}],
            repo_root=str(repo),
        )
        assert result["total_slices_read"] == 1
        assert result["results"][0]["content"] == "line1\nline2\n"

    def test_reads_multiple_spans_in_order(self, repo):
        (repo / "f.txt").write_text("a\nb\nc\nd\n")
        result = read_files(
            requests=[
                {"path": "f.txt", "start_line": 3, "end_line": 4},
                {"path": "f.txt", "start_line": 1, "end_line": 1},
            ],
            repo_root=str(repo),
        )
        assert [item["content"] for item in result["results"]] == ["c\nd\n", "a\n"]

    def test_global_budget_omits_later_requests(self, repo):
        (repo / "a.txt").write_text("1\n2\n3\n")
        (repo / "b.txt").write_text("4\n5\n6\n")
        result = read_files(
            requests=[{"path": "a.txt"}, {"path": "b.txt"}],
            repo_root=str(repo),
            max_total_lines=3,
        )
        assert result["total_slices_read"] == 1
        assert result["omitted_count"] == 1
        assert "max_total_lines=3" in result["omitted_reason"]

    def test_rejects_empty_requests(self, repo):
        with pytest.raises(ValueError, match="empty"):
            read_files(requests=[], repo_root=str(repo))


# ── list_files ────────────────────────────────────────────────────────────────

class TestListFiles:
    def test_lists_top_level(self, repo):
        (repo / "a.py").write_text("x")
        (repo / "b.md").write_text("y")
        result = list_files(repo_root=str(repo))
        names = [e["name"] for e in result["entries"]]
        assert "a.py" in names
        assert "b.md" in names

    def test_glob_pattern_filters(self, repo):
        (repo / "a.py").write_text("x")
        (repo / "b.md").write_text("y")
        result = list_files(repo_root=str(repo), pattern="*.py")
        names = [e["name"] for e in result["entries"]]
        assert all(n.endswith(".py") for n in names)
        assert "b.md" not in names

    def test_max_depth_respected(self, repo):
        sub = repo / "subdir" / "deep"
        sub.mkdir(parents=True)
        (sub / "deep_file.py").write_text("x")
        result = list_files(repo_root=str(repo), max_depth=1)
        names = [e["name"] for e in result["entries"]]
        assert not any("deep_file" in n for n in names)

    def test_hidden_excluded_by_default(self, repo):
        (repo / ".hidden").write_text("x")
        (repo / "visible.py").write_text("y")
        result = list_files(repo_root=str(repo))
        names = [e["name"] for e in result["entries"]]
        assert ".hidden" not in names
        assert "visible.py" in names

    def test_include_hidden(self, repo):
        (repo / ".hidden").write_text("x")
        result = list_files(repo_root=str(repo), include_hidden=True)
        names = [e["name"] for e in result["entries"]]
        assert ".hidden" in names

    def test_entries_have_correct_types(self, repo):
        (repo / "myfile.py").write_text("x")
        (repo / "subdir").mkdir()
        result = list_files(repo_root=str(repo))
        by_name = {e["name"]: e for e in result["entries"]}
        assert by_name["myfile.py"]["type"] == "file"
        assert by_name["myfile.py"]["size_bytes"] is not None
        assert by_name["subdir"]["type"] == "dir"
        assert by_name["subdir"]["size_bytes"] is None

    def test_directory_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            list_files("nonexistent/", repo_root=str(repo))


# ── search_repo ───────────────────────────────────────────────────────────────

class TestSearchRepo:
    def test_finds_literal_match(self, repo):
        (repo / "code.py").write_text("def hello():\n    return 'world'\n")
        result = search_repo("hello", repo_root=str(repo))
        assert result["match_count"] >= 1
        assert any("hello" in m["text"] for m in result["matches"])

    def test_returns_file_and_line(self, repo):
        (repo / "code.py").write_text("alpha\nbeta\ngamma\n")
        result = search_repo("beta", repo_root=str(repo))
        match = result["matches"][0]
        assert match["file"].endswith("code.py")
        assert match["line"] == 2
        assert "beta" in match["text"]

    def test_context_lines_populated(self, repo):
        (repo / "code.py").write_text("before\ntarget\nafter\n")
        result = search_repo("target", repo_root=str(repo), context_lines=1)
        match = result["matches"][0]
        assert "before" in match["context_before"]
        assert "after" in match["context_after"]

    def test_file_pattern_filter(self, repo):
        (repo / "code.py").write_text("needle\n")
        (repo / "notes.md").write_text("needle\n")
        result = search_repo("needle", repo_root=str(repo), file_pattern="*.py")
        assert all(m["file"].endswith(".py") for m in result["matches"])

    def test_case_insensitive(self, repo):
        (repo / "code.py").write_text("Hello World\n")
        result = search_repo("hello", repo_root=str(repo), case_insensitive=True)
        assert result["match_count"] >= 1

    def test_case_sensitive_misses(self, repo):
        (repo / "code.py").write_text("Hello World\n")
        result = search_repo("hello", repo_root=str(repo), case_insensitive=False)
        assert result["match_count"] == 0

    def test_regex_pattern(self, repo):
        (repo / "code.py").write_text("foo123\nbar456\n")
        result = search_repo(r"foo\d+", repo_root=str(repo))
        assert result["match_count"] >= 1
        assert any("foo123" in m["text"] for m in result["matches"])

    def test_no_matches_returns_empty(self, repo):
        (repo / "code.py").write_text("nothing here\n")
        result = search_repo("zzznomatch", repo_root=str(repo))
        assert result["match_count"] == 0
        assert result["matches"] == []


# ── get_file_outline ──────────────────────────────────────────────────────────

class TestGetFileOutline:
    def test_python_top_level_function(self, repo):
        (repo / "mod.py").write_text(
            "def greet(name):\n    return f'hi {name}'\n\ndef farewell():\n    pass\n"
        )
        result = get_file_outline("mod.py", repo_root=str(repo))
        assert result["language"] == "python"
        names = [e["name"] for e in result["outline"]]
        assert "greet" in names
        assert "farewell" in names
        for entry in result["outline"]:
            assert entry["kind"] == "function"

    def test_python_class_with_methods(self, repo):
        src = (
            "class Animal:\n"
            "    def speak(self):\n"
            "        pass\n"
            "    def move(self):\n"
            "        pass\n"
        )
        (repo / "animal.py").write_text(src)
        result = get_file_outline("animal.py", repo_root=str(repo))
        kinds = {e["name"]: e["kind"] for e in result["outline"]}
        assert kinds["Animal"] == "class"
        assert kinds["speak"] == "method"
        assert kinds["move"] == "method"
        parents = {e["name"]: e["parent"] for e in result["outline"]}
        assert parents["speak"] == "Animal"

    def test_python_returns_line_numbers(self, repo):
        (repo / "mod.py").write_text("def first():\n    pass\n\ndef second():\n    pass\n")
        result = get_file_outline("mod.py", repo_root=str(repo))
        by_name = {e["name"]: e for e in result["outline"]}
        assert by_name["first"]["line"] == 1
        assert by_name["second"]["line"] == 4

    def test_python_syntax_error_returns_gracefully(self, repo):
        (repo / "bad.py").write_text("def incomplete(\n")
        result = get_file_outline("bad.py", repo_root=str(repo))
        assert result["language"] == "python"
        assert "parse_error" in result
        assert result["outline"] == []

    def test_generic_js_functions(self, repo):
        src = (
            "function doThing() {\n"
            "  return 1;\n"
            "}\n"
            "class MyClass {\n"
            "  method() {}\n"
            "}\n"
        )
        (repo / "app.js").write_text(src)
        result = get_file_outline("app.js", repo_root=str(repo))
        assert result["language"] == "generic"
        names = [e["name"] for e in result["outline"]]
        assert "doThing" in names
        assert "MyClass" in names

    def test_total_lines_reported(self, repo):
        content = "def f():\n    pass\n"
        (repo / "mod.py").write_text(content)
        result = get_file_outline("mod.py", repo_root=str(repo))
        assert result["total_lines"] == content.count("\n")

    def test_file_not_found(self, repo):
        with pytest.raises(FileNotFoundError):
            get_file_outline("missing.py", repo_root=str(repo))
