from __future__ import annotations

from io import StringIO

from rich.console import Console

from aca.console import AgentConsole


def test_quiet_console_hides_internal_turn_markers() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.step("JAMES turn #1")
    agent_console.llm_call("minimax/minimax-m2.7:nitro", 0, 5)
    agent_console.tool_call("read_file", {"path": "docs/ARCHITECTURE.md"})
    agent_console.tool_result(
        "read_file",
        success=True,
        latency_ms=4,
        output={"path": "docs/ARCHITECTURE.md", "lines_returned": 120},
    )
    agent_console.steering_junction("route", "routing prompt")

    rendered = buffer.getvalue()

    assert "Reviewed docs/ARCHITECTURE.md" in rendered
    assert "LLM call" not in rendered
    assert "STEERING" not in rendered
    assert "JAMES turn" not in rendered


def test_quiet_console_streams_assistant_tokens_cleanly() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.begin_user_turn()
    agent_console.streaming_token("**Hello**")
    agent_console.streaming_token(" world")
    agent_console.streaming_done(stop_reason="end_turn", tokens=12, latency_ms=10)

    rendered = buffer.getvalue()

    assert "ACA" in rendered
    assert "Hello world" in rendered
    assert "**Hello**" not in rendered
    assert "stop=" not in rendered
    assert agent_console.consume_streamed_response_flag() is True
    assert agent_console.consume_streamed_response_flag() is False


def test_quiet_console_flushes_stream_as_markdown_before_tool_updates() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.begin_user_turn()
    agent_console.streaming_token("## Heading")
    agent_console.tool_call("read_file", {"path": "README.md"})
    agent_console.tool_result(
        "read_file",
        success=True,
        latency_ms=4,
        output={"path": "README.md", "lines_returned": 10},
    )

    rendered = buffer.getvalue()

    assert "ACA" in rendered
    assert "Heading" in rendered
    assert "## Heading" not in rendered
    assert "Reviewed README.md" in rendered


def test_quiet_console_flushes_plain_sentence_before_stream_end() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.begin_user_turn()
    agent_console.streaming_token("This is a plain response sentence. ")

    rendered = buffer.getvalue()

    assert "ACA" in rendered
    assert "This is a plain response sentence." in rendered


def test_quiet_console_shows_aca_header_on_first_token_even_before_flush() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.begin_user_turn()
    agent_console.streaming_token("Hi")

    rendered = buffer.getvalue()

    assert "ACA" in rendered
    assert rendered.count("ACA") == 1
    assert "Hi" not in rendered


def test_quiet_console_summarizes_search_repo_with_query() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.tool_call("search_repo", {"query": "challenger"})
    agent_console.tool_result(
        tool_name="search_repo",
        success=True,
        latency_ms=5,
        output={"query": "challenger", "match_count": 3, "matches": []},
    )

    rendered = buffer.getvalue()
    assert "Searched repo for challenger" in rendered


def test_quiet_console_summarizes_search_repo_with_file_pattern() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.tool_call("search_repo", {"query": "challenger", "file_pattern": "*.py"})
    agent_console.tool_result(
        tool_name="search_repo",
        success=True,
        latency_ms=5,
        output={"query": "challenger", "file_pattern": "*.py", "match_count": 2, "matches": []},
    )

    rendered = buffer.getvalue()
    assert "Searched repo (*.py) for challenger" in rendered


def test_quiet_console_summarizes_read_files() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.tool_call("read_files", {"requests": [{"path": "README.md"}, {"path": "pyproject.toml"}]})
    agent_console.tool_result(
        tool_name="read_files",
        success=True,
        latency_ms=5,
        output={"total_slices_read": 2, "results": []},
    )

    rendered = buffer.getvalue()
    assert "Read 2 file slices" in rendered


def test_quiet_console_summarizes_edit_file() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.tool_call("edit_file", {"path": "aca/agents/challenger.py"})
    agent_console.tool_result(
        tool_name="edit_file",
        success=True,
        latency_ms=5,
        output={"path": "aca/agents/challenger.py", "edits_applied": 2},
    )

    rendered = buffer.getvalue()
    assert "Edited 2 exact replacements in aca/agents/challenger.py" in rendered


def test_tool_result_accepts_tool_name_keyword() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, highlight=False)
    agent_console = AgentConsole(console=console, verbosity="quiet")

    agent_console.tool_call("read_file", {"path": "docs/ARCHITECTURE.md"})
    agent_console.tool_result(
        tool_name="read_file",
        success=True,
        latency_ms=4,
        output={"path": "docs/ARCHITECTURE.md", "lines_returned": 120},
    )

    rendered = buffer.getvalue()
    assert "Reviewed docs/ARCHITECTURE.md" in rendered
