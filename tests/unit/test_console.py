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
    agent_console.streaming_token("Hello")
    agent_console.streaming_token(" world")
    agent_console.streaming_done(stop_reason="end_turn", tokens=12, latency_ms=10)

    rendered = buffer.getvalue()

    assert "ACA" in rendered
    assert "Hello world" in rendered
    assert "stop=" not in rendered
    assert agent_console.consume_streamed_response_flag() is True
    assert agent_console.consume_streamed_response_flag() is False


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