"""
AgentConsole — real-time rich console for ACA agent activity.

Writes to stderr so output is NEVER captured by pytest or redirected pipes
that only capture stdout.  Every method flushes immediately.

Usage
-----
    from aca.console import AgentConsole
    con = AgentConsole()

    con.step("Step 1: Write calculator.py")
    con.llm_call(model="minimax/minimax-m2.7:nitro", call_index=0)
    con.streaming_token("Hello")   # call repeatedly during stream
    con.streaming_done()
    con.tool_call("write_file", {"path": "calculator.py", ...})
    con.tool_result("write_file", success=True, latency_ms=3, output={"action": "created"})
    con.llm_response("File created successfully.")
    con.divider()
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.text import Text


# ── Colour palette ────────────────────────────────────────────────────────────

_STEP_STYLE        = Style(color="bright_cyan",   bold=True)
_LLM_CALL_STYLE    = Style(color="bright_blue",   bold=True)
_TOOL_CALL_STYLE   = Style(color="bright_yellow", bold=True)
_TOOL_OK_STYLE     = Style(color="bright_green")
_TOOL_ERR_STYLE    = Style(color="bright_red",    bold=True)
_LLM_RESP_STYLE    = Style(color="white")
_STREAM_STYLE      = Style(color="bright_white",  dim=True)
_META_STYLE        = Style(color="grey50")
_ASSERT_OK_STYLE   = Style(color="green",         bold=True)
_ASSERT_ERR_STYLE  = Style(color="red",           bold=True)
_WARN_STYLE        = Style(color="yellow")


class AgentConsole:
    """
    Thread-safe, pytest-safe rich console.

    All output goes to stderr (force_terminal=True ensures colours even when
    stderr is not a TTY, which happens when pytest redirects it).
    """

    def __init__(self, *, force_colors: bool = True) -> None:
        self._con = Console(
            stderr=True,
            force_terminal=force_colors,
            highlight=False,
            markup=True,
        )
        self._stream_buf: list[str] = []
        self._stream_started = False

    # ── High-level step marker ────────────────────────────────────────────────

    def step(self, label: str) -> None:
        """Print a bold section header."""
        self._flush_stream()
        self._con.print()
        self._con.print(Rule(f"[bold cyan]{label}[/]", style="cyan"))

    # ── LLM call markers ─────────────────────────────────────────────────────

    def llm_call(self, model: str, call_index: int, tool_count: int = 0) -> None:
        """Print when an LLM API call is about to be made."""
        self._flush_stream()
        tools_note = f"  tools_available={tool_count}" if tool_count else ""
        self._con.print(
            Text(f"  ↗ LLM call #{call_index}  model={model}{tools_note}", style=_LLM_CALL_STYLE)
        )

    def llm_response(self, content: str | None, stop_reason: str = "", tokens: int = 0, latency_ms: int = 0) -> None:
        """Print the final text response from the LLM (non-streaming)."""
        self._flush_stream()
        meta = f"  [stop={stop_reason}  tokens={tokens}  {latency_ms}ms]"
        self._con.print(Text(meta, style=_META_STYLE))
        if content:
            preview = content[:500] + ("…" if len(content) > 500 else "")
            self._con.print(Text(f"  ↙ {preview}", style=_LLM_RESP_STYLE))

    # ── Streaming helpers ─────────────────────────────────────────────────────

    def streaming_token(self, token: str) -> None:
        """
        Call once per streaming token.  Tokens are buffered and printed on one
        line without newlines so the output looks like a live typewriter.
        """
        if not self._stream_started:
            self._con.print(Text("  ↙ ", style=_LLM_CALL_STYLE), end="")
            self._stream_started = True
        self._con.print(Text(token, style=_STREAM_STYLE), end="")
        self._stream_buf.append(token)

    def streaming_done(self, stop_reason: str = "", tokens: int = 0, latency_ms: int = 0) -> None:
        """Call once when the stream is complete."""
        if self._stream_started:
            meta = f"  [stop={stop_reason}  tokens={tokens}  {latency_ms}ms]"
            self._con.print()  # newline after stream
            self._con.print(Text(meta, style=_META_STYLE))
        self._stream_buf.clear()
        self._stream_started = False

    def _flush_stream(self) -> None:
        """If a stream was partially printed, close it before printing anything else."""
        if self._stream_started:
            self._con.print()
            self._stream_buf.clear()
            self._stream_started = False

    # ── Tool call / result ────────────────────────────────────────────────────

    def tool_call(self, tool_name: str, args: dict) -> None:
        """Print when a tool call is being dispatched."""
        self._flush_stream()
        # Show args compactly — truncate large values
        compact: dict = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 120:
                compact[k] = v[:120] + "…"
            elif isinstance(v, list) and len(v) > 5:
                compact[k] = v[:5] + [f"…+{len(v)-5} more"]
            else:
                compact[k] = v
        args_str = json.dumps(compact, indent=None, ensure_ascii=False)
        if len(args_str) > 200:
            args_str = args_str[:200] + "…"
        self._con.print(
            Text(f"  ⚙ tool_call  {tool_name}", style=_TOOL_CALL_STYLE),
            Text(f"  args: {args_str}", style=_META_STYLE),
        )

    def tool_result(
        self,
        tool_name: str,
        success: bool,
        latency_ms: int,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        """Print after a tool has been dispatched and the result is back."""
        self._flush_stream()
        if success:
            # Summarise output compactly
            if isinstance(output, dict):
                summary_keys = ["action", "path", "bytes_written", "lines_returned",
                                "match_count", "deleted", "task_id", "edits_applied",
                                "success", "exit_code"]
                summary = {k: output[k] for k in summary_keys if k in output}
                out_str = json.dumps(summary, ensure_ascii=False) if summary else str(output)[:120]
            else:
                out_str = str(output)[:120] if output else ""
            self._con.print(
                Text(f"  ✓ {tool_name}", style=_TOOL_OK_STYLE),
                Text(f"  {latency_ms}ms  {out_str}", style=_META_STYLE),
            )
        else:
            self._con.print(
                Text(f"  ✗ {tool_name}  FAILED  {latency_ms}ms", style=_TOOL_ERR_STYLE),
                Text(f"  error: {error}", style=_TOOL_ERR_STYLE),
            )

    # ── Assertions / checks ───────────────────────────────────────────────────

    def check_ok(self, label: str) -> None:
        self._con.print(Text(f"  ✓ {label}", style=_ASSERT_OK_STYLE))

    def check_fail(self, label: str) -> None:
        self._con.print(Text(f"  ✗ {label}", style=_ASSERT_ERR_STYLE))

    def warn(self, msg: str) -> None:
        self._con.print(Text(f"  ⚠ {msg}", style=_WARN_STYLE))

    # ── Generic output ────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._flush_stream()
        self._con.print(Text(f"  {msg}", style=_META_STYLE))

    def content_panel(self, title: str, body: str, max_chars: int = 2000) -> None:
        """Print a bordered panel with content (e.g. file contents)."""
        self._flush_stream()
        truncated = body[:max_chars] + ("\n… (truncated)" if len(body) > max_chars else "")
        self._con.print(Panel(truncated, title=title, border_style="dim"))

    def tool_limit_warning(self, limit: int) -> None:
        self._flush_stream()
        self._con.print(
            Text(f"  ⚠ tool_call_limit={limit} reached — forcing final text response", style=_WARN_STYLE)
        )

    def divider(self) -> None:
        self._flush_stream()
        self._con.print(Rule(style="dim"))


# ── Module-level default instance (convenience) ───────────────────────────────

_default_console: AgentConsole | None = None


def get_console() -> AgentConsole:
    """Return the module-level default AgentConsole, creating it on first call."""
    global _default_console
    if _default_console is None:
        _default_console = AgentConsole()
    return _default_console
