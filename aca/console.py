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
import re
import sys
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
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
_THINKING_STYLE    = Style(color="grey58",         italic=True)   # dim italic — distinct from main output
_META_STYLE        = Style(color="grey50")
_ASSERT_OK_STYLE   = Style(color="green",         bold=True)
_ASSERT_ERR_STYLE  = Style(color="red",           bold=True)
_WARN_STYLE        = Style(color="yellow")
_STEERING_STYLE    = Style(color="bright_magenta", bold=True)
_QUIET_LABEL_STYLE = Style(color="cyan", bold=True)
_ORDERED_LIST_RE   = re.compile(r"\d+\.\s")


class AgentConsole:
    """
    Thread-safe, pytest-safe rich console.

    All output goes to stderr (force_terminal=True ensures colours even when
    stderr is not a TTY, which happens when pytest redirects it).
    """

    def __init__(
        self,
        *,
        force_colors: bool = True,
        stderr: bool = True,
        console: "Console | None" = None,
        verbosity: str = "verbose",
    ) -> None:
        if console is not None:
            self._con = console
        else:
            self._con = Console(
                stderr=stderr,
                force_terminal=force_colors,
                highlight=False,
                markup=True,
            )
        if verbosity not in {"verbose", "quiet"}:
            raise ValueError("verbosity must be 'verbose' or 'quiet'.")
        self._verbosity = verbosity
        self._stream_buf: list[str] = []
        self._stream_started = False
        self._thinking_buf: list[str] = []
        self._thinking_started = False
        self._pending_tool_name: str | None = None
        self._pending_tool_args: dict[str, Any] = {}
        self._streamed_response_in_turn = False
        self._quiet_response_started = False
        self._status: Status | None = None

    def begin_user_turn(self) -> None:
        """Reset per-turn UI state before starting a new user-visible turn."""
        self._flush_stream()
        self._pending_tool_name = None
        self._pending_tool_args = {}
        self._streamed_response_in_turn = False
        self._quiet_response_started = False
        self._start_status()

    def consume_streamed_response_flag(self) -> bool:
        """Return whether any assistant text streamed in the current turn and reset the flag."""
        self._stop_status()
        value = self._streamed_response_in_turn
        self._streamed_response_in_turn = False
        return value

    def end_user_turn(self) -> None:
        """Stop transient UI state at the end of a user-visible turn."""
        self._stop_status()

    # ── High-level step marker ────────────────────────────────────────────────

    def step(self, label: str) -> None:
        """Print a bold section header."""
        self.begin_user_turn()
        if self._verbosity == "quiet":
            return
        self._flush_stream()
        self._con.print()
        self._con.print(Rule(f"[bold cyan]{label}[/]", style="cyan"))

    # ── LLM call markers ─────────────────────────────────────────────────────

    def llm_call(self, model: str, call_index: int, tool_count: int = 0) -> None:
        """Print when an LLM API call is about to be made."""
        if self._verbosity == "quiet":
            return
        self._flush_stream()
        tools_note = f"  tools_available={tool_count}" if tool_count else ""
        self._con.print(
            Text(f"  \u2197 LLM call #{call_index}  model={model}{tools_note}", style=_LLM_CALL_STYLE)
        )

    def llm_response(self, content: str | None, stop_reason: str = "", tokens: int = 0, latency_ms: int = 0) -> None:
        """Print the final text response from the LLM (non-streaming)."""
        if self._verbosity == "quiet":
            return
        self._flush_stream()
        meta = f"  [stop={stop_reason}  tokens={tokens}  {latency_ms}ms]"
        self._con.print(Text(meta, style=_META_STYLE))
        if content:
            preview = content[:500] + ("…" if len(content) > 500 else "")
            self._con.print(Text(f"  \u2198 {preview}", style=_LLM_RESP_STYLE))

    # ── Streaming helpers ─────────────────────────────────────────────────────

    def streaming_token(self, token: str) -> None:
        """
        Call once per streaming token.  Tokens are buffered and printed on one
        line without newlines so the output looks like a live typewriter.
        """
        self._streamed_response_in_turn = True
        if self._verbosity == "quiet":
            self._flush_thinking_stream()  # close thinking block if one was open
            self._ensure_quiet_response_header()
            self._stream_buf.append(token)
            self._stream_started = True
            self._flush_quiet_stream_blocks()
            return
        self._flush_thinking_stream()  # close thinking block if one was open
        if not self._stream_started:
            self._con.print(Text("  \u2198 ", style=_LLM_CALL_STYLE), end="")
            self._stream_started = True
        self._con.print(Text(token, style=_STREAM_STYLE), end="")
        self._stream_buf.append(token)

    def streaming_thinking_token(self, token: str) -> None:
        """
        Call once per thinking/reasoning token.  Rendered in a distinct dim
        italic style to differentiate from regular response tokens.
        """
        if self._verbosity == "quiet":
            return
        if not self._thinking_started:
            self._con.print()  # ensure fresh line
            self._con.print(Text("  ⟨thinking⟩ ", style=_THINKING_STYLE), end="")
            self._thinking_started = True
        self._con.print(Text(token, style=_THINKING_STYLE), end="")
        self._thinking_buf.append(token)

    def streaming_thinking_done(self) -> None:
        """Close the thinking block — prints a closing marker and newline."""
        if self._verbosity == "quiet":
            self._thinking_buf.clear()
            self._thinking_started = False
            return
        if self._thinking_started:
            self._con.print()  # newline after last thinking token
            self._con.print(Text("  ⟩thinking⟨", style=_THINKING_STYLE))
        self._thinking_buf.clear()
        self._thinking_started = False

    def streaming_done(self, stop_reason: str = "", tokens: int = 0, latency_ms: int = 0) -> None:
        """Call once when the stream is complete."""
        self._flush_thinking_stream()
        if self._verbosity == "quiet":
            self._flush_quiet_stream_blocks(force=True)
            if self._quiet_response_started:
                self._con.print()
            self._stream_buf.clear()
            self._stream_started = False
            return
        if self._stream_started:
            meta = f"  [stop={stop_reason}  tokens={tokens}  {latency_ms}ms]"
            self._con.print()  # newline after stream
            self._con.print(Text(meta, style=_META_STYLE))
        self._stream_buf.clear()
        self._stream_started = False

    def _flush_stream(self) -> None:
        """If a stream was partially printed, close it before printing anything else."""
        self._flush_thinking_stream()
        if self._verbosity == "quiet":
            self._flush_quiet_stream_blocks(force=True)
            self._stream_buf.clear()
            self._stream_started = False
            return
        if self._stream_started:
            self._con.print()
            self._stream_buf.clear()
            self._stream_started = False

    def _flush_thinking_stream(self) -> None:
        """If a thinking stream was partially printed, close it cleanly."""
        if self._thinking_started:
            self._con.print()  # newline
            self._thinking_buf.clear()
            self._thinking_started = False

    def _start_status(self) -> None:
        if self._verbosity != "quiet" or self._status is not None:
            return
        self._status = self._con.status("  [dim cyan]Working…[/]", spinner="dots", spinner_style="cyan")
        self._status.start()

    def _stop_status(self) -> None:
        if self._status is None:
            return
        self._status.stop()
        self._status = None

    def _flush_quiet_stream_blocks(self, *, force: bool = False) -> None:
        pending = "".join(self._stream_buf)
        if not pending:
            return
        renderable, remainder = self._split_renderable_markdown(pending, force=force)
        if renderable.strip():
            self._render_quiet_stream_markdown(renderable)
        self._stream_buf = [remainder] if remainder else []

    def _split_renderable_markdown(self, text: str, *, force: bool) -> tuple[str, str]:
        if force:
            return text, ""

        in_code_fence = False
        line_start = 0
        cutoff = 0
        idx = 0
        length = len(text)

        while idx < length:
            if text[idx] != "\n":
                idx += 1
                continue

            line = text[line_start:idx]
            stripped = line.lstrip()

            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                if not in_code_fence:
                    cutoff = idx + 1
            elif not in_code_fence:
                if idx + 1 < length and text[idx + 1] == "\n":
                    cutoff = idx + 2
                elif (
                    stripped.startswith("#")
                    or stripped.startswith("- ")
                    or stripped.startswith("* ")
                    or stripped.startswith("> ")
                    or _ORDERED_LIST_RE.match(stripped)
                ):
                    cutoff = idx + 1

            line_start = idx + 1
            idx += 1

        if cutoff == 0:
            sentence_cutoff = self._sentence_cutoff(text)
            if sentence_cutoff > 0:
                cutoff = sentence_cutoff

        return text[:cutoff], text[cutoff:]

    def _sentence_cutoff(self, text: str) -> int:
        if len(text) < 24:
            return 0
        if text.count("```") % 2 != 0:
            return 0
        if text.count("**") % 2 != 0:
            return 0
        if text.count("`") % 2 != 0:
            return 0

        for marker in ("! ", "? ", ". "):
            idx = text.rfind(marker)
            if idx >= 0:
                return idx + len(marker)
        return 0

    def _render_quiet_stream_markdown(self, markdown_text: str) -> None:
        self._ensure_quiet_response_header()
        body = markdown_text.strip()
        if not body:
            return
        self._con.print(Padding(Markdown(body), (0, 0, 0, 2)))

    def _ensure_quiet_response_header(self) -> None:
        if self._quiet_response_started:
            return
        self._stop_status()
        self._con.print()
        self._con.print(Text("  ACA", style=_QUIET_LABEL_STYLE))
        self._quiet_response_started = True

    # ── Tool call / result ────────────────────────────────────────────────────

    def tool_call(self, tool_name: str, args: dict) -> None:
        """Print when a tool call is being dispatched."""
        self._flush_stream()
        self._pending_tool_name = tool_name
        self._pending_tool_args = dict(args)
        if self._verbosity == "quiet":
            return
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
            Text(f"  \u2699 tool_call  {tool_name}", style=_TOOL_CALL_STYLE),
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
        args = self._pending_tool_args if self._pending_tool_name == tool_name else {}
        if self._verbosity == "quiet":
            summary = self._summarize_activity(
                tool_name=tool_name,
                args=args,
                success=success,
                output=output,
                error=error,
            )
            style = _TOOL_OK_STYLE if success else _TOOL_ERR_STYLE
            self._con.print(Text(f"  • {summary}", style=style))
            self._pending_tool_name = None
            self._pending_tool_args = {}
            return
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
                Text(f"  \u2713 {tool_name}", style=_TOOL_OK_STYLE),
                Text(f"  {latency_ms}ms  {out_str}", style=_META_STYLE),
            )
        else:
            self._con.print(
                Text(f"  \u2717 {tool_name}  FAILED  {latency_ms}ms", style=_TOOL_ERR_STYLE),
                Text(f"  error: {error}", style=_TOOL_ERR_STYLE),
            )
        self._pending_tool_name = None
        self._pending_tool_args = {}

    def _summarize_activity(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        output: Any,
        error: str | None,
    ) -> str:
        if not success:
            detail = (error or "tool failed").strip()
            if len(detail) > 120:
                detail = detail[:117] + "..."
            return f"{tool_name} failed: {detail}"

        data = output if isinstance(output, dict) else {}
        path = str(data.get("path") or args.get("path") or "").strip()
        filename = str(data.get("filename") or args.get("filename") or "").strip()

        if tool_name == "read_file":
            return f"Reviewed {path or 'a file'}"
        if tool_name == "read_files":
            slices = data.get("total_slices_read") or 0
            if slices:
                label = "slice" if slices == 1 else "file slices"
                return f"Read {slices} {label}"
            return "Read file slices"
        if tool_name == "list_files":
            target = str(args.get("path") or data.get("path") or ".")
            pattern = str(args.get("pattern") or data.get("pattern") or "").strip()
            if pattern:
                return f"Listed files ({pattern}) under {target}"
            return f"Listed files under {target}"
        if tool_name == "search_repo":
            query = str(args.get("query") or data.get("query") or "").strip()
            file_pattern = str(args.get("file_pattern") or data.get("file_pattern") or "").strip()
            if query and file_pattern:
                return f"Searched repo ({file_pattern}) for {query}"
            if query:
                return f"Searched repo for {query}"
            return "Searched repo"
        if tool_name == "get_file_outline":
            return f"Mapped {path or 'file structure'}"
        if tool_name == "search_memory":
            return "Checked memory"
        if tool_name == "run_command":
            command = str(data.get("command") or args.get("command") or "").strip()
            if command:
                if len(command) > 80:
                    command = command[:77] + "..."
                return f"Ran command: {command}"
            return "Ran command"
        if tool_name == "run_tests":
            return "Ran tests"
        if tool_name == "write_file":
            action = str(data.get("action") or "updated")
            verb = "Created" if action == "created" else "Updated"
            return f"{verb} {path or 'file'}"
        if tool_name == "update_file":
            return f"Updated {path or 'file'}"
        if tool_name == "multi_update_file":
            return f"Applied grouped edits to {path or 'file'}"
        if tool_name == "edit_file":
            edits_applied = data.get("edits_applied")
            if isinstance(edits_applied, int) and edits_applied > 0:
                noun = "replacement" if edits_applied == 1 else "replacements"
                return f"Edited {edits_applied} exact {noun} in {path or 'file'}"
            return f"Edited {path or 'file'}"
        if tool_name == "apply_patch":
            return f"Applied patch to {path or 'file'}"
        if tool_name == "delete_file":
            return f"Deleted {path or 'file'}"
        if tool_name == "create_task_workspace":
            workspace_path = str(data.get("workspace_path") or ".aca/active")
            return f"Opened task workspace {workspace_path}"
        if tool_name == "write_task_file":
            if filename:
                return f"Saved {filename}"
            return "Saved task artifact"
        if tool_name == "get_next_todo":
            item = str(data.get("item") or "").strip()
            return f"Started: {item}" if item else "Checked next task step"
        if tool_name == "advance_todo":
            action = str(data.get("action") or args.get("action") or "advanced")
            item = str(data.get("completed_item") or data.get("skipped_item") or "").strip()
            verb = "Completed" if action == "complete" else "Skipped"
            return f"{verb}: {item}" if item else f"{verb} current task step"
        return f"Finished {tool_name}"

    # ── Assertions / checks ───────────────────────────────────────────────────

    def check_ok(self, label: str) -> None:
        self._con.print(Text(f"  \u2713 {label}", style=_ASSERT_OK_STYLE))

    def check_fail(self, label: str) -> None:
        self._con.print(Text(f"  \u2717 {label}", style=_ASSERT_ERR_STYLE))

    def steering_junction(self, junction_key: str, user_msg: str) -> None:
        """
        Print a visible phase transition for the human operator.

        `junction_key` is one of: route, write_artifacts, execute_simple,
        delegate, force_reply, worker_start, worker_done, james_wake —
        picks a quirky ASCII prefix.
        """
        if self._verbosity == "quiet":
            return
        self._flush_stream()
        icons: dict[str, str] = {
            "route": "[o_o]",
            "write_artifacts": "[|]",
            "execute_simple": "[#>]",
            "delegate": "[=>]",
            "force_reply": "[!!]",
            "worker_start": "[@]",
            "worker_done": "[+]",
            "james_wake": "[<<]",
        }
        prefix = icons.get(junction_key, "[*]")
        self._con.print()
        self._con.print(
            Text(f"  {prefix} STEERING — {user_msg}", style=_STEERING_STYLE)
        )

    def warn(self, msg: str) -> None:
        self._con.print(Text(f"  \u26a0 {msg}", style=_WARN_STYLE))

    # ── Generic output ────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._flush_stream()
        self._con.print(Text(f"  {msg}", style=_META_STYLE))

    def content_panel(self, title: str, body: str, max_chars: int = 2000) -> None:
        """Print a bordered panel with content (e.g. file contents)."""
        self._flush_stream()
        truncated = body[:max_chars] + ("\n… (truncated)" if len(body) > max_chars else "")
        self._con.print(Panel(truncated, title=title, border_style="dim"))

    def make_stream_callback(self, thinking: bool = False):
        """
        Return a stream_callback callable suitable for passing to call_llm.

        The returned function accepts a ChatCompletionChunk and pipes any
        text delta tokens through self.streaming_token() for live display.
        When thinking=True, reasoning_content deltas are routed through
        self.streaming_thinking_token() for distinct visual styling.
        """
        def _callback(chunk) -> None:
            try:
                delta = chunk.choices[0].delta
                if delta is None:
                    return
                # Thinking / reasoning tokens — rendered in distinct dim italic style
                if thinking:
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        self.streaming_thinking_token(reasoning)
                        return
                # Regular content tokens
                if delta.content:
                    self.streaming_thinking_done()  # close thinking block if open
                    self.streaming_token(delta.content)
            except Exception:  # noqa: BLE001
                pass
        return _callback

    def tool_limit_warning(self, limit: int) -> None:
        self._flush_stream()
        if self._verbosity == "quiet":
            self._con.print(
                Text("  • Tool budget reached. Wrapping up with the best available answer.", style=_WARN_STYLE)
            )
            return
        self._con.print(
            Text(f"  \u26a0 tool_call_limit={limit} reached — forcing final text response", style=_WARN_STYLE)
        )

    def divider(self) -> None:
        self._flush_stream()
        if self._verbosity == "quiet":
            return
        self._con.print(Rule(style="dim"))


# ── Module-level default instance (convenience) ───────────────────────────────

_default_console: AgentConsole | None = None


def get_console() -> AgentConsole:
    """Return the module-level default AgentConsole, creating it on first call."""
    global _default_console
    if _default_console is None:
        _default_console = AgentConsole()
    return _default_console
