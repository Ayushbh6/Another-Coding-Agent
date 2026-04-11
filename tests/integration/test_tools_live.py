"""
Integration test: LLM-driven write → edit → run → save → read → delete flow.

Uses the real OpenRouter API. Requires OPENROUTER_API_KEY in .env.
Run with:  pytest tests/integration/test_tools_live.py -v -s -m integration

All agent activity is printed in real time to stderr using AgentConsole
(rich, coloured, bypasses pytest stdout capture entirely).

Flow:
  1. LLM writes a ~100-line Python Calculator module
  2. LLM edits it — adds gcd() via multi_update_file
  3. We run the module with python — assert exit 0
  4. LLM reads the final module and saves a Markdown summary
  5. We print the summary to the console
  6. LLM deletes both files — assert they are gone
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from aca.console import AgentConsole
from aca.llm.client import call_llm
from aca.llm.models import DEFAULT_MODEL
from aca.tools import read, write
from aca.tools.registry import PermissionMode, ToolRegistry

# ── Constants ─────────────────────────────────────────────────────────────────

TOOL_CALL_LIMIT = 30     # max tool calls per agent turn before soft-stop
REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    read.register(reg)
    write.register(reg)
    return reg


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("llm_tools_live")


# ── Agentic turn helper ───────────────────────────────────────────────────────

def run_agent_turn(
    messages: list[dict],
    system_prompt: str,
    registry: ToolRegistry,
    con: AgentConsole,
    label: str,
    tool_call_limit: int = TOOL_CALL_LIMIT,
) -> tuple[str | None, list[dict]]:
    """
    Run a single agentic turn with full real-time console output.

    - Streams LLM tokens live to the console as they arrive
    - Prints every tool call + result immediately after dispatch
    - Enforces tool_call_limit with a soft stop (forces final text response)

    Returns (final_text_content, updated_messages_list).
    """
    con.step(label)

    loop_msgs = list(messages)
    schemas = registry.get_schemas(PermissionMode.EDIT)
    calls_made = 0
    call_index = 0

    while calls_made < tool_call_limit:
        forced_stop = False

        # ── Streaming LLM call ────────────────────────────────────────────────
        con.llm_call(model=DEFAULT_MODEL, call_index=call_index, tool_count=len(schemas))
        call_index += 1

        accumulated_content: list[str] = []
        accumulated_tool_calls: list[dict] = {}  # index → partial dict

        def on_chunk(chunk) -> None:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                return
            delta = choice.delta
            # Stream text tokens live
            if delta.content:
                con.streaming_token(delta.content)
                accumulated_content.append(delta.content)

        response = call_llm(
            loop_msgs,
            model=DEFAULT_MODEL,
            system_prompt=system_prompt,
            tools=schemas,
            tool_choice="auto",
            stream=True,
            stream_callback=on_chunk,
        )

        con.streaming_done(
            stop_reason=response.stop_reason,
            tokens=response.input_tokens + response.output_tokens,
            latency_ms=response.latency_ms,
        )

        assert response.stop_reason != "error", (
            f"LLM returned error: {response.content}"
        )

        # Build assistant message
        assistant_msg: dict = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant_msg["tool_calls"] = response.tool_calls
        loop_msgs.append(assistant_msg)

        # No tool calls → turn is done
        if not response.tool_calls:
            return response.content, loop_msgs

        # ── Dispatch tool calls ───────────────────────────────────────────────
        for tc in response.tool_calls:
            tool_name = tc.get("function", {}).get("name", "?")
            raw_args = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"raw": raw_args}

            con.tool_call(tool_name, args)
            result = registry.dispatch(tc, mode=PermissionMode.EDIT)
            con.tool_result(
                tool_name,
                success=result.success,
                latency_ms=result.latency_ms,
                output=result.output,
                error=result.error,
            )
            loop_msgs.append(registry.to_tool_message(result))
            calls_made += 1

            if calls_made >= tool_call_limit:
                forced_stop = True
                break

        if forced_stop:
            break

    # ── Soft stop: force a final text response ────────────────────────────────
    if calls_made >= tool_call_limit:
        con.tool_limit_warning(tool_call_limit)
        con.llm_call(model=DEFAULT_MODEL, call_index=call_index, tool_count=0)

        def on_final_chunk(chunk) -> None:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta.content:
                con.streaming_token(choice.delta.content)

        response = call_llm(
            loop_msgs,
            model=DEFAULT_MODEL,
            system_prompt=system_prompt,
            tools=None,
            tool_choice="none",
            stream=True,
            stream_callback=on_final_chunk,
        )
        con.streaming_done(
            stop_reason=response.stop_reason,
            tokens=response.input_tokens + response.output_tokens,
            latency_ms=response.latency_ms,
        )
        loop_msgs.append({"role": "assistant", "content": response.content})
        return response.content, loop_msgs

    return response.content, loop_msgs


# ── The test ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_llm_write_edit_run_save_read_delete(registry, work_dir):
    """
    Full end-to-end agentic flow with live console output.
    All agent activity is visible in real time on stderr.
    """
    con = AgentConsole()
    con.divider()
    con.info(f"Working directory: {work_dir}")
    con.info(f"Model: {DEFAULT_MODEL}")
    con.info(f"Tool call limit per turn: {TOOL_CALL_LIMIT}")
    con.divider()

    work_str = str(work_dir)
    py_file = "calculator.py"
    md_file = "calculator_summary.md"
    py_path = work_dir / py_file
    md_path = work_dir / md_file

    system_prompt = textwrap.dedent(f"""
        You are a coding assistant. You have access to read and write tools.
        The working directory for all file operations is: {work_str}
        Always use that as repo_root when calling tools.
        Be precise, write complete code, and confirm each action with a tool call.
    """).strip()

    messages: list[dict] = []

    # ── Step 1: Write the Python module ──────────────────────────────────────
    messages.append({
        "role": "user",
        "content": (
            f"Write a Python module called '{py_file}' (~100 lines) that implements a "
            "Calculator class with methods: add, subtract, multiply, divide, power, "
            "factorial, is_prime, fibonacci(n), mean(numbers), and a main() function "
            "that demonstrates each method with print statements. "
            "Write the complete file using the write_file tool. "
            f"Use repo_root='{work_str}'."
        ),
    })

    _, messages = run_agent_turn(
        messages, system_prompt, registry, con,
        label="Step 1 — Write calculator.py",
    )

    assert py_path.exists(), f"calculator.py was not created at {py_path}"
    py_source = py_path.read_text()
    assert len(py_source.splitlines()) >= 50, "File too short — LLM likely failed to write it"
    assert "Calculator" in py_source
    assert "def add" in py_source or "def subtract" in py_source
    con.check_ok(f"calculator.py created — {len(py_source.splitlines())} lines")

    # ── Step 2: Edit — add gcd() method ──────────────────────────────────────
    messages.append({
        "role": "user",
        "content": (
            "Now add a gcd(a, b) method to the Calculator class that computes the "
            "Greatest Common Divisor using the Euclidean algorithm. "
            "Use multi_update_file (not write_file) so you only change the relevant section. "
            "Also add a call to gcd in the main() function. "
            f"repo_root='{work_str}'."
        ),
    })

    _, messages = run_agent_turn(
        messages, system_prompt, registry, con,
        label="Step 2 — Add gcd() via multi_update_file",
    )

    updated_source = py_path.read_text()
    assert "gcd" in updated_source, "gcd method was not added"
    con.check_ok(f"calculator.py updated — {len(updated_source.splitlines())} lines, gcd present")

    # ── Step 3: Run the module ────────────────────────────────────────────────
    con.step("Step 3 — Run calculator.py")
    run_result = subprocess.run(
        ["python", str(py_path)],
        capture_output=True, text=True, timeout=15,
    )
    con.info(f"exit_code={run_result.returncode}  ({len(run_result.stdout.splitlines())} lines of output)")
    if run_result.stdout:
        con.content_panel("stdout", run_result.stdout, max_chars=1000)
    if run_result.stderr:
        con.content_panel("stderr", run_result.stderr, max_chars=500)

    assert run_result.returncode == 0, (
        f"calculator.py exited {run_result.returncode}.\nstderr: {run_result.stderr}"
    )
    con.check_ok("Module ran — exit 0")

    # ── Step 4: Save Markdown summary ────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": (
            f"Read the final '{py_file}' using read_file, then write a concise Markdown "
            f"summary of the module to '{md_file}'. Include: the module's purpose, "
            "a table of all methods with signatures and one-line descriptions, "
            "and any notable implementation details. "
            f"repo_root='{work_str}'."
        ),
    })

    _, messages = run_agent_turn(
        messages, system_prompt, registry, con,
        label="Step 4 — Save calculator_summary.md",
    )

    assert md_path.exists(), f"calculator_summary.md not created at {md_path}"
    md_content = md_path.read_text()
    assert len(md_content) > 100, "Summary file too short"
    assert "#" in md_content, "No markdown headings in summary"
    con.check_ok(f"calculator_summary.md created — {len(md_content.splitlines())} lines")

    # ── Step 5: Display the .md ───────────────────────────────────────────────
    con.step("Step 5 — Summary file contents")
    con.content_panel("calculator_summary.md", md_content)

    # ── Step 6: Delete both files ─────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": (
            f"Delete both '{py_file}' and '{md_file}' using the delete_file tool. "
            f"repo_root='{work_str}'."
        ),
    })

    _, messages = run_agent_turn(
        messages, system_prompt, registry, con,
        label="Step 6 — Delete both files",
    )

    assert not py_path.exists(), f"calculator.py still exists after deletion"
    assert not md_path.exists(), f"calculator_summary.md still exists after deletion"
    con.check_ok("Both files deleted")
    con.divider()
