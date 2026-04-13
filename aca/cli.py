"""
ACA — Another Coding Agent
Command-line interface.

Entry point: ``aca`` (installed via ``pip install -e .`` inside a venv).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from aca.agents.james import JamesAgent
from aca.console import AgentConsole
from aca.db import open_db
from aca.llm.models import DEFAULT_MODEL, OpenAIModels, OpenRouterModels
from aca.llm.providers import ProviderName
from aca.tools import build_registry
from aca.tools.registry import PermissionMode


# ── Paths ─────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path.home() / ".aca" / "config.json"
_DB_PATH = Path("data") / "aca.db"
_REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements.txt"


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_venv() -> None:
    """
    Abort with a clear message if we are NOT running inside a virtual
    environment.  Checked before any Rich import so the message is plain text
    (Rich may not be installed yet).
    """
    in_venv = (
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("CONDA_DEFAULT_ENV")
        or (sys.prefix != getattr(sys, "base_prefix", sys.prefix))
    )
    if not in_venv:
        print("\n  ACA needs a virtual environment.")
        print("  Please create and activate one first:\n")
        print("    python -m venv .venv")
        print("    source .venv/bin/activate   # macOS / Linux")
        print("    .venv\\Scripts\\activate      # Windows")
        print("\n  Then re-run:  pip install -e .  and  aca\n")
        sys.exit(1)


def _bootstrap_packages(con: Console) -> None:
    """
    Ensure all requirements are installed.  Runs ``pip install -r
    requirements.txt -q`` if the sentinel package ``openai`` is missing,
    or if requirements.txt is newer than the last pip install marker.
    """
    marker = Path(sys.prefix) / ".aca_deps_installed"
    req_path = _REQUIREMENTS_PATH

    needs_install = (
        not marker.exists()
        or (req_path.exists() and req_path.stat().st_mtime > marker.stat().st_mtime)
    )
    # Also check openai is importable as a quick sanity check
    try:
        import openai  # noqa: F401
    except ImportError:
        needs_install = True

    if needs_install and req_path.exists():
        con.print("  [dim]Installing / verifying packages…[/]")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            con.print(f"  [red]Package install failed:[/]\n{result.stderr.strip()}")
            sys.exit(1)
        marker.touch()
        con.print("  [green]Packages ready.[/]\n")


def _setup_global_aca_dir() -> None:
    """
    Ensure ~/.aca/ and ~/.aca/example_guidelines/ exist.
    Template files are created only if they don't already exist.
    """
    guidelines_dir = Path.home() / ".aca" / "example_guidelines"
    guidelines_dir.mkdir(parents=True, exist_ok=True)

    templates: dict[str, str] = {
        "task.md": """\
task_id: <short-slug>
type: analysis | implement
delegation: simple | delegated
status: active
created_at: <ISO 8601 datetime>
session_id: <uuid>
turn_id: <uuid>

## Description
<1–3 sentences: what needs to be done and why>
""",
        "plan.md": """\
task_id: <short-slug>

## Objective
<One specific sentence: what success looks like>

## Steps
1. <Step — action and expected outcome>
2. <Step — ...>

## Risks and Mitigations
- <Risk>: <How to handle it>
""",
        "todo.md": """\
task_id: <short-slug>

## Todos
- [ ] <Step 1 — concrete, specific, independently verifiable>
- [ ] <Step 2>

## Current step
<updated automatically by get_next_todo / advance_todo — do not write manually>
""",
        "findings.md": """\
task_id: <short-slug>
status: complete
completed_at: <ISO 8601 datetime>

## Findings
<Detailed results of analysis or investigation>

## Summary
<One paragraph conclusion>
""",
        "output.md": """\
task_id: <short-slug>
status: complete
completed_at: <ISO 8601 datetime>

## Output
<Description of what was produced, created, or changed>

## Files Changed
- `<path>` — <what changed>
""",
    }
    for filename, content in templates.items():
        dest = guidelines_dir / filename
        if not dest.exists():
            dest.write_text(content)


def _setup_local_aca_dir(repo_path: Path) -> None:
    """Ensure .aca/active/ and .aca/exports/ exist inside the repo."""
    (repo_path / ".aca" / "active").mkdir(parents=True, exist_ok=True)
    (repo_path / ".aca" / "exports").mkdir(parents=True, exist_ok=True)


# ── Cyberpunk banner ──────────────────────────────────────────────────────────

_BANNER = """\
[bright_cyan bold]
  ░░░░░░  ░░░░░░  ░░░░░
  ██████  ██      ██   ██
  ██████  ██      ███████
  ██  ██  ██      ██   ██
  ██  ██  ██████  ██   ██
[/bright_cyan bold]
[cyan]  ╔══════════════════════════════════════════╗
  ║   A N O T H E R   C O D I N G   A G E N T  ║
  ╚══════════════════════════════════════════╝[/cyan]
[dim]  ⚡  jack in. stay sharp. ship it.[/dim]
"""

_DIVIDER_STYLE = "dim cyan"


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ── All available models ──────────────────────────────────────────────────────

def _all_models() -> list[tuple[str, str]]:
    """Return list of (display_name, model_id) for all configured models."""
    models: list[tuple[str, str]] = []
    for attr in vars(OpenRouterModels):
        if not attr.startswith("_"):
            val = getattr(OpenRouterModels, attr)
            if isinstance(val, str):
                models.append((f"[OpenRouter]  {attr}", val))
    for attr in vars(OpenAIModels):
        if not attr.startswith("_"):
            val = getattr(OpenAIModels, attr)
            if isinstance(val, str):
                models.append((f"[OpenAI]     {attr}", val))
    return models


# ── Repo detection ────────────────────────────────────────────────────────────

def _detect_repo(cwd: Path) -> Path:
    """Walk up from cwd looking for a .git directory. Fall back to cwd."""
    current = cwd.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return cwd.resolve()


# ── Bootstrap scan ────────────────────────────────────────────────────────────

def _bootstrap_scan(repo_path: Path) -> str:
    """
    Lightweight read-only scan of the repo.
    Returns a formatted context string injected into James's system prompt.
    """
    lines: list[str] = [
        f"Repo path: {repo_path}",
        f"Repo name: {repo_path.name}",
    ]

    # Top-level structure
    try:
        entries = sorted(repo_path.iterdir(), key=lambda p: (p.is_file(), p.name))
        visible = [e for e in entries if not e.name.startswith(".")]
        structure = [("📁 " if e.is_dir() else "📄 ") + e.name for e in visible[:30]]
        lines.append("\nTop-level structure:")
        lines.extend(f"  {s}" for s in structure)
    except Exception:  # noqa: BLE001
        pass

    # Tech stack detection
    indicators = {
        "pyproject.toml":   "Python (pyproject)",
        "setup.py":         "Python (setup.py)",
        "requirements.txt": "Python (requirements)",
        "package.json":     "Node.js / JavaScript",
        "Cargo.toml":       "Rust",
        "go.mod":           "Go",
        "pom.xml":          "Java (Maven)",
        "build.gradle":     "Java/Kotlin (Gradle)",
        "Gemfile":          "Ruby",
        "composer.json":    "PHP",
    }
    detected = [label for file, label in indicators.items() if (repo_path / file).exists()]
    if detected:
        lines.append(f"\nDetected stack: {', '.join(detected)}")

    # README preview (first 60 lines)
    for readme_name in ("README.md", "README.rst", "README.txt", "README"):
        readme = repo_path / readme_name
        if readme.exists():
            try:
                content = readme.read_text(errors="replace")
                preview = content.splitlines()[:60]
                lines.append(f"\n{readme_name} (first 60 lines):\n")
                lines.extend(preview)
            except Exception:  # noqa: BLE001
                pass
            break

    return "\n".join(lines)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _create_session(
    db: Any,
    session_id: str,
    repo_path: Path,
    model: str,
    user_name: str,
) -> None:
    db.execute(
        """
        INSERT INTO sessions (session_id, repo_path, started_at, model, permission_mode, user_name)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, str(repo_path), int(time.time() * 1000), model, "edit", user_name),
    )
    db.commit()


def _end_session(db: Any, session_id: str) -> None:
    db.execute(
        "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
        (int(time.time() * 1000), session_id),
    )
    db.commit()


# ── Slash-command handlers ────────────────────────────────────────────────────

_COMMANDS: dict[str, str] = {
    "/show":     "Show this help listing all commands",
    "/thinking": "Toggle extended thinking on / off",
    "/model":    "List all models and switch the active one",
    "/mode":     "Switch permission mode (read / edit / full)",
    "/allow":    "Unlock a gitignored file for reading this session: /allow <path>",
    "/clear":    "Clear conversation history and start fresh in this session",
    "/delete":   "Delete this session entirely and start a new one",
    "/list":     "List all previous conversations and resume any one",
    "/history":  "Print a preview of all turns in the current conversation",
    "/export":   "Export the current conversation to a Markdown file",
    "/status":   "Show current session info (model, turns, repo, etc.)",
    "/exit":     "Exit ACA",
}


def _cmd_show(con: Console) -> None:
    table = Table(
        title="ACA Commands",
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Command", style="bright_cyan", no_wrap=True, min_width=12)
    table.add_column("Description")
    for cmd, desc in _COMMANDS.items():
        table.add_row(cmd, desc)
    con.print()
    con.print(table)
    con.print()


def _cmd_thinking(con: Console, state: dict) -> None:
    state["thinking"] = not state["thinking"]
    status = "[green]ON[/green]" if state["thinking"] else "[red]OFF[/red]"
    con.print(f"\n  Thinking: {status}\n")


def _cmd_allow(
    con: Console,
    raw_input: str,
    registry: ToolRegistry,
    repo_path: Path,
) -> None:
    """
    Add a specific file to the per-session read allowlist.
    Usage: /allow <path>   (relative to repo root or absolute)
    """
    parts = raw_input.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        con.print("\n  Usage: [bright_cyan]/allow <path>[/]  (relative or absolute)\n")
        return
    path_arg = parts[1].strip()
    resolved = registry.add_to_read_allowlist(path_arg, repo_root=str(repo_path))
    con.print(f"\n  [green]Allowed for this session:[/] {resolved}\n")


def _cmd_model(con: Console, state: dict) -> None:
    models = _all_models()
    table = Table(
        title="Available Models",
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("Provider / Name", style="bright_cyan")
    table.add_column("Model ID", style="dim")
    for i, (name, mid) in enumerate(models, 1):
        marker = "  ◀ active" if mid == state["model"] else ""
        table.add_row(str(i), name + marker, mid)
    con.print()
    con.print(table)
    raw = Prompt.ask("  Select number (Enter to cancel)", console=con, default="").strip()
    if not raw:
        con.print()
        return
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            state["model"] = models[idx][1]
            con.print(f"\n  Model → [bright_cyan]{state['model']}[/]\n")
        else:
            con.print("\n  [red]Invalid selection.[/]\n")
    except ValueError:
        con.print("\n  [red]Invalid input.[/]\n")


def _cmd_list(
    con: Console,
    db: Any,
    james: JamesAgent,
    current_session_id: str,
) -> None:
    """
    Show all previous sessions. Let the user pick one to resume by loading
    its turns into the current James instance.
    """
    rows = db.execute(
        """
        SELECT s.session_id, s.started_at, s.model,
               COUNT(t.turn_id) AS turn_count,
               MIN(CASE WHEN t.turn_index = 0 THEN t.user_message END) AS first_message
        FROM sessions s
        LEFT JOIN turns t ON t.session_id = s.session_id
        WHERE s.session_id != ?
        GROUP BY s.session_id
        ORDER BY s.started_at DESC
        LIMIT 30
        """,
        (current_session_id,),
    ).fetchall()

    if not rows:
        con.print("\n  [dim]No previous conversations found.[/]\n")
        return

    table = Table(
        title="Previous Conversations",
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#",             style="dim",         width=4,  no_wrap=True)
    table.add_column("Date",          style="bright_cyan",           no_wrap=True)
    table.add_column("Model",         style="dim")
    table.add_column("Turns",         style="dim",         width=6,  justify="right")
    table.add_column("First message", no_wrap=False)

    for i, (sid, started_at_ms, model, turn_count, first_msg) in enumerate(rows, 1):
        dt = datetime.fromtimestamp(started_at_ms / 1000).strftime("%Y-%m-%d %H:%M")
        model_short = model.split("/")[-1] if "/" in model else model
        preview = (first_msg[:70] + "…") if first_msg and len(first_msg) > 70 else (first_msg or "—")
        table.add_row(str(i), dt, model_short, str(turn_count or 0), preview)

    con.print()
    con.print(table)
    raw = Prompt.ask(
        "  Select number to resume (Enter to cancel)", console=con, default=""
    ).strip()
    if not raw:
        con.print()
        return
    try:
        idx = int(raw) - 1
        if not (0 <= idx < len(rows)):
            con.print("\n  [red]Invalid selection.[/]\n")
            return
    except ValueError:
        con.print("\n  [red]Invalid input.[/]\n")
        return

    selected_session_id = rows[idx][0]
    turns = db.execute(
        """
        SELECT turn_id, user_message, final_response
        FROM turns
        WHERE session_id = ?
        ORDER BY turn_index ASC
        """,
        (selected_session_id,),
    ).fetchall()

    james._history.clear()
    james._history_meta.clear()
    for turn_id, user_message, final_response in turns:
        james._history.append({"role": "user",      "content": user_message})
        james._history.append({"role": "assistant", "content": final_response or ""})
        james._history_meta.append({"turn_id": turn_id})

    loaded = len(turns)
    con.print(
        f"\n  [green]Loaded {loaded} turn{'s' if loaded != 1 else ''} from past conversation.[/]"
        "  Continue where you left off.\n"
    )


def _cmd_mode(con: Console, james: JamesAgent) -> None:
    modes = [PermissionMode.READ, PermissionMode.EDIT, PermissionMode.FULL]
    current = james._permission_mode
    con.print(f"\n  Current mode: [bright_cyan]{current.value.upper()}[/]")
    for i, m in enumerate(modes, 1):
        marker = "  ◀ active" if m == current else ""
        con.print(f"  {i}. {m.value.upper()}{marker}")
    raw = Prompt.ask("  Select number (Enter to cancel)", console=con, default="").strip()
    if not raw:
        con.print()
        return
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(modes):
            selected = modes[idx]
            james._permission_mode = selected
            james._full_permission_mode = selected
            con.print(f"\n  Mode → [bright_cyan]{selected.value.upper()}[/]\n")
        else:
            con.print("\n  [red]Invalid selection.[/]\n")
    except (ValueError, IndexError):
        con.print("\n  [red]Invalid input.[/]\n")


def _cmd_clear(con: Console, james: JamesAgent) -> None:
    james._history.clear()
    james._history_meta.clear()
    con.print("\n  [green]Conversation cleared.[/] Context window reset.\n")


def _cmd_status(
    con: Console,
    state: dict,
    james: JamesAgent,
    session_id: str,
    repo_path: Path,
    turn_count: int,
) -> None:
    table = Table(
        title="Session Status",
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Field", style="bright_cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Session ID",      session_id)
    table.add_row("Repo",            str(repo_path))
    table.add_row("Model",           state["model"])
    table.add_row("Permission mode", james._permission_mode.value.upper())
    table.add_row("Thinking",        "ON" if state["thinking"] else "OFF")
    table.add_row("Turns this session", str(turn_count))
    table.add_row("History pairs",   str(len(james._history_meta)))
    con.print()
    con.print(table)
    con.print()


def _cmd_history(con: Console, james: JamesAgent) -> None:
    if not james._history_meta:
        con.print("\n  [dim]No history yet.[/]\n")
        return
    con.print()
    for i, meta in enumerate(james._history_meta):
        turn_id_short = meta["turn_id"][:8]
        user_raw = james._history[i * 2].get("content", "")
        asst_raw = james._history[i * 2 + 1].get("content", "")
        user_preview = (user_raw[:100] + "…") if len(user_raw) > 100 else user_raw
        asst_preview = (asst_raw[:100] + "…") if len(asst_raw) > 100 else asst_raw
        con.print(f"  [dim]── Turn {i + 1}  (id: {turn_id_short}…)[/]")
        con.print(f"  [bright_cyan]You:[/] {user_preview}")
        con.print(f"  [white]ACA:[/] {asst_preview}")
        con.print()


def _cmd_export(
    con: Console,
    james: JamesAgent,
    session_id: str,
    repo_path: Path,
) -> None:
    if not james._history_meta:
        con.print("\n  [dim]Nothing to export yet.[/]\n")
        return
    export_dir = repo_path / ".aca" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"session-{session_id[:8]}.md"
    export_path = export_dir / filename
    lines = [
        f"# ACA Session Export\n",
        f"> Session ID: `{session_id}`  \n> Repo: `{repo_path}`\n\n---\n",
    ]
    for i, meta in enumerate(james._history_meta):
        user_msg = james._history[i * 2].get("content", "")
        asst_msg = james._history[i * 2 + 1].get("content", "")
        lines.append(f"## Turn {i + 1}\n\n**You:** {user_msg}\n\n**ACA:**\n\n{asst_msg}\n\n---\n")
    export_path.write_text("\n".join(lines))
    con.print(f"\n  [green]Exported →[/] {export_path}\n")


# ── Response renderer ─────────────────────────────────────────────────────────

def _render_response(con: Console, response: str, user_name: str) -> None:
    """Render James's final response as formatted Markdown."""
    del user_name
    con.print()
    try:
        con.print(Markdown(response))
    except Exception:  # noqa: BLE001
        con.print(Text(response))
    con.print()


# ── Build James for a new session ─────────────────────────────────────────────

def _build_james(
    session_id: str,
    repo_path: Path,
    model: str,
    thinking: bool,
    db: Any,
    agent_console: AgentConsole,
    repo_context: str,
    user_name: str,
) -> JamesAgent:
    registry = build_registry()
    full_context = f"User's name: {user_name}\n\n{repo_context}" if user_name else repo_context
    # Infer provider from model string — OpenAI models don't contain "/"
    provider = (
        ProviderName.OPENAI
        if not "/" in model and any(
            model == getattr(OpenAIModels, a)
            for a in vars(OpenAIModels)
            if not a.startswith("_")
        )
        else ProviderName.OPENROUTER
    )
    return JamesAgent(
        registry=registry,
        permission_mode=PermissionMode.EDIT,
        model=model,
        thinking=thinking,
        stream=True,
        db=db,
        session_id=session_id,
        console=agent_console,
        repo_context=full_context,
        provider=provider,
    )


# ── One full REPL session ─────────────────────────────────────────────────────

def _run_session(
    con: Console,
    user_name: str,
    state: dict,
    repo_path: Path,
    repo_context: str,
    db: Any,
) -> bool:
    """
    Run one REPL session. Returns True if the caller should start a new session
    (/delete), False if ACA should exit entirely.
    """
    session_id = str(uuid.uuid4())
    _create_session(db, session_id, repo_path, state["model"], user_name)

    agent_console = AgentConsole(console=con, verbosity="quiet")  # share same Console — enables Status/Live
    agent_console._show_content_panel = False  # _render_response is the single display in CLI
    james = _build_james(
        session_id=session_id,
        repo_path=repo_path,
        model=state["model"],
        thinking=state["thinking"],
        db=db,
        agent_console=agent_console,
        repo_context=repo_context,
        user_name=user_name,
    )

    con.print(Rule(style=_DIVIDER_STYLE))
    con.print(f"\n  [bright_cyan]Session started.[/]  Type [dim]/show[/] for commands.\n")

    turn_count = 0

    while True:
        # ── Prompt ────────────────────────────────────────────────────────────
        try:
            raw = Prompt.ask(
                f"\n  [bright_cyan]{user_name}[/]",
                console=con,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            con.print("\n\n  [dim]Ctrl-C — exiting.[/]\n")
            _end_session(db, session_id)
            return False

        if not raw:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        cmd = raw.lower().split()[0] if raw.startswith("/") else None

        if cmd == "/show":
            _cmd_show(con)
            continue

        if cmd == "/thinking":
            _cmd_thinking(con, state)
            # Rebuild james with new thinking flag
            james._thinking = state["thinking"]
            continue

        if cmd == "/model":
            _cmd_model(con, state)
            # Rebuild james so model change takes effect immediately
            _end_session(db, session_id)
            session_id = str(uuid.uuid4())
            _create_session(db, session_id, repo_path, state["model"], user_name)
            james = _build_james(
                session_id=session_id,
                repo_path=repo_path,
                model=state["model"],
                thinking=state["thinking"],
                db=db,
                agent_console=agent_console,
                repo_context=repo_context,
                user_name=user_name,
            )
            turn_count = 0
            con.print(f"  [dim]New session started with [bright_cyan]{state['model']}[/].[/]\n")
            continue

        if cmd == "/mode":
            _cmd_mode(con, james)
            continue

        if cmd == "/allow":
            _cmd_allow(con, raw, james._registry, repo_path)
            continue

        if cmd == "/clear":
            _cmd_clear(con, james)
            turn_count = 0
            continue

        if cmd == "/delete":
            _end_session(db, session_id)
            con.print("\n  [yellow]Session deleted.[/] Starting new session...\n")
            return True  # restart

        if cmd == "/history":
            _cmd_history(con, james)
            continue

        if cmd == "/list":
            _cmd_list(con, db, james, session_id)
            continue

        if cmd == "/export":
            _cmd_export(con, james, session_id, repo_path)
            continue

        if cmd == "/status":
            _cmd_status(con, state, james, session_id, repo_path, turn_count)
            continue

        if cmd in ("/exit", "/quit", "/q"):
            _end_session(db, session_id)
            con.print("\n  [dim]Disconnecting. Stay dangerous.[/]\n")
            return False

        if cmd and cmd not in _COMMANDS:
            con.print(f"\n  [red]Unknown command:[/] {cmd}  — type [bright_cyan]/show[/] for help.\n")
            continue

        # ── Agent turn ────────────────────────────────────────────────────────
        try:
            agent_console.begin_user_turn()
            response, _ = james.run_turn(raw)
        except KeyboardInterrupt:
            con.print("\n  [yellow]Interrupted.[/]\n")
            continue
        except Exception as exc:  # noqa: BLE001
            con.print(f"\n  [red]Error:[/] {exc}\n")
            continue

        turn_count += 1
        if agent_console.consume_streamed_response_flag():
            con.print()
        else:
            _render_response(con, response, user_name)

    # unreachable, but satisfies type checkers
    return False  # pragma: no cover


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Pre-flight: must be inside a venv ─────────────────────────────────────
    _check_venv()

    # Rich console for all user-facing output (stdout)
    con = Console(highlight=False, markup=True)

    # ── Bootstrap packages ────────────────────────────────────────────────────
    _bootstrap_packages(con)

    # ── Global ~/.aca/ setup ──────────────────────────────────────────────────
    _setup_global_aca_dir()

    # ── Banner ────────────────────────────────────────────────────────────────
    con.clear()
    con.print(_BANNER)

    # ── Repo detection ────────────────────────────────────────────────────────
    repo_path = _detect_repo(Path(os.getcwd()))
    con.print(f"  [dim]Repo detected:[/] [bright_cyan]{repo_path}[/]\n")

    # ── Load .env (repo root first, then cwd) ────────────────────────────────
    load_dotenv(repo_path / ".env", override=False)
    load_dotenv(override=False)  # also try cwd walk-up as fallback

    # ── Local .aca/ folder setup ──────────────────────────────────────────────
    _setup_local_aca_dir(repo_path)

    # ── DB ────────────────────────────────────────────────────────────────────
    db_path = repo_path / "data" / "aca.db"
    db = open_db(db_path)

    # ── User name (persisted in ~/.aca/config.json) ───────────────────────────
    cfg = _load_config()
    user_name: str = cfg.get("user_name", "").strip()

    if not user_name:
        con.print(
            Panel(
                "[bright_cyan]First time here?[/]  What should I call you?",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        user_name = Prompt.ask("  Your name", console=con).strip()
        if not user_name:
            user_name = "User"
        cfg["user_name"] = user_name
        _save_config(cfg)
        con.print(f"\n  [green]Got it, {user_name}.[/] Welcome to ACA.\n")
    else:
        con.print(f"  [bright_cyan]Welcome back, {user_name}.[/]\n")

    # ── Bootstrap scan ────────────────────────────────────────────────────────
    with Status("  [dim cyan]Scanning repo…[/]", console=con, spinner="dots"):
        repo_context = _bootstrap_scan(repo_path)

    con.print(f"  [dim]Bootstrap scan complete.[/]\n")

    # ── Session state (mutable across /model etc.) ────────────────────────────
    state: dict = {
        "model":    DEFAULT_MODEL,
        "thinking": False,
    }

    # ── REPL ──────────────────────────────────────────────────────────────────
    restart = True
    while restart:
        restart = _run_session(
            con=con,
            user_name=user_name,
            state=state,
            repo_path=repo_path,
            repo_context=repo_context,
            db=db,
        )

    db.close()


if __name__ == "__main__":
    main()
