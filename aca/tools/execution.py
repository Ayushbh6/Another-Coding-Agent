"""
Execution tools for ACA.

run_command  — allowlist-gated shell command execution (FULL mode only).
run_tests    — runs pytest within the repo (FULL mode only).

Both tools are disabled in READ and EDIT modes (they won't even appear in the
LLM's tool list via the registry's schema filtering).

Safety principles per TOOLS_AND_PERMISSIONS.md §7:
  - hard-blocked command patterns (rm -rf, sudo, chmod on sensitive paths, etc.)
  - repo-root boundary enforced for all working directories
  - execution timeout enforced to prevent hung processes
  - all output captured and returned; never executed silently

Register into a ToolRegistry via:
    from aca.tools import execution
    execution.register(registry)
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from aca.tools.registry import ToolCategory, ToolDefinition, ToolRegistry
from aca.tools.read import _resolve_and_guard


# ── Safety constants ──────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_BYTES = 256_000   # 256 KB cap on stdout/stderr returned to the model

_SAFE_EXECUTABLES = {
    "python", "python3", "pytest",
    "rg", "grep", "sed", "awk", "find", "ls", "pwd", "cat",
    "head", "tail", "wc", "sort", "uniq", "cut", "tr", "tee",
    "echo", "printf", "true", "false", "test", "[",
    "xargs", "env", "which", "dirname", "basename", "realpath",
    "diff", "touch", "mkdir",
    "git",
    "make", "cmake",
    "node", "npm", "npx", "pnpm", "yarn",
    "pip", "pip3", "uv", "go", "cargo", "swift", "xcodebuild",
}
_SAFE_GIT_SUBCOMMANDS = {
    # read-only
    "status", "diff", "show", "log", "ls-files", "grep",
    "rev-parse", "branch", "blame", "remote", "symbolic-ref", "describe",
    # write (safe, local-only)
    "add", "commit", "checkout", "switch", "stash", "reset",
    "cherry-pick", "rebase", "merge", "tag",
}
_BLOCKED_GIT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+\S*\s*--force\b", re.IGNORECASE),
    re.compile(r"\bgit\s+\S*\s*-f\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
]

# Patterns that are unconditionally blocked regardless of mode
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-[^\s]*r", re.IGNORECASE),     # rm -r, rm -rf, rm -Rf, etc.
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bchmod\b.*?(\.ssh|\.aws|id_rsa|authorized)", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|.*\bsh\b", re.IGNORECASE),  # curl | sh pattern
    re.compile(r"\bwget\b.*\|.*\bsh\b", re.IGNORECASE),
    re.compile(r"\beval\b", re.IGNORECASE),
    re.compile(r"\bkill\b.*-9", re.IGNORECASE),          # SIGKILL
    re.compile(r">\s*/etc/", re.IGNORECASE),              # redirect into /etc/
    re.compile(r"\bdd\b.*if=", re.IGNORECASE),            # disk operations
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bshred\b", re.IGNORECASE),
    re.compile(r"\b:(){ :|:& };:", re.IGNORECASE),        # fork bomb
]


def _extract_executables(command: str) -> list[str]:
    """
    Return the leading executable name for each pipeline / chained segment.

    Splits on ``|``, ``&&``, ``||``, and ``;`` to find every command that
    would actually be exec'd by the shell, then resolves to the basename so
    e.g. ``/usr/bin/python3`` → ``python3``.
    """
    segments = re.split(r"\|\||&&|[|;]", command)
    executables: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Strip leading env-var assignments (FOO=bar cmd …)
        while "=" in seg.split()[0] if seg.split() else False:
            seg = seg.split(None, 1)[1] if len(seg.split(None, 1)) > 1 else ""
        try:
            argv = shlex.split(seg)
        except ValueError:
            argv = seg.split()
        if argv:
            executables.append(Path(argv[0]).name.lower())
    return executables


def _check_command_safety(command: str) -> None:
    """Raise ValueError if the command matches any blocked pattern."""
    # ── Unconditional destructive-pattern blocks ──────────────────────────
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(command):
            raise ValueError(
                f"Command blocked by safety policy. "
                f"Matched pattern: '{pattern.pattern}'. "
                "If you believe this is a false positive, request explicit user approval."
            )

    # ── Git-specific blocks (push, force, clean, reset --hard) ───────────
    for pattern in _BLOCKED_GIT_PATTERNS:
        if pattern.search(command):
            raise ValueError(
                f"Command blocked by safety policy. "
                f"Matched git pattern: '{pattern.pattern}'. "
                "Only local git operations are permitted."
            )

    # ── Executable allowlist — every segment must use a safe executable ──
    executables = _extract_executables(command)
    if not executables:
        raise ValueError("Command blocked by safety policy. Empty command.")

    for exe in executables:
        if exe not in _SAFE_EXECUTABLES:
            raise ValueError(
                f"Command blocked by safety policy. Executable '{exe}' is not on the safe allowlist."
            )

    # ── Git subcommand validation ────────────────────────────────────────
    # Parse each git invocation to verify its subcommand is allowed.
    segments = re.split(r"\|\||&&|[|;]", command)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            argv = shlex.split(seg)
        except ValueError:
            argv = seg.split()
        if not argv:
            continue
        if Path(argv[0]).name.lower() == "git":
            if len(argv) < 2 or argv[1] not in _SAFE_GIT_SUBCOMMANDS:
                raise ValueError(
                    "Command blocked by safety policy. Git subcommand not on the safe list. "
                    f"Allowed: {sorted(_SAFE_GIT_SUBCOMMANDS)}."
                )


def _resolve_working_dir(working_dir: str | None, repo_root: str) -> Path:
    """Resolve working_dir; default to repo_root; enforce repo boundary."""
    root = Path(repo_root).resolve()
    if working_dir is None:
        return root
    target = (root / working_dir).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"working_dir '{working_dir}' is outside the repo root. "
            "Commands must run within the repository."
        )
    return target


# ── Tool implementations ──────────────────────────────────────────────────────

def run_command(
    command: str,
    working_dir: str | None = None,
    repo_root: str = ".",
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Run a shell command within the repo. FULL permission mode only.

    The command is passed to /bin/sh -c. Pipes, chaining (&&, ||, ;), and
    redirection are allowed. Every executable in the pipeline must be on the
    safe allowlist. Destructive patterns (rm -rf, sudo, curl|sh, git push,
    etc.) are hard-blocked. Output is capped at 256 KB.

    Returns:
      {
        "command": str,
        "exit_code": int,
        "stdout": str,
        "stderr": str,
        "timed_out": bool,
        "truncated": bool,
      }
    """
    _check_command_safety(command)
    cwd = _resolve_working_dir(working_dir, repo_root)

    timed_out = False
    try:
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        exit_code = -1
        timed_out = True

    # ── Output size cap ──────────────────────────────────────────────────
    truncated = False
    if len(stdout) > _MAX_OUTPUT_BYTES:
        stdout = stdout[:_MAX_OUTPUT_BYTES] + "\n... [stdout truncated]"
        truncated = True
    if len(stderr) > _MAX_OUTPUT_BYTES:
        stderr = stderr[:_MAX_OUTPUT_BYTES] + "\n... [stderr truncated]"
        truncated = True

    return {
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "truncated": truncated,
    }


def run_tests(
    path: str = ".",
    extra_args: list[str] | None = None,
    repo_root: str = ".",
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Run pytest for the given path inside the repo. FULL permission mode only.

    `path` can be a directory or a specific test file.
    `extra_args` are appended to the pytest command (e.g. ["-v", "-k", "test_foo"]).

    Returns:
      {
        "path": str,
        "exit_code": int,         # 0 = all passed, 1 = failures, 2 = error, 5 = no tests
        "stdout": str,
        "stderr": str,
        "timed_out": bool,
      }
    """
    root = Path(repo_root).resolve()
    target = _resolve_and_guard(path, root)

    cmd = ["python", "-m", "pytest", str(target)]
    if extra_args:
        cmd.extend(extra_args)

    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        exit_code = -1
        timed_out = True

    return {
        "path": str(target.relative_to(root)),
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
    }


# ── Schema definitions ────────────────────────────────────────────────────────

_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command inside the repo. FULL permission mode only. "
                "Pipes, chaining (&&, ||, ;), and redirection are allowed. "
                "Every executable in the pipeline must be on the safe allowlist. "
                "Destructive patterns (rm -rf, sudo, curl|sh, git push, etc.) are hard-blocked. "
                "Output is capped at 256 KB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Passed to /bin/sh -c.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Relative path within repo to run the command from. Defaults to repo root.",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Timeout in seconds. Defaults to {_DEFAULT_TIMEOUT_SECONDS}.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run pytest for a path inside the repo. FULL permission mode only. "
                "Returns exit code (0=pass, 1=failures, 2=error, 5=no tests collected), "
                "stdout, and stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to a test file or directory. Defaults to repo root.",
                    },
                    "extra_args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra pytest arguments, e.g. [\"-v\", \"-k\", \"test_foo\"].",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Timeout in seconds. Defaults to {_DEFAULT_TIMEOUT_SECONDS}.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# ── Registration ──────────────────────────────────────────────────────────────

_FN_MAP = {
    "run_command": run_command,
    "run_tests": run_tests,
}


def register(registry: ToolRegistry) -> None:
    """Register all execution tools into the given ToolRegistry."""
    for schema in _SCHEMAS:
        name = schema["function"]["name"]
        registry.register(ToolDefinition(
            fn=_FN_MAP[name],
            schema=schema,
            category=ToolCategory.EXECUTION,
        ))
