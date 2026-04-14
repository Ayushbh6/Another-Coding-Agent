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

_SHELL_CONTROL_TOKENS = ("&&", "||", "|", ";", ">", "<", "`", "$(", "\n")
_SAFE_EXECUTABLES = {
    "python", "python3", "pytest",
    "rg", "grep", "sed", "awk", "find", "ls", "pwd", "cat",
    "git",
    "make", "cmake",
    "node", "npm", "npx", "pnpm", "yarn",
    "uv", "go", "cargo", "swift", "xcodebuild",
}
_SAFE_GIT_SUBCOMMANDS = {
    "status", "diff", "show", "log", "ls-files", "grep",
    "rev-parse", "branch", "blame", "remote", "symbolic-ref", "describe",
}

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


def _check_command_safety(command: str) -> None:
    """Raise ValueError if the command matches any blocked pattern."""
    if any(token in command for token in _SHELL_CONTROL_TOKENS):
        raise ValueError(
            "Command blocked by safety policy. Shell control operators, pipes, "
            "redirection, command substitution, and multi-command chains are not allowed. "
            "Run a single direct command instead."
        )

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Command blocked by safety policy. Could not parse command: {exc}") from exc

    if not argv:
        raise ValueError("Command blocked by safety policy. Empty command.")

    exe = Path(argv[0]).name.lower()
    if exe not in _SAFE_EXECUTABLES:
        raise ValueError(
            f"Command blocked by safety policy. Executable '{argv[0]}' is not on the safe allowlist."
        )

    if exe == "git":
        if len(argv) < 2 or argv[1] not in _SAFE_GIT_SUBCOMMANDS:
            raise ValueError(
                "Command blocked by safety policy. Only read-only git subcommands are allowed "
                f"via run_command: {sorted(_SAFE_GIT_SUBCOMMANDS)}."
            )

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(command):
            raise ValueError(
                f"Command blocked by safety policy. "
                f"Matched pattern: '{pattern.pattern}'. "
                "If you believe this is a false positive, request explicit user approval."
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

    The command is passed to /bin/sh -c. A hard-blocked pattern list prevents
    destructive commands. The command must run inside the repo boundary.

    Returns:
      {
        "command": str,
        "exit_code": int,
        "stdout": str,
        "stderr": str,
        "timed_out": bool,
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

    return {
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
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
                "Destructive commands (rm -rf, sudo, curl|sh, etc.) are hard-blocked. "
                "The command must run within the repository boundary. "
                "Use sparingly — prefer read and write tools for file operations."
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
