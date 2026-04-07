from __future__ import annotations

import re
from enum import Enum


class CommandRisk(str, Enum):
    READ_ONLY = "read_only"
    DEV_SAFE = "dev_safe"
    HIGH_RISK = "high_risk"


HIGH_RISK_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+checkout\s+--\b",
    r"\bchmod\s+-R\b",
    r"\bchown\s+-R\b",
    r">\s*/dev/",
    r"\bmkfs\b",
)

READ_ONLY_PREFIXES = (
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "sed ",
    "grep ",
    "rg ",
    "find ",
    "git status",
    "git diff",
    "git show",
    "git log",
    "which ",
    "echo ",
)

DEV_SAFE_PREFIXES = (
    "python ",
    ".venv/bin/python ",
    "pytest",
    ".venv/bin/pytest",
    "npm test",
    "npm run ",
    "pnpm ",
    "yarn ",
    "make ",
    "ruff ",
    "mypy ",
)


def classify_command(command: str) -> CommandRisk:
    cleaned = command.strip()
    lowered = cleaned.lower()
    if any(re.search(pattern, lowered) for pattern in HIGH_RISK_PATTERNS):
        return CommandRisk.HIGH_RISK
    if any(lowered.startswith(prefix) for prefix in READ_ONLY_PREFIXES) and not _looks_mutating(lowered):
        return CommandRisk.READ_ONLY
    if any(lowered.startswith(prefix) for prefix in DEV_SAFE_PREFIXES):
        return CommandRisk.DEV_SAFE
    return CommandRisk.HIGH_RISK


def _looks_mutating(command: str) -> bool:
    return any(token in command for token in (" -i", ">>", ">", "| tee", "touch ", "mkdir ", "mv ", "cp ", "git add", "git commit"))
