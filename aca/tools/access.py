from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


PathAccessDecision = Literal["deny", "allow_once", "allow_session"]


@dataclass(slots=True, frozen=True)
class PathAccessPromptRequest:
    relative_path: str
    absolute_path: str
    sensitive: bool
    explicit_in_user_turn: bool


PromptCallback = Callable[[PathAccessPromptRequest], PathAccessDecision]


def _normalize_relative(path: str | Path) -> str:
    return Path(path).as_posix().strip()


@dataclass(slots=True)
class PathAccessManager:
    workspace_root: Path
    prompt: PromptCallback | None = None
    _current_user_input: str = ""
    _turn_allowlist: set[str] = field(default_factory=set)
    _session_allowlist: set[str] = field(default_factory=set)

    def begin_turn(self, user_input: str) -> None:
        self._current_user_input = user_input
        self._turn_allowlist.clear()

    def can_read(self, relative_path: str | Path, *, sensitive: bool) -> bool:
        normalized = _normalize_relative(relative_path)
        if normalized in self._turn_allowlist or normalized in self._session_allowlist:
            return True
        if not self.was_explicitly_requested(normalized):
            return False
        if self.prompt is None:
            return False
        absolute = str((self.workspace_root / normalized).resolve())
        decision = self.prompt(
            PathAccessPromptRequest(
                relative_path=normalized,
                absolute_path=absolute,
                sensitive=sensitive,
                explicit_in_user_turn=True,
            )
        )
        if decision == "allow_once":
            self._turn_allowlist.add(normalized)
            return True
        if decision == "allow_session":
            self._session_allowlist.add(normalized)
            return True
        return False

    def was_explicitly_requested(self, relative_path: str | Path) -> bool:
        normalized = _normalize_relative(relative_path)
        absolute = str((self.workspace_root / normalized).resolve())
        return _contains_exact_path(self._current_user_input, normalized) or _contains_exact_path(self._current_user_input, absolute)


def _contains_exact_path(text: str, path: str) -> bool:
    if not text or not path:
        return False
    candidates = {
        path,
        f"`{path}`",
        f'"{path}"',
        f"'{path}'",
        f"({path})",
        f"[{path}]",
    }
    return any(candidate in text for candidate in candidates)
