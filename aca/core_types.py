from __future__ import annotations

from enum import Enum


class TurnIntent(str, Enum):
    CHAT = "chat"
    ANALYZE = "analyze"
    IMPLEMENT = "implement"
