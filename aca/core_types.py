from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TurnIntent(str, Enum):
    CHAT = "chat"
    ANALYZE = "analyze"
    IMPLEMENT = "implement"


@dataclass(slots=True)
class MasterClassification:
    intent: TurnIntent
    task_title: str
    task_description: str
    reasoning_summary: str

