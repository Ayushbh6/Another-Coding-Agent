from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ApprovalRequest:
    agent_id: str
    tool_name: str
    payload: dict[str, Any]
    preview: str


class ApprovalPolicy(Protocol):
    def request(self, request: ApprovalRequest) -> bool:
        ...


class AllowAllApprovalPolicy:
    def request(self, request: ApprovalRequest) -> bool:
        return True


class DenyAllApprovalPolicy:
    def request(self, request: ApprovalRequest) -> bool:
        return False
