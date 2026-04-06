from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..types import ProviderEvent, ProviderRequest, RunResult


class LLMProvider(ABC):
    provider_name: str

    @abstractmethod
    def stream_turn(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        """Yield normalized provider events and end with `response.completed`."""

    def run_turn(self, request: ProviderRequest) -> RunResult:
        final_result: RunResult | None = None

        for event in self.stream_turn(request):
            if event.type == "response.completed" and event.result is not None:
                final_result = event.result

        if final_result is None:
            raise RuntimeError("Provider stream ended without a final result.")

        return final_result

