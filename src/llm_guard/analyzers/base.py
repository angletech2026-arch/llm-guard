from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalyzerResult:
    analyzer_name: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    @abstractmethod
    async def analyze(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult | None: ...
