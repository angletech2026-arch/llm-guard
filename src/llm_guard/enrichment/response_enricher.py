from __future__ import annotations

import json
import logging
from typing import Any

from llm_guard.analyzers.base import AnalyzerResult
from llm_guard.config import GuardConfig

logger = logging.getLogger("llm_guard")


def enrich_response(
    response_data: dict[str, Any],
    results: list[AnalyzerResult],
    config: GuardConfig,
    logprobs_injected: bool = False,
) -> dict[str, Any]:
    """Attach analyzer results to the response."""
    guard_data: dict[str, Any] = {}

    for result in results:
        guard_data[result.analyzer_name] = result.data

    mode = config.output.mode

    if mode in ("metadata", "both"):
        response_data["llm_guard"] = guard_data

    # Strip injected logprobs from response if user didn't ask for them
    if logprobs_injected:
        for choice in response_data.get("choices", []):
            if "logprobs" in choice:
                del choice["logprobs"]

    return response_data


def build_analysis_sse_chunk(results: list[AnalyzerResult]) -> str:
    """Build an SSE chunk containing analysis results for streaming mode."""
    guard_data: dict[str, Any] = {}
    for result in results:
        guard_data[result.analyzer_name] = result.data

    chunk = {"object": "llm_guard.analysis", **guard_data}
    return f"data: {json.dumps(chunk)}\n\n"
