from __future__ import annotations

import json
import logging
from typing import Any

from llm_guard.analyzers.base import AnalyzerResult
from llm_guard.config import GuardConfig

logger = logging.getLogger("llm_guard")


def build_analysis_headers(results: list[AnalyzerResult]) -> dict[str, str]:
    """Build HTTP headers from analyzer results."""
    headers: dict[str, str] = {}
    for result in results:
        if result.analyzer_name == "confidence":
            headers["X-LLM-Guard-Confidence"] = result.data.get("overall", "")
            score = result.data.get("overall_score")
            if score is not None:
                headers["X-LLM-Guard-Confidence-Score"] = str(score)
            method = result.data.get("method", "logprobs")
            headers["X-LLM-Guard-Confidence-Method"] = method
        elif result.analyzer_name == "conflicts":
            items = result.data.get("items", [])
            headers["X-LLM-Guard-Conflicts-Count"] = str(len(items))
            if items:
                headers["X-LLM-Guard-Conflicts"] = json.dumps(items)
        elif result.analyzer_name == "verification":
            passed = result.data.get("pass")
            if passed is not None:
                headers["X-LLM-Guard-Verification"] = str(passed).lower()
            summary = result.data.get("summary", "")
            if summary:
                headers["X-LLM-Guard-Verification-Summary"] = summary[:200]
    return headers


def enrich_response(
    response_data: dict[str, Any],
    results: list[AnalyzerResult],
    config: GuardConfig,
    logprobs_injected: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Attach analyzer results to the response.

    Returns (response_data, extra_headers).
    """
    guard_data: dict[str, Any] = {}
    for result in results:
        guard_data[result.analyzer_name] = result.data

    mode = config.output.mode
    extra_headers: dict[str, str] = {}

    if mode in ("metadata", "both"):
        response_data["llm_guard"] = guard_data

    if mode in ("header", "both"):
        extra_headers = build_analysis_headers(results)

    # Strip injected logprobs from response if user didn't ask for them
    if logprobs_injected:
        for choice in response_data.get("choices", []):
            if "logprobs" in choice:
                del choice["logprobs"]

    return response_data, extra_headers


def build_analysis_sse_chunk(results: list[AnalyzerResult]) -> str:
    """Build an SSE chunk containing analysis results for streaming mode."""
    guard_data: dict[str, Any] = {}
    for result in results:
        guard_data[result.analyzer_name] = result.data

    chunk = {"object": "llm_guard.analysis", **guard_data}
    return f"data: {json.dumps(chunk)}\n\n"
