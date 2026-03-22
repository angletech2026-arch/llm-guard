from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from llm_guard.analyzers.base import AnalyzerResult, BaseAnalyzer
from llm_guard.config import GuardConfig
from llm_guard.enrichment.response_enricher import build_analysis_sse_chunk

logger = logging.getLogger("llm_guard")


async def stream_and_analyze(
    upstream_response: httpx.Response,
    analyzers: list[BaseAnalyzer],
    config: GuardConfig,
    request_data: dict[str, Any] | None = None,
    logprobs_injected: bool = False,
) -> AsyncGenerator[bytes, None]:
    """Stream SSE chunks to client while buffering for analysis."""
    buffer_content: list[str] = []
    buffer_logprobs: list[dict[str, Any]] = []

    try:
        async for line in upstream_response.aiter_lines():
            stripped = line.strip()
            if not stripped:
                continue

            # Forward immediately
            yield f"{stripped}\n\n".encode("utf-8")

            # Parse and buffer
            if stripped.startswith("data: "):
                raw = stripped[6:].strip()

                if raw == "[DONE]":
                    # Stream complete - run analysis
                    if analyzers and buffer_content:
                        response_text = "".join(buffer_content)
                        lp = buffer_logprobs if buffer_logprobs else None
                        messages = (request_data or {}).get("messages", [])

                        results: list[AnalyzerResult] = []
                        for analyzer in analyzers:
                            try:
                                result = await analyzer.analyze(messages, response_text, lp)
                                if result is not None:
                                    results.append(result)
                            except Exception as e:
                                logger.error("Stream analyzer %s failed: %s", type(analyzer).__name__, e)

                        if results:
                            chunk = build_analysis_sse_chunk(results)
                            yield chunk.encode("utf-8")

                    return

                try:
                    parsed = json.loads(raw)
                    _extract_streaming_data(parsed, buffer_content, buffer_logprobs)
                except (json.JSONDecodeError, ValueError):
                    pass

    except Exception as e:
        logger.error("Streaming error: %s", e)
        error_chunk = f'data: {{"error": "{e}"}}\n\n'
        yield error_chunk.encode("utf-8")
    finally:
        await upstream_response.aclose()


def _extract_streaming_data(
    parsed: dict[str, Any],
    buffer_content: list[str],
    buffer_logprobs: list[dict[str, Any]],
) -> None:
    """Extract content and logprobs from a streaming chunk."""
    choices = parsed.get("choices", [])
    if not choices:
        return

    delta = choices[0].get("delta", {})
    content = delta.get("content")
    if content:
        buffer_content.append(content)

    # Extract streaming logprobs
    lp = choices[0].get("logprobs")
    if lp and "content" in lp and lp["content"]:
        buffer_logprobs.extend(lp["content"])
