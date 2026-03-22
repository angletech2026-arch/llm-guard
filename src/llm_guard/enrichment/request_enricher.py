from __future__ import annotations

import logging
from typing import Any

from llm_guard.config import GuardConfig

logger = logging.getLogger("llm_guard")


def enrich_request(body: dict[str, Any], config: GuardConfig) -> tuple[dict[str, Any], bool]:
    """Inject logprobs into chat completion request if not already set.

    Returns:
        Tuple of (modified body, whether we injected logprobs).
    """
    if not config.analyzers.confidence.enabled:
        return body, False

    injected = False
    if "logprobs" not in body or body["logprobs"] is None:
        body["logprobs"] = True
        body["top_logprobs"] = 5
        injected = True
        logger.debug("Injected logprobs=true into request")

    return body, injected
