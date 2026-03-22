from __future__ import annotations

from llm_guard.config import GuardConfig
from llm_guard.enrichment.request_enricher import enrich_request


def test_inject_logprobs():
    config = GuardConfig()
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
    enriched, injected = enrich_request(body, config)
    assert enriched["logprobs"] is True
    assert enriched["top_logprobs"] == 5
    assert injected is True


def test_preserve_existing_logprobs():
    config = GuardConfig()
    body = {"model": "gpt-4o", "messages": [], "logprobs": True, "top_logprobs": 3}
    enriched, injected = enrich_request(body, config)
    assert enriched["logprobs"] is True
    assert enriched["top_logprobs"] == 3
    assert injected is False


def test_disabled_confidence():
    config = GuardConfig()
    config.analyzers.confidence.enabled = False
    body = {"model": "gpt-4o", "messages": []}
    enriched, injected = enrich_request(body, config)
    assert "logprobs" not in enriched
    assert injected is False
