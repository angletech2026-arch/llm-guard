from __future__ import annotations

from llm_guard.analyzers.base import AnalyzerResult
from llm_guard.config import GuardConfig
from llm_guard.enrichment.response_enricher import (
    build_analysis_headers,
    build_analysis_sse_chunk,
    enrich_response,
)


def test_enrich_header_mode():
    config = GuardConfig()  # default is "header"
    assert config.output.mode == "header"

    response = {"choices": [{"message": {"content": "Hi"}}]}
    results = [
        AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH", "overall_score": 0.95}),
    ]
    enriched, headers = enrich_response(response, results, config)
    # Body should NOT have llm_guard key
    assert "llm_guard" not in enriched
    # Headers should have confidence info
    assert headers["X-LLM-Guard-Confidence"] == "HIGH"
    assert headers["X-LLM-Guard-Confidence-Score"] == "0.95"


def test_enrich_metadata_mode():
    config = GuardConfig()
    config.output.mode = "metadata"

    response = {
        "id": "test",
        "choices": [{"message": {"content": "Hi"}, "logprobs": {"content": []}}],
    }
    results = [
        AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH", "overall_score": 0.95}),
    ]
    enriched, headers = enrich_response(response, results, config, logprobs_injected=True)
    assert "llm_guard" in enriched
    assert enriched["llm_guard"]["confidence"]["overall"] == "HIGH"
    assert "logprobs" not in enriched["choices"][0]
    assert headers == {}  # no headers in metadata mode


def test_enrich_both_mode():
    config = GuardConfig()
    config.output.mode = "both"

    response = {"choices": [{"message": {"content": "Hi"}}]}
    results = [
        AnalyzerResult(analyzer_name="confidence", data={"overall": "MEDIUM", "overall_score": 0.5}),
    ]
    enriched, headers = enrich_response(response, results, config)
    assert "llm_guard" in enriched
    assert headers["X-LLM-Guard-Confidence"] == "MEDIUM"


def test_enrich_keeps_logprobs_if_not_injected():
    config = GuardConfig()
    config.output.mode = "metadata"
    response = {
        "choices": [{"message": {"content": "Hi"}, "logprobs": {"content": []}}],
    }
    results = [AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH"})]
    enriched, _ = enrich_response(response, results, config, logprobs_injected=False)
    assert "logprobs" in enriched["choices"][0]


def test_header_confidence_values():
    results = [
        AnalyzerResult(analyzer_name="confidence", data={
            "overall": "LOW", "overall_score": 0.12, "method": "multi_sample_fallback"
        }),
    ]
    headers = build_analysis_headers(results)
    assert headers["X-LLM-Guard-Confidence"] == "LOW"
    assert headers["X-LLM-Guard-Confidence-Score"] == "0.12"
    assert headers["X-LLM-Guard-Confidence-Method"] == "multi_sample_fallback"


def test_header_conflicts_values():
    results = [
        AnalyzerResult(analyzer_name="conflicts", data={
            "items": [{"type": "language_conflict", "message": "en vs zh"}]
        }),
    ]
    headers = build_analysis_headers(results)
    assert headers["X-LLM-Guard-Conflicts-Count"] == "1"
    assert "en vs zh" in headers["X-LLM-Guard-Conflicts"]


def test_header_verification_values():
    results = [
        AnalyzerResult(analyzer_name="verification", data={
            "pass": False, "summary": "Error found"
        }),
    ]
    headers = build_analysis_headers(results)
    assert headers["X-LLM-Guard-Verification"] == "false"
    assert headers["X-LLM-Guard-Verification-Summary"] == "Error found"


def test_header_mode_body_untouched():
    config = GuardConfig()  # header mode
    original_body = {"choices": [{"message": {"content": "Hi"}}], "id": "test123"}
    results = [AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH"})]
    enriched, _ = enrich_response(original_body, results, config)
    assert "llm_guard" not in enriched
    assert enriched["id"] == "test123"


def test_build_analysis_sse_chunk():
    results = [AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH"})]
    chunk = build_analysis_sse_chunk(results)
    assert chunk.startswith("data: ")
    assert "llm_guard.analysis" in chunk
    assert "HIGH" in chunk
