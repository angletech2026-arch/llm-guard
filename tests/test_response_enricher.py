from __future__ import annotations

from llm_guard.analyzers.base import AnalyzerResult
from llm_guard.config import GuardConfig
from llm_guard.enrichment.response_enricher import build_analysis_sse_chunk, enrich_response


def test_enrich_metadata_mode():
    config = GuardConfig()
    config.output.mode = "metadata"

    response = {
        "id": "test",
        "choices": [{"message": {"content": "Hi"}, "logprobs": {"content": []}}],
    }

    results = [
        AnalyzerResult(
            analyzer_name="confidence",
            data={"overall": "HIGH", "overall_score": 0.95},
        )
    ]

    enriched = enrich_response(response, results, config, logprobs_injected=True)
    assert "llm_guard" in enriched
    assert enriched["llm_guard"]["confidence"]["overall"] == "HIGH"
    # logprobs should be stripped since we injected them
    assert "logprobs" not in enriched["choices"][0]


def test_enrich_keeps_logprobs_if_not_injected():
    config = GuardConfig()
    response = {
        "choices": [{"message": {"content": "Hi"}, "logprobs": {"content": []}}],
    }
    results = [AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH"})]
    enriched = enrich_response(response, results, config, logprobs_injected=False)
    assert "logprobs" in enriched["choices"][0]


def test_build_analysis_sse_chunk():
    results = [AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH"})]
    chunk = build_analysis_sse_chunk(results)
    assert chunk.startswith("data: ")
    assert "llm_guard.analysis" in chunk
    assert "HIGH" in chunk
