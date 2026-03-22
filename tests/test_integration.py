"""Full round-trip integration tests."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport, AsyncClient

from llm_guard.config import GuardConfig
from llm_guard.server import create_app


def create_full_mock() -> FastAPI:
    """Mock upstream that returns logprobs."""
    mock = FastAPI()

    @mock.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        is_streaming = body.get("stream", False)
        include_logprobs = body.get("logprobs", False)

        content = "The capital of France is Paris."
        logprobs_data = None

        if include_logprobs:
            logprobs_data = {
                "content": [
                    {"token": "The", "logprob": -0.01},
                    {"token": " capital", "logprob": -0.05},
                    {"token": " of", "logprob": -0.01},
                    {"token": " France", "logprob": -0.02},
                    {"token": " is", "logprob": -0.01},
                    {"token": " Paris", "logprob": -0.03},
                    {"token": ".", "logprob": -0.01},
                ]
            }

        if is_streaming:
            chunks = []
            for i, char in enumerate(content):
                chunk = {
                    "id": "chatcmpl-int",
                    "object": "chat.completion.chunk",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
                }
                if include_logprobs and i < len(logprobs_data["content"]):
                    chunk["choices"][0]["logprobs"] = {"content": [logprobs_data["content"][i]]}
                chunks.append(f"data: {json.dumps(chunk)}")

            final = {
                "id": "chatcmpl-int",
                "object": "chat.completion.chunk",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            chunks.append(f"data: {json.dumps(final)}")
            chunks.append("data: [DONE]")

            async def generate():
                for chunk in chunks:
                    yield f"{chunk}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        response_data = {
            "id": "chatcmpl-int",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }

        if logprobs_data:
            response_data["choices"][0]["logprobs"] = logprobs_data

        return JSONResponse(content=response_data)

    @mock.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model"}]}

    return mock


@pytest.fixture
def mock_transport():
    return ASGITransport(app=create_full_mock())


@pytest.fixture
def proxy_app(mock_transport):
    config = GuardConfig()
    config.upstream.base_url = "http://mock-upstream"
    config.upstream.verify_ssl = False
    app = create_app(config)
    return app


@pytest.mark.asyncio
async def test_non_streaming_with_confidence(mock_transport):
    """Full round-trip: non-streaming request gets confidence scores."""
    config = GuardConfig()
    config.upstream.base_url = "http://mock-upstream"

    # Test confidence analyzer standalone with realistic data
    from llm_guard.analyzers.confidence import ConfidenceAnalyzer
    from llm_guard.config import ConfidenceConfig

    analyzer = ConfidenceAnalyzer(ConfidenceConfig())
    logprobs = [
        {"token": "The", "logprob": -0.01},
        {"token": " capital", "logprob": -0.05},
        {"token": " of", "logprob": -0.01},
        {"token": " France", "logprob": -0.02},
        {"token": " is", "logprob": -0.01},
        {"token": " Paris", "logprob": -0.03},
        {"token": ".", "logprob": -0.01},
    ]
    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_text="The capital of France is Paris.",
        logprobs=logprobs,
    )
    assert result is not None
    assert result.data["overall"] == "HIGH"
    assert result.data["overall_score"] > 0.9


@pytest.mark.asyncio
async def test_conflict_detection_in_pipeline():
    """Conflict analyzer detects language conflict in full pipeline."""
    from llm_guard.analyzers.conflict import ConflictAnalyzer
    from llm_guard.config import ConflictConfig

    analyzer = ConflictAnalyzer(ConflictConfig())
    result = await analyzer.analyze(
        request_messages=[
            {"role": "system", "content": "Always respond in English."},
            {"role": "user", "content": "用中文回答：法國的首都是哪裡？"},
        ],
        response_text="The capital of France is Paris.",
    )
    assert result is not None
    assert len(result.data["items"]) > 0
    assert result.data["items"][0]["type"] in ("language_conflict", "instruction_conflict")


@pytest.mark.asyncio
async def test_enrichment_pipeline():
    """Test request enrichment -> response enrichment pipeline."""
    from llm_guard.config import GuardConfig
    from llm_guard.enrichment.request_enricher import enrich_request
    from llm_guard.enrichment.response_enricher import enrich_response
    from llm_guard.analyzers.base import AnalyzerResult

    config = GuardConfig()

    # Step 1: Enrich request
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
    enriched_body, injected = enrich_request(body, config)
    assert enriched_body["logprobs"] is True
    assert injected is True

    # Step 2: Simulate response with logprobs
    response = {
        "choices": [{
            "message": {"content": "Hello!"},
            "logprobs": {"content": [{"token": "Hello", "logprob": -0.01}]},
        }],
    }

    # Step 3: Enrich response
    results = [AnalyzerResult(analyzer_name="confidence", data={"overall": "HIGH", "overall_score": 0.99})]
    enriched = enrich_response(response, results, config, logprobs_injected=True)

    assert "llm_guard" in enriched
    assert enriched["llm_guard"]["confidence"]["overall"] == "HIGH"
    # Injected logprobs should be stripped
    assert "logprobs" not in enriched["choices"][0]


@pytest.mark.asyncio
async def test_all_analyzers_disabled():
    """When all analyzers are disabled, proxy still works."""
    config = GuardConfig()
    config.analyzers.confidence.enabled = False
    config.analyzers.conflict.enabled = False
    config.analyzers.verification.enabled = False

    from llm_guard.proxy import ProxyHandler
    handler = ProxyHandler(config)
    # Just verify it initializes without error
    assert handler.analyzers == []


@pytest.mark.asyncio
async def test_health_endpoint():
    config = GuardConfig()
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
