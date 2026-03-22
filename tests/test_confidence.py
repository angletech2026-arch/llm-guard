from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse as FastJSONResponse
from httpx import ASGITransport, AsyncClient

from llm_guard.analyzers.confidence import ConfidenceAnalyzer
from llm_guard.config import ConfidenceConfig


@pytest.fixture
def analyzer():
    return ConfidenceAnalyzer(ConfidenceConfig())


@pytest.fixture
def high_confidence_logprobs():
    """Logprobs where all tokens have high confidence."""
    return [
        {"token": "Hello", "logprob": -0.01},
        {"token": "!", "logprob": -0.05},
        {"token": " How", "logprob": -0.1},
        {"token": " can", "logprob": -0.02},
        {"token": " I", "logprob": -0.01},
        {"token": " help", "logprob": -0.03},
        {"token": "?", "logprob": -0.01},
    ]


@pytest.fixture
def mixed_confidence_logprobs():
    """Logprobs with some low-confidence tokens."""
    return [
        {"token": "The", "logprob": -0.1},
        {"token": " capital", "logprob": -0.2},
        {"token": " is", "logprob": -0.1},
        {"token": " Xyz", "logprob": -5.0},
        {"token": "burg", "logprob": -4.5},
        {"token": "istan", "logprob": -6.0},
        {"token": ".", "logprob": -0.3},
    ]


@pytest.mark.asyncio
async def test_high_confidence(analyzer, high_confidence_logprobs):
    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "Hi"}],
        response_text="Hello! How can I help?",
        logprobs=high_confidence_logprobs,
    )
    assert result is not None
    assert result.analyzer_name == "confidence"
    assert result.data["overall"] == "HIGH"
    assert result.data["overall_score"] > 0.8


@pytest.mark.asyncio
async def test_mixed_confidence(analyzer, mixed_confidence_logprobs):
    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "What is the capital?"}],
        response_text="The capital is Xyzburgistan.",
        logprobs=mixed_confidence_logprobs,
    )
    assert result is not None
    assert len(result.data["low_confidence_segments"]) > 0
    seg = result.data["low_confidence_segments"][0]
    assert "Xyz" in seg["text"]
    assert seg["level"] == "LOW"


@pytest.mark.asyncio
async def test_no_logprobs(analyzer):
    result = await analyzer.analyze(
        request_messages=[],
        response_text="Hello",
        logprobs=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_empty_logprobs(analyzer):
    result = await analyzer.analyze(
        request_messages=[],
        response_text="Hello",
        logprobs=[],
    )
    assert result is None


@pytest.mark.asyncio
async def test_disabled():
    config = ConfidenceConfig(enabled=False)
    analyzer = ConfidenceAnalyzer(config)
    result = await analyzer.analyze(
        request_messages=[],
        response_text="Hello",
        logprobs=[{"token": "Hello", "logprob": -0.1}],
    )
    assert result is None


@pytest.mark.asyncio
async def test_logprobs_method_field(analyzer, high_confidence_logprobs):
    """Logprobs-based analysis includes method='logprobs'."""
    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "Hi"}],
        response_text="Hello! How can I help?",
        logprobs=high_confidence_logprobs,
    )
    assert result is not None
    assert result.data["method"] == "logprobs"


# --- Fallback tests ---


def _make_chat_mock(content: str = "Paris is the capital of France.") -> FastAPI:
    mock = FastAPI()

    @mock.post("/v1/chat/completions")
    async def completions(request: Request) -> FastJSONResponse:
        return FastJSONResponse(content={
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        })

    return mock


@pytest.mark.asyncio
async def test_fallback_disabled_returns_none():
    """Fallback disabled + no logprobs -> None."""
    config = ConfidenceConfig(fallback_enabled=False)
    analyzer = ConfidenceAnalyzer(config)
    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "Hi"}],
        response_text="Hello there!",
        logprobs=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_fallback_high_consistency():
    """All samples return same text -> HIGH confidence."""
    mock = _make_chat_mock("Paris is the capital of France.")
    transport = ASGITransport(app=mock)

    config = ConfidenceConfig(fallback_enabled=True, fallback_samples=3)
    analyzer = ConfidenceAnalyzer(config=config, upstream_base_url="http://mock")
    analyzer._client = AsyncClient(transport=transport, base_url="http://mock")

    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_text="Paris is the capital of France.",
        logprobs=None,
    )
    assert result is not None
    assert result.data["method"] == "multi_sample_fallback"
    assert result.data["overall"] == "HIGH"
    assert result.data["overall_score"] == 1.0
    await analyzer.close()


@pytest.mark.asyncio
async def test_fallback_not_used_when_logprobs_present():
    """Fallback not called when logprobs are provided."""
    config = ConfidenceConfig(fallback_enabled=True)
    analyzer = ConfidenceAnalyzer(config)
    result = await analyzer.analyze(
        request_messages=[{"role": "user", "content": "Hi"}],
        response_text="Hello",
        logprobs=[{"token": "Hello", "logprob": -0.1}],
    )
    assert result is not None
    assert result.data["method"] == "logprobs"
