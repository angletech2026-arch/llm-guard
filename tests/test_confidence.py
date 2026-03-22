from __future__ import annotations

import pytest

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
