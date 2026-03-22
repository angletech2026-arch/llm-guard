from __future__ import annotations

import math

from llm_guard.utils.logprobs_math import (
    aggregate_scores,
    classify,
    find_consecutive_low,
    token_confidence,
)


def test_token_confidence_zero():
    # logprob=0 means probability=1
    assert token_confidence(0.0) == 1.0


def test_token_confidence_negative():
    # logprob=-1 means ~0.368
    result = token_confidence(-1.0)
    assert abs(result - math.exp(-1.0)) < 1e-6


def test_token_confidence_very_negative():
    # logprob=-10 means ~0.0000454
    result = token_confidence(-10.0)
    assert result > 0.0
    assert result < 0.001


def test_aggregate_mean():
    assert abs(aggregate_scores([0.8, 0.6, 1.0], "mean") - 0.8) < 1e-9


def test_aggregate_min():
    assert aggregate_scores([0.8, 0.6, 1.0], "min") == 0.6


def test_aggregate_p10():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    result = aggregate_scores(scores, "p10")
    # p10 of sorted [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # idx = int(10 * 0.1) = 1 -> sorted[1] = 0.1
    assert result == 0.1


def test_aggregate_empty():
    assert aggregate_scores([], "mean") == 1.0


def test_classify_high():
    # score=0.5 > exp(-1.0)=0.368 -> HIGH
    assert classify(0.5, -3.0, -1.0) == "HIGH"


def test_classify_medium():
    # exp(-3.0)=0.0498, exp(-1.0)=0.368
    # score=0.1 -> between 0.0498 and 0.368 -> MEDIUM
    assert classify(0.1, -3.0, -1.0) == "MEDIUM"


def test_classify_low():
    # score=0.01 < exp(-3.0)=0.0498 -> LOW
    assert classify(0.01, -3.0, -1.0) == "LOW"


def test_find_consecutive_low_basic():
    tokens = [
        {"token": "The", "logprob": -0.5},
        {"token": " population", "logprob": -4.0},
        {"token": " of", "logprob": -5.0},
        {"token": " Mars", "logprob": -6.0},
        {"token": " is", "logprob": -0.1},
    ]
    segments = find_consecutive_low(tokens, threshold=-3.0, min_run=3)
    assert len(segments) == 1
    assert segments[0]["text"] == " population of Mars"
    assert segments[0]["start_idx"] == 1
    assert segments[0]["end_idx"] == 3


def test_find_consecutive_low_none():
    tokens = [
        {"token": "Hello", "logprob": -0.1},
        {"token": " world", "logprob": -0.2},
    ]
    segments = find_consecutive_low(tokens, threshold=-3.0, min_run=3)
    assert len(segments) == 0


def test_find_consecutive_low_too_short():
    tokens = [
        {"token": "a", "logprob": -5.0},
        {"token": "b", "logprob": -5.0},
    ]
    # min_run=3, only 2 consecutive -> no match
    segments = find_consecutive_low(tokens, threshold=-3.0, min_run=3)
    assert len(segments) == 0
