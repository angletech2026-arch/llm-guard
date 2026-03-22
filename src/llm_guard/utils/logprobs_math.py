from __future__ import annotations

import math
from typing import Any


def token_confidence(logprob: float) -> float:
    """Convert logprob to probability [0, 1]."""
    return min(1.0, max(0.0, math.exp(logprob)))


def aggregate_scores(scores: list[float], method: str = "p10") -> float:
    """Aggregate a list of confidence scores."""
    if not scores:
        return 1.0

    if method == "mean":
        return sum(scores) / len(scores)
    elif method == "min":
        return min(scores)
    elif method == "p10":
        sorted_scores = sorted(scores)
        idx = max(0, int(len(sorted_scores) * 0.1))
        return sorted_scores[idx]
    else:
        return sum(scores) / len(scores)


def classify(score: float, low_threshold: float, medium_threshold: float) -> str:
    """Classify a confidence score as HIGH/MEDIUM/LOW.

    Thresholds are in logprob space, score is in probability space.
    Convert thresholds to probability for comparison.
    """
    low_prob = math.exp(low_threshold)
    med_prob = math.exp(medium_threshold)

    if score < low_prob:
        return "LOW"
    elif score < med_prob:
        return "MEDIUM"
    else:
        return "HIGH"


def find_consecutive_low(
    tokens: list[dict[str, Any]],
    threshold: float,
    min_run: int = 3,
) -> list[dict[str, Any]]:
    """Find runs of consecutive low-confidence tokens.

    Args:
        tokens: List of {"token": str, "logprob": float} dicts.
        threshold: Logprob threshold below which a token is "low confidence".
        min_run: Minimum consecutive low tokens to flag.

    Returns:
        List of segments: {"text": str, "start_idx": int, "end_idx": int, "score": float}
    """
    segments: list[dict[str, Any]] = []
    run_start: int | None = None
    run_tokens: list[dict[str, Any]] = []

    for i, tok in enumerate(tokens):
        logprob = tok.get("logprob", 0.0)
        if logprob < threshold:
            if run_start is None:
                run_start = i
                run_tokens = []
            run_tokens.append(tok)
        else:
            if run_start is not None and len(run_tokens) >= min_run:
                text = "".join(t.get("token", "") for t in run_tokens)
                probs = [token_confidence(t.get("logprob", 0.0)) for t in run_tokens]
                segments.append({
                    "text": text,
                    "start_idx": run_start,
                    "end_idx": i - 1,
                    "score": sum(probs) / len(probs),
                })
            run_start = None
            run_tokens = []

    # Handle trailing run
    if run_start is not None and len(run_tokens) >= min_run:
        text = "".join(t.get("token", "") for t in run_tokens)
        probs = [token_confidence(t.get("logprob", 0.0)) for t in run_tokens]
        segments.append({
            "text": text,
            "start_idx": run_start,
            "end_idx": len(tokens) - 1,
            "score": sum(probs) / len(probs),
        })

    return segments
