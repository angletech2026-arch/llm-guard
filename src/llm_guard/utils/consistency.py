from __future__ import annotations

from typing import Any


def compute_consistency(original: str, samples: list[str]) -> dict[str, Any]:
    """Jaccard word-overlap consistency between original and samples."""
    original_words = set(original.lower().split())
    scores: list[float] = []
    divergent: int = 0

    for sample in samples:
        sample_words = set(sample.lower().split())
        if not original_words or not sample_words:
            scores.append(0.0)
            divergent += 1
            continue

        intersection = original_words & sample_words
        union = original_words | sample_words
        jaccard = len(intersection) / len(union) if union else 0.0
        scores.append(jaccard)

        if jaccard < 0.3:
            divergent += 1

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score > 0.7:
        summary = "Responses are highly consistent"
    elif avg_score > 0.4:
        summary = f"Moderate consistency ({divergent}/{len(samples)} divergent samples)"
    else:
        summary = f"Low consistency — {divergent}/{len(samples)} samples diverged significantly"

    return {
        "score": round(avg_score, 4),
        "divergent": divergent,
        "summary": summary,
    }
