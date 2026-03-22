from __future__ import annotations

import logging
from typing import Any

from llm_guard.analyzers.base import AnalyzerResult, BaseAnalyzer
from llm_guard.config import ConfidenceConfig
from llm_guard.utils.logprobs_math import (
    aggregate_scores,
    classify,
    find_consecutive_low,
    token_confidence,
)
from llm_guard.utils.text_patterns import split_sentences

logger = logging.getLogger("llm_guard")


class ConfidenceAnalyzer(BaseAnalyzer):
    def __init__(self, config: ConfidenceConfig) -> None:
        self.config = config

    async def analyze(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult | None:
        if not self.config.enabled:
            return None

        if logprobs is None or len(logprobs) == 0:
            logger.debug("No logprobs available, skipping confidence analysis")
            return None

        # Extract token-level data
        tokens = self._extract_tokens(logprobs)
        if not tokens:
            return None

        # Compute per-token confidence
        all_scores = [token_confidence(t["logprob"]) for t in tokens]

        # Overall score
        overall_score = aggregate_scores(all_scores, self.config.aggregate_method)
        overall_level = classify(
            overall_score, self.config.low_threshold, self.config.medium_threshold
        )

        # Find low-confidence segments
        low_segments = find_consecutive_low(
            tokens, self.config.low_threshold, self.config.min_consecutive_low
        )

        # Per-sentence breakdown
        sentences = self._score_sentences(response_text, tokens)

        data: dict[str, Any] = {
            "overall": overall_level,
            "overall_score": round(overall_score, 4),
            "low_confidence_segments": [
                {
                    "text": seg["text"],
                    "position": [seg["start_idx"], seg["end_idx"]],
                    "score": round(seg["score"], 4),
                    "level": "LOW",
                }
                for seg in low_segments
            ],
            "sentence_scores": sentences,
        }

        return AnalyzerResult(analyzer_name="confidence", data=data)

    def _extract_tokens(self, logprobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract token/logprob pairs from OpenAI logprobs format."""
        tokens: list[dict[str, Any]] = []
        for entry in logprobs:
            token = entry.get("token", "")
            logprob = entry.get("logprob")
            if logprob is not None:
                tokens.append({"token": token, "logprob": logprob})
        return tokens

    def _score_sentences(
        self, text: str, tokens: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score each sentence by mapping tokens back to sentence boundaries."""
        sentences = split_sentences(text)
        if not sentences:
            return []

        result: list[dict[str, Any]] = []
        token_idx = 0
        reconstructed = ""

        for sentence in sentences:
            sentence_scores: list[float] = []

            while token_idx < len(tokens):
                tok_text = tokens[token_idx]["token"]
                tok_logprob = tokens[token_idx]["logprob"]
                reconstructed += tok_text
                sentence_scores.append(token_confidence(tok_logprob))
                token_idx += 1

                if len(reconstructed.rstrip()) >= len(sentence.rstrip()):
                    break

            if sentence_scores:
                score = aggregate_scores(sentence_scores, self.config.aggregate_method)
                level = classify(score, self.config.low_threshold, self.config.medium_threshold)
                result.append({
                    "text": sentence[:80],
                    "score": round(score, 4),
                    "level": level,
                })

            reconstructed = ""

        return result
