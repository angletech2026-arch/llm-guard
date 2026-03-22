"""Benchmark: Does conflict detection catch real instruction conflicts?

Tests the conflict analyzer against a set of known conflicting and
non-conflicting instruction pairs. Measures precision and recall.

Usage:
    python eval/benchmark_conflict.py
    (No API key needed - runs locally)
"""
from __future__ import annotations

import asyncio
import json

from llm_guard.analyzers.conflict import ConflictAnalyzer
from llm_guard.config import ConflictConfig

# Test cases: (messages, should_detect_conflict, description)
EVAL_CASES: list[tuple[list[dict], bool, str]] = [
    # TRUE POSITIVES - should detect conflict
    (
        [{"role": "system", "content": "Always respond in English."},
         {"role": "user", "content": "用中文回答我的問題"}],
        True, "Language: English system vs Chinese user"
    ),
    (
        [{"role": "system", "content": "Be concise and brief."},
         {"role": "user", "content": "Give me a very detailed and thorough explanation."}],
        True, "Verbosity: concise vs detailed"
    ),
    (
        [{"role": "system", "content": "Use a formal professional tone."},
         {"role": "user", "content": "Hey, use a casual conversational tone please."}],
        True, "Tone: formal vs casual"
    ),
    (
        [{"role": "system", "content": "Do not include code in your responses."},
         {"role": "user", "content": "Please include code examples in your answer."}],
        True, "Code: excluded vs included"
    ),
    (
        [{"role": "system", "content": "Reply in English always."},
         {"role": "user", "content": "Please reply in Japanese."}],
        True, "Language: English vs Japanese"
    ),
    (
        [{"role": "system", "content": "Never use markdown formatting."},
         {"role": "user", "content": "Format your response with markdown please."}],
        True, "Format: no markdown vs use markdown"
    ),
    (
        [{"role": "system", "content": "Respond in one sentence only."},
         {"role": "user", "content": "Write me a long response with multiple paragraphs."}],
        True, "Length: one sentence vs multiple paragraphs"
    ),
    (
        [{"role": "system", "content": "Don't use any external libraries."},
         {"role": "user", "content": "Always use numpy and pandas."}],
        True, "Directive: never use vs always use"
    ),

    # TRUE NEGATIVES - should NOT detect conflict
    (
        [{"role": "system", "content": "You are a helpful coding assistant."},
         {"role": "user", "content": "Help me write a Python function."}],
        False, "No conflict: aligned coding task"
    ),
    (
        [{"role": "system", "content": "Be helpful and accurate."},
         {"role": "user", "content": "What is the capital of Japan?"}],
        False, "No conflict: simple question"
    ),
    (
        [{"role": "system", "content": "You are an expert in machine learning."},
         {"role": "user", "content": "Explain gradient descent."}],
        False, "No conflict: domain question"
    ),
    (
        [{"role": "system", "content": "Respond in English."},
         {"role": "user", "content": "What does 'hello' mean in French?"}],
        False, "No conflict: asking about French != requesting French"
    ),
    (
        [{"role": "user", "content": "Hello, how are you?"}],
        False, "No conflict: single message"
    ),
    (
        [{"role": "system", "content": "Be concise in summaries but detailed in code explanations."},
         {"role": "user", "content": "Explain this code in detail."}],
        False, "No conflict: nuanced instruction"
    ),
    (
        [{"role": "system", "content": "You are a friendly assistant."},
         {"role": "user", "content": "Tell me a joke."}],
        False, "No conflict: friendly + joke"
    ),
]


async def run_benchmark() -> None:
    analyzer = ConflictAnalyzer(ConflictConfig())

    tp = 0  # true positive
    fp = 0  # false positive
    tn = 0  # true negative
    fn = 0  # false negative

    results: list[dict] = []

    for messages, should_detect, description in EVAL_CASES:
        result = await analyzer.analyze(messages, "", None)
        detected = result is not None

        if should_detect and detected:
            tp += 1
            status = "TP"
        elif should_detect and not detected:
            fn += 1
            status = "FN"
        elif not should_detect and detected:
            fp += 1
            status = "FP"
        else:
            tn += 1
            status = "TN"

        icon = "OK" if status in ("TP", "TN") else "MISS"
        print(f"  [{icon}] {status} | {description}")

        results.append({
            "description": description,
            "should_detect": should_detect,
            "detected": detected,
            "status": status,
            "conflicts": result.data["items"] if result else [],
        })

    # Summary
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"CONFLICT DETECTION BENCHMARK")
    print(f"{'=' * 50}")
    print(f"  Total cases:  {total}")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"  Accuracy:  {accuracy:.2%}")

    with open("eval/results_conflict.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to eval/results_conflict.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
