"""Benchmark: Does confidence scoring correlate with actual correctness?

Sends factual questions through LLM-Guard proxy, records confidence scores,
then checks if low-confidence answers are actually more likely to be wrong.

Usage:
    # Start proxy first: llm-guard --port 8400
    # Then: python eval/benchmark_confidence.py
    # Requires OPENAI_API_KEY env var
"""
from __future__ import annotations

import json
import os
import sys

import httpx

# Questions with known answers for evaluation
EVAL_QUESTIONS = [
    # Easy / high confidence expected
    {"q": "What is 2 + 2?", "expected": "4", "difficulty": "easy"},
    {"q": "What is the capital of France?", "expected": "Paris", "difficulty": "easy"},
    {"q": "What color is the sky on a clear day?", "expected": "blue", "difficulty": "easy"},
    {"q": "How many days are in a week?", "expected": "7", "difficulty": "easy"},
    {"q": "What planet do we live on?", "expected": "Earth", "difficulty": "easy"},
    # Medium
    {"q": "What year did World War I begin?", "expected": "1914", "difficulty": "medium"},
    {"q": "What is the chemical symbol for gold?", "expected": "Au", "difficulty": "medium"},
    {"q": "Who wrote Romeo and Juliet?", "expected": "Shakespeare", "difficulty": "medium"},
    {"q": "What is the speed of light in km/s (approximately)?", "expected": "300000", "difficulty": "medium"},
    {"q": "What is the largest ocean on Earth?", "expected": "Pacific", "difficulty": "medium"},
    # Hard / low confidence expected (obscure facts)
    {"q": "What is the population of Liechtenstein to the nearest thousand?", "expected": "39000", "difficulty": "hard"},
    {"q": "In what year was the Treaty of Tordesillas signed?", "expected": "1494", "difficulty": "hard"},
    {"q": "What is the atomic number of Rutherfordium?", "expected": "104", "difficulty": "hard"},
    {"q": "Who was the 23rd President of the United States?", "expected": "Benjamin Harrison", "difficulty": "hard"},
    {"q": "What is the GDP of Tuvalu in USD (approximately)?", "expected": "60 million", "difficulty": "hard"},
]

PROXY_URL = "http://127.0.0.1:8400/v1/chat/completions"


def run_benchmark() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    results: list[dict] = []

    with httpx.Client(timeout=60) as client:
        for i, item in enumerate(EVAL_QUESTIONS):
            print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {item['q'][:50]}...", end=" ", flush=True)

            try:
                response = client.post(
                    PROXY_URL,
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "Answer concisely in one sentence."},
                            {"role": "user", "content": item["q"]},
                        ],
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                data = response.json()
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            guard = data.get("llm_guard", {})
            confidence = guard.get("confidence", {})

            overall = confidence.get("overall", "N/A")
            score = confidence.get("overall_score", -1)
            low_segs = len(confidence.get("low_confidence_segments", []))

            # Simple correctness check
            correct = item["expected"].lower() in answer.lower()

            results.append({
                "question": item["q"],
                "difficulty": item["difficulty"],
                "expected": item["expected"],
                "answer": answer[:100],
                "correct": correct,
                "confidence_level": overall,
                "confidence_score": score,
                "low_segments": low_segs,
            })

            status = "OK" if correct else "WRONG"
            print(f"{status} | confidence={overall} ({score:.3f})")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for difficulty in ["easy", "medium", "hard"]:
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue

        correct_count = sum(1 for r in group if r["correct"])
        avg_score = sum(r["confidence_score"] for r in group if r["confidence_score"] >= 0) / max(1, len(group))
        avg_low_segs = sum(r["low_segments"] for r in group) / max(1, len(group))

        print(f"\n{difficulty.upper()} (n={len(group)}):")
        print(f"  Accuracy: {correct_count}/{len(group)} ({100*correct_count/len(group):.0f}%)")
        print(f"  Avg confidence score: {avg_score:.3f}")
        print(f"  Avg low-confidence segments: {avg_low_segs:.1f}")

    # Correlation check
    correct_scores = [r["confidence_score"] for r in results if r["correct"] and r["confidence_score"] >= 0]
    wrong_scores = [r["confidence_score"] for r in results if not r["correct"] and r["confidence_score"] >= 0]

    if correct_scores and wrong_scores:
        print(f"\nCORRELATION CHECK:")
        print(f"  Avg confidence (correct answers): {sum(correct_scores)/len(correct_scores):.3f}")
        print(f"  Avg confidence (wrong answers):   {sum(wrong_scores)/len(wrong_scores):.3f}")
        if sum(correct_scores)/len(correct_scores) > sum(wrong_scores)/len(wrong_scores):
            print("  -> Confidence scores DO correlate with correctness")
        else:
            print("  -> Confidence scores do NOT correlate with correctness")
    elif not wrong_scores:
        print(f"\nAll answers correct — cannot measure correlation with errors.")

    # Save raw results
    with open("eval/results_confidence.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to eval/results_confidence.json")


if __name__ == "__main__":
    run_benchmark()
