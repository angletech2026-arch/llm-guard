"""Demo: Show LLM-Guard in action.

Start the proxy first:
    llm-guard --port 8400

Then run this script:
    OPENAI_API_KEY=sk-... python examples/demo.py
"""
from __future__ import annotations

import json
import os
import sys

import httpx

PROXY_URL = "http://127.0.0.1:8400/v1/chat/completions"


def demo_confidence() -> None:
    """Show confidence scoring on a factual question."""
    print("=" * 60)
    print("DEMO 1: Confidence Calibration")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    response = httpx.post(
        PROXY_URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What is the population of Nauru?"},
            ],
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    data = response.json()

    answer = data["choices"][0]["message"]["content"]
    guard = data.get("llm_guard", {})

    print(f"\nQuestion: What is the population of Nauru?")
    print(f"Answer: {answer}")
    print(f"\nLLM-Guard Analysis:")
    print(json.dumps(guard, indent=2))


def demo_conflict() -> None:
    """Show conflict detection with contradictory instructions."""
    print("\n" + "=" * 60)
    print("DEMO 2: Instruction Conflict Detection")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    response = httpx.post(
        PROXY_URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Always respond in English. Be concise."},
                {"role": "user", "content": "用中文詳細解釋什麼是量子力學"},
            ],
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    data = response.json()

    answer = data["choices"][0]["message"]["content"]
    guard = data.get("llm_guard", {})

    print(f"\nSystem: Always respond in English. Be concise.")
    print(f"User: 用中文詳細解釋什麼是量子力學")
    print(f"\nAnswer: {answer[:200]}...")
    print(f"\nLLM-Guard Analysis:")
    print(json.dumps(guard, indent=2))


def demo_no_conflict() -> None:
    """Show that normal requests pass through cleanly."""
    print("\n" + "=" * 60)
    print("DEMO 3: Normal Request (no issues)")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    response = httpx.post(
        PROXY_URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    data = response.json()

    answer = data["choices"][0]["message"]["content"]
    guard = data.get("llm_guard", {})

    print(f"\nQuestion: What is 2+2?")
    print(f"Answer: {answer}")
    print(f"\nLLM-Guard Analysis:")
    if guard:
        print(json.dumps(guard, indent=2))
    else:
        print("  (no issues detected)")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first.")
        print("Usage: OPENAI_API_KEY=sk-... python examples/demo.py")
        sys.exit(1)

    demo_confidence()
    demo_conflict()
    demo_no_conflict()
