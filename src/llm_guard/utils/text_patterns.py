from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?。！？\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def detect_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))


def detect_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r"[\u0400-\u04ff]", text))


def detect_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06ff]", text))
