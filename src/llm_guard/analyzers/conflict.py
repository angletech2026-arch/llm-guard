from __future__ import annotations

import logging
import re
from typing import Any

from llm_guard.analyzers.base import AnalyzerResult, BaseAnalyzer
from llm_guard.config import ConflictConfig
from llm_guard.utils.text_patterns import detect_arabic, detect_cjk, detect_cyrillic

logger = logging.getLogger("llm_guard")

# Contradictory instruction pattern pairs
# Each tuple: (pattern_a, pattern_b, description)
INSTRUCTION_CONFLICT_PATTERNS: list[tuple[str, str, str]] = [
    # Language conflicts
    (r"(?:respond|reply|answer|write)\s+(?:in\s+)?english", r"(?:respond|reply|answer|write)\s+(?:in\s+)?chinese", "Language: English vs Chinese"),
    (r"(?:respond|reply|answer|write)\s+(?:in\s+)?english", r"用中文", "Language: English vs Chinese"),
    (r"(?:respond|reply|answer|write)\s+(?:in\s+)?english", r"(?:respond|reply|answer|write)\s+(?:in\s+)?spanish", "Language: English vs Spanish"),
    (r"(?:respond|reply|answer|write)\s+(?:in\s+)?english", r"(?:respond|reply|answer|write)\s+(?:in\s+)?french", "Language: English vs French"),
    (r"(?:respond|reply|answer|write)\s+(?:in\s+)?english", r"(?:respond|reply|answer|write)\s+(?:in\s+)?japanese", "Language: English vs Japanese"),
    (r"(?:respond|reply|answer|write)\s+(?:in\s+)?english", r"日本語で", "Language: English vs Japanese"),
    # Verbosity conflicts
    (r"\b(?:be\s+)?(?:concise|brief|short|terse)\b", r"\b(?:be\s+)?(?:detailed|verbose|thorough|comprehensive|elaborate)\b", "Verbosity: concise vs detailed"),
    # Tone conflicts
    (r"\b(?:formal|professional)\s+(?:tone|style|language)\b", r"\b(?:casual|informal|friendly|conversational)\s+(?:tone|style|language)\b", "Tone: formal vs casual"),
    # Code inclusion
    (r"\b(?:no\s+code|(?:don'?t|do\s+not)\s+(?:include|show|write)\s+code|without\s+code)\b", r"\b(?:include\s+code|show\s+code|write\s+code|with\s+code\s+examples?)\b", "Code: excluded vs required"),
    # Format conflicts
    (r"\b(?:no\s+markdown|(?:never|don'?t|do\s+not)\s+(?:use\s+)?markdown|plain\s+text|without\s+(?:markdown|formatting))\b", r"\b(?:(?:use|with|format\s+(?:with|in|using))\s+markdown)\b", "Format: plain text vs markdown"),
    # Length conflicts
    (r"\b(?:one\s+(?:sentence|line|word)|single\s+(?:sentence|line))\b", r"\b(?:multiple\s+paragraphs?|long\s+(?:response|answer|explanation))\b", "Length: short vs long"),
    # Action conflicts
    (r"\bdo\s+not\s+(?:modify|change|edit|alter|touch)\b", r"\b(?:modify|change|edit|update|fix|alter)\b", "Action: do not modify vs modify"),
    (r"\b(?:never|don'?t|do\s+not)\s+(?:use|include|add)\b", r"\b(?:always|must|should)\s+(?:use|include|add)\b", "Directive: never use vs always use"),
]


class ConflictAnalyzer(BaseAnalyzer):
    def __init__(self, config: ConflictConfig) -> None:
        self.config = config

    async def analyze(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult | None:
        if not self.config.enabled:
            return None

        if not request_messages:
            return None

        conflicts: list[dict[str, Any]] = []

        if self.config.check_language_conflict:
            conflicts.extend(self._check_language_conflicts(request_messages))

        if self.config.check_instruction_conflict:
            conflicts.extend(self._check_instruction_conflicts(request_messages))

        conflicts.extend(self._check_custom_rules(request_messages))

        if not conflicts:
            return None

        return AnalyzerResult(analyzer_name="conflicts", data={"items": conflicts})

    def _get_priority(self, role: str) -> int:
        """Lower number = higher priority."""
        try:
            return self.config.priority.index(role)
        except ValueError:
            return len(self.config.priority)

    def _resolve(self, role_a: str, role_b: str) -> str:
        """Return which role wins based on priority config."""
        if self._get_priority(role_a) <= self._get_priority(role_b):
            return role_a
        return role_b

    def _check_language_conflicts(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect language mismatches between messages of different roles."""
        conflicts: list[dict[str, Any]] = []

        role_languages: dict[str, set[str]] = {}
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            langs = self._detect_languages(content)
            if role not in role_languages:
                role_languages[role] = set()
            role_languages[role].update(langs)

        # Check for explicit language instruction conflicts across roles
        roles = list(role_languages.keys())
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                role_a, role_b = roles[i], roles[j]
                lang_a = self._detect_explicit_language(messages, role_a)
                lang_b = self._detect_explicit_language(messages, role_b)

                if lang_a and lang_b and lang_a != lang_b:
                    winner = self._resolve(role_a, role_b)
                    winning_lang = lang_a if winner == role_a else lang_b
                    conflicts.append({
                        "type": "language_conflict",
                        "severity": "HIGH",
                        "message": f"{role_a} requests '{lang_a}' but {role_b} requests '{lang_b}'",
                        "sources": [role_a, role_b],
                        "resolution": f"Following {winner} (higher priority): {winning_lang}",
                    })

        return conflicts

    def _detect_languages(self, text: str) -> set[str]:
        """Detect which scripts/languages are present in text."""
        langs: set[str] = set()
        if detect_cjk(text):
            langs.add("cjk")
        if detect_cyrillic(text):
            langs.add("cyrillic")
        if detect_arabic(text):
            langs.add("arabic")
        if re.search(r"[a-zA-Z]{3,}", text):
            langs.add("latin")
        return langs

    def _detect_explicit_language(
        self, messages: list[dict[str, Any]], role: str
    ) -> str | None:
        """Detect if a specific role explicitly requests a language."""
        lang_patterns = {
            "english": r"(?:respond|reply|answer|write|speak)\s+(?:in\s+)?english",
            "chinese": r"(?:(?:respond|reply|answer|write|speak)\s+(?:in\s+)?chinese|用中文|繁體中文|簡體中文)",
            "japanese": r"(?:(?:respond|reply|answer|write|speak)\s+(?:in\s+)?japanese|日本語で)",
            "spanish": r"(?:respond|reply|answer|write|speak)\s+(?:in\s+)?spanish",
            "french": r"(?:respond|reply|answer|write|speak)\s+(?:in\s+)?french",
            "korean": r"(?:(?:respond|reply|answer|write|speak)\s+(?:in\s+)?korean|한국어로)",
        }

        for msg in messages:
            if msg.get("role") != role:
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            for lang, pattern in lang_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    return lang

        return None

    def _check_instruction_conflicts(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check for contradictory instruction patterns across different roles."""
        conflicts: list[dict[str, Any]] = []

        # Group message content by role
        role_content: dict[str, str] = {}
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if role not in role_content:
                role_content[role] = ""
            role_content[role] += " " + content

        roles = list(role_content.keys())

        for pattern_a, pattern_b, description in INSTRUCTION_CONFLICT_PATTERNS:
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    role_a, role_b = roles[i], roles[j]
                    text_a = role_content[role_a]
                    text_b = role_content[role_b]

                    match_a_in_a = re.search(pattern_a, text_a, re.IGNORECASE)
                    match_b_in_b = re.search(pattern_b, text_b, re.IGNORECASE)

                    match_b_in_a = re.search(pattern_b, text_a, re.IGNORECASE)
                    match_a_in_b = re.search(pattern_a, text_b, re.IGNORECASE)

                    if match_a_in_a and match_b_in_b:
                        winner = self._resolve(role_a, role_b)
                        conflicts.append({
                            "type": "instruction_conflict",
                            "severity": "MEDIUM",
                            "message": f"{description} — {role_a} vs {role_b}",
                            "sources": [role_a, role_b],
                            "resolution": f"Following {winner} (higher priority)",
                        })
                    elif match_b_in_a and match_a_in_b:
                        winner = self._resolve(role_a, role_b)
                        conflicts.append({
                            "type": "instruction_conflict",
                            "severity": "MEDIUM",
                            "message": f"{description} — {role_a} vs {role_b}",
                            "sources": [role_a, role_b],
                            "resolution": f"Following {winner} (higher priority)",
                        })

        return conflicts

    def _check_custom_rules(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check user-defined custom conflict rules."""
        conflicts: list[dict[str, Any]] = []

        all_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if isinstance(msg.get("content"), str)
        )

        for rule in self.config.custom_rules:
            match_a = re.search(rule.pattern_a, all_text, re.IGNORECASE)
            match_b = re.search(rule.pattern_b, all_text, re.IGNORECASE)

            if match_a and match_b:
                conflicts.append({
                    "type": "custom_conflict",
                    "severity": "MEDIUM",
                    "message": rule.message or f"Custom rule conflict: '{rule.pattern_a}' vs '{rule.pattern_b}'",
                    "sources": ["custom_rule"],
                    "resolution": "User-defined rule triggered",
                })

        return conflicts
