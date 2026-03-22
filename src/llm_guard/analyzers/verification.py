from __future__ import annotations

import logging
from typing import Any

from llm_guard.analyzers.base import AnalyzerResult, BaseAnalyzer, LLMClientMixin
from llm_guard.config import VerificationConfig
from llm_guard.utils.consistency import compute_consistency

logger = logging.getLogger("llm_guard")

SELF_CHECK_PROMPT = """You are a reasoning verification assistant. Analyze the following LLM response for:
1. Logical errors or contradictions
2. Factual inconsistencies within the response
3. Unsupported conclusions or reasoning jumps
4. Mathematical or numerical errors

Original user request:
{user_message}

LLM response to verify:
{response_text}

Respond in this exact JSON format:
{{"pass": true/false, "issues": ["issue 1", "issue 2"], "summary": "brief summary"}}

If the response is logically sound and consistent, set pass=true and issues=[].
Only flag clear errors, not style preferences."""


class VerificationAnalyzer(BaseAnalyzer, LLMClientMixin):
    def __init__(
        self,
        config: VerificationConfig,
        upstream_base_url: str = "",
        upstream_api_key: str = "",
        upstream_timeout: int = 120,
        verify_ssl: bool = True,
    ) -> None:
        self.config = config
        self._init_llm_client(upstream_base_url, upstream_api_key, upstream_timeout, verify_ssl)

    async def analyze(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult | None:
        if not self.config.enabled:
            return None

        word_count = len(response_text.split())
        if word_count < self.config.min_response_length:
            return None

        if self.config.mode == "self_check":
            return await self._self_check(request_messages, response_text)
        elif self.config.mode == "multi_sample":
            return await self._multi_sample(request_messages, response_text)

        return None

    async def _self_check(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
    ) -> AnalyzerResult | None:
        user_message = self._extract_user_message(request_messages)
        prompt = SELF_CHECK_PROMPT.format(
            user_message=user_message,
            response_text=response_text,
        )

        try:
            text = await self._call_llm(prompt, self.config.model, self.config.max_tokens)
            data = self._parse_json_response(text)
            result = {
                "pass": data.get("pass"),
                "issues": data.get("issues", []),
                "summary": data.get("summary", text[:500]),
            }
            return AnalyzerResult(analyzer_name="verification", data=result)
        except Exception as e:
            logger.error("Verification self_check failed: %s", e)
            return AnalyzerResult(
                analyzer_name="verification",
                data={"pass": None, "issues": [], "summary": f"Verification failed: {e}", "error": True},
            )

    async def _multi_sample(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
    ) -> AnalyzerResult | None:
        try:
            samples: list[str] = []
            for _ in range(self.config.samples):
                sample = await self._call_llm_chat(
                    request_messages, self.config.model, self.config.max_tokens, temperature=1.0
                )
                samples.append(sample)

            consistency = compute_consistency(response_text, samples)
            return AnalyzerResult(
                analyzer_name="verification",
                data={
                    "pass": consistency["score"] > 0.7,
                    "consistency_score": consistency["score"],
                    "divergent_samples": consistency["divergent"],
                    "summary": consistency["summary"],
                },
            )
        except Exception as e:
            logger.error("Verification multi_sample failed: %s", e)
            return AnalyzerResult(
                analyzer_name="verification",
                data={"pass": None, "issues": [], "summary": f"Verification failed: {e}", "error": True},
            )

    @staticmethod
    def _extract_user_message(messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""
