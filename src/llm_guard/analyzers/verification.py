from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from llm_guard.analyzers.base import AnalyzerResult, BaseAnalyzer
from llm_guard.config import VerificationConfig

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


class VerificationAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        config: VerificationConfig,
        upstream_base_url: str,
        upstream_api_key: str = "",
        upstream_timeout: int = 120,
        verify_ssl: bool = True,
    ) -> None:
        self.config = config
        self.upstream_base_url = upstream_base_url
        self.upstream_api_key = upstream_api_key
        self.upstream_timeout = upstream_timeout
        self.verify_ssl = verify_ssl
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.upstream_base_url,
                timeout=httpx.Timeout(self.upstream_timeout),
                verify=self.verify_ssl,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def analyze(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult | None:
        if not self.config.enabled:
            return None

        # Skip short responses
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
            verification_response = await self._call_llm(prompt)
            result = self._parse_verification_response(verification_response)
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
                sample = await self._call_llm_chat(request_messages, temperature=1.0)
                samples.append(sample)

            consistency = self._compute_consistency(response_text, samples)
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

    async def _call_llm(self, prompt: str) -> str:
        client = await self._get_client()
        headers: dict[str, str] = {"content-type": "application/json"}
        if self.upstream_api_key:
            headers["authorization"] = f"Bearer {self.upstream_api_key}"

        # Internal request header to prevent proxy recursion
        headers["x-llm-guard-internal"] = "true"

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.0,
        }

        response = await client.post("/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_llm_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
    ) -> str:
        client = await self._get_client()
        headers: dict[str, str] = {"content-type": "application/json"}
        if self.upstream_api_key:
            headers["authorization"] = f"Bearer {self.upstream_api_key}"
        headers["x-llm-guard-internal"] = "true"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": temperature,
        }

        response = await client.post("/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _extract_user_message(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _parse_verification_response(self, text: str) -> dict[str, Any]:
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            return {
                "pass": data.get("pass", None),
                "issues": data.get("issues", []),
                "summary": data.get("summary", ""),
            }
        except (json.JSONDecodeError, ValueError):
            # If LLM didn't return valid JSON, treat as a text summary
            return {
                "pass": None,
                "issues": [],
                "summary": text[:500],
            }

    def _compute_consistency(
        self, original: str, samples: list[str]
    ) -> dict[str, Any]:
        """Simple word-overlap consistency check between original and samples."""
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
