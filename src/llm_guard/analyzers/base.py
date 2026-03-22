from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("llm_guard")


@dataclass
class AnalyzerResult:
    analyzer_name: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    @abstractmethod
    async def analyze(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult | None: ...


class LLMClientMixin:
    """Mixin for analyzers that need to call upstream LLM."""

    def _init_llm_client(
        self,
        upstream_base_url: str = "",
        upstream_api_key: str = "",
        upstream_timeout: int = 120,
        verify_ssl: bool = True,
    ) -> None:
        self.upstream_base_url = upstream_base_url
        self.upstream_api_key = upstream_api_key
        self.upstream_timeout = upstream_timeout
        self.verify_ssl = verify_ssl
        self._client: httpx.AsyncClient | None = None

    @property
    def _has_upstream(self) -> bool:
        return bool(self.upstream_base_url)

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

    def _build_llm_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"content-type": "application/json"}
        if self.upstream_api_key:
            headers["authorization"] = f"Bearer {self.upstream_api_key}"
        headers["x-llm-guard-internal"] = "true"
        return headers

    async def _call_llm(self, prompt: str, model: str, max_tokens: int = 1024) -> str:
        client = await self._get_client()
        headers = self._build_llm_headers()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        response = await client.post("/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_llm_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> str:
        client = await self._get_client()
        headers = self._build_llm_headers()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await client.post("/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {}
