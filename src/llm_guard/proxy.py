from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_guard.analyzers.base import AnalyzerResult, BaseAnalyzer
from llm_guard.analyzers.confidence import ConfidenceAnalyzer
from llm_guard.analyzers.conflict import ConflictAnalyzer
from llm_guard.analyzers.verification import VerificationAnalyzer
from llm_guard.config import GuardConfig
from llm_guard.enrichment.request_enricher import enrich_request
from llm_guard.enrichment.response_enricher import enrich_response
from llm_guard.streaming import stream_and_analyze

logger = logging.getLogger("llm_guard")

FORWARDED_HEADERS = {
    "authorization",
    "content-type",
    "accept",
    "user-agent",
}


class ProxyHandler:
    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self.client: httpx.AsyncClient | None = None
        self.analyzers: list[BaseAnalyzer] = []

    async def startup(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.config.upstream.base_url,
            timeout=httpx.Timeout(self.config.upstream.timeout),
            verify=self.config.upstream.verify_ssl,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

        if self.config.analyzers.confidence.enabled:
            self.analyzers.append(ConfidenceAnalyzer(self.config.analyzers.confidence))

        if self.config.analyzers.conflict.enabled:
            self.analyzers.append(ConflictAnalyzer(self.config.analyzers.conflict))

        if self.config.analyzers.verification.enabled:
            self.analyzers.append(VerificationAnalyzer(
                config=self.config.analyzers.verification,
                upstream_base_url=self.config.upstream.base_url,
                upstream_api_key=self.config.upstream.api_key,
                upstream_timeout=self.config.upstream.timeout,
                verify_ssl=self.config.upstream.verify_ssl,
            ))

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
        for analyzer in self.analyzers:
            if isinstance(analyzer, VerificationAnalyzer):
                await analyzer.close()

    def _build_headers(self, request: Request) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            if key.lower() in FORWARDED_HEADERS:
                headers[key] = value

        if self.config.upstream.api_key:
            headers["authorization"] = f"Bearer {self.config.upstream.api_key}"

        return headers

    def _is_internal(self, request: Request) -> bool:
        """Check if this is an internal request from verification analyzer."""
        return request.headers.get("x-llm-guard-internal") == "true"

    async def handle(self, request: Request, path: str) -> JSONResponse | StreamingResponse:
        assert self.client is not None

        # Skip analysis for internal requests (prevents recursion)
        if self._is_internal(request):
            logger.debug("Internal request, passthrough: %s", path)
            headers = self._build_headers(request)
            body = await request.body()
            return await self._handle_normal(method=request.method, path=path, headers=headers, body=body)

        headers = self._build_headers(request)
        body = await request.body()

        is_chat_completions = path.rstrip("/") == "/v1/chat/completions"
        is_streaming = False
        request_data: dict[str, Any] | None = None
        logprobs_injected = False

        if is_chat_completions and body:
            try:
                request_data = json.loads(body)
                is_streaming = request_data.get("stream", False)

                # Enrich request (inject logprobs)
                request_data, logprobs_injected = enrich_request(request_data, self.config)
                body = json.dumps(request_data).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        logger.debug("Proxying %s %s (stream=%s)", request.method, path, is_streaming)

        try:
            if is_streaming:
                return await self._handle_streaming(
                    request.method, path, headers, body,
                    request_data, logprobs_injected,
                )
            else:
                return await self._handle_normal(
                    request.method, path, headers, body,
                    request_data, logprobs_injected,
                )
        except httpx.ConnectError as e:
            logger.error("Cannot connect to upstream: %s", e)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Cannot connect to upstream: {e}", "type": "proxy_error"}},
            )
        except httpx.TimeoutException as e:
            logger.error("Upstream timeout: %s", e)
            return JSONResponse(
                status_code=504,
                content={"error": {"message": f"Upstream timeout: {e}", "type": "proxy_error"}},
            )

    async def _run_analyzers(
        self,
        request_data: dict[str, Any] | None,
        response_text: str,
        logprobs: list[dict[str, Any]] | None,
    ) -> list[AnalyzerResult]:
        results: list[AnalyzerResult] = []
        messages = (request_data or {}).get("messages", [])

        for analyzer in self.analyzers:
            try:
                result = await analyzer.analyze(messages, response_text, logprobs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error("Analyzer %s failed: %s", type(analyzer).__name__, e)

        return results

    async def _handle_normal(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        request_data: dict[str, Any] | None = None,
        logprobs_injected: bool = False,
    ) -> JSONResponse:
        assert self.client is not None

        response = await self.client.request(
            method=method,
            url=path,
            headers=headers,
            content=body,
        )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            data = {"raw": response.text}

        # Run analyzers on chat completions
        if request_data is not None and "choices" in data:
            response_text = ""
            logprobs_data = None

            choice = data.get("choices", [{}])[0] if data.get("choices") else {}
            message = choice.get("message", {})
            response_text = message.get("content", "")

            lp = choice.get("logprobs")
            if lp and "content" in lp:
                logprobs_data = lp["content"]

            results = await self._run_analyzers(request_data, response_text, logprobs_data)

            if results:
                data = enrich_response(data, results, self.config, logprobs_injected)

        return JSONResponse(status_code=response.status_code, content=data)

    async def _handle_streaming(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        request_data: dict[str, Any] | None = None,
        logprobs_injected: bool = False,
    ) -> StreamingResponse:
        assert self.client is not None

        req = self.client.build_request(
            method=method,
            url=path,
            headers=headers,
            content=body,
        )
        upstream_response = await self.client.send(req, stream=True)

        return StreamingResponse(
            stream_and_analyze(
                upstream_response,
                self.analyzers,
                self.config,
                request_data,
                logprobs_injected,
            ),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "connection": "keep-alive"},
        )
