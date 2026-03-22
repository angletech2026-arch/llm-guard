from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import Response

from llm_guard.config import GuardConfig
from llm_guard.proxy import ProxyHandler

logger = logging.getLogger("llm_guard")


def create_app(config: GuardConfig) -> FastAPI:
    handler = ProxyHandler(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await handler.startup()
        logger.info("LLM-Guard proxy ready")
        yield
        await handler.shutdown()

    app = FastAPI(title="LLM-Guard", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], response_model=None)
    async def proxy_v1(request: Request, path: str) -> Response:
        return await handler.handle(request, f"/v1/{path}")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], response_model=None)
    async def proxy_fallback(request: Request, path: str) -> Response:
        return await handler.handle(request, f"/{path}")

    return app
