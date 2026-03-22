from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_guard.config import GuardConfig
from llm_guard.server import create_app


def make_chat_response(
    content: str = "Hello! How can I help you?",
    model: str = "gpt-4o-mini",
    logprobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }
    if logprobs is not None:
        choice["logprobs"] = logprobs

    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "model": model,
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def make_streaming_chunks(
    content: str = "Hello! How can I help you?",
    model: str = "gpt-4o-mini",
) -> list[str]:
    chunks = []
    for i, char in enumerate(content):
        chunk = {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(chunk)}")

    final = {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    chunks.append(f"data: {json.dumps(final)}")
    chunks.append("data: [DONE]")
    return chunks


def create_mock_upstream() -> FastAPI:
    mock = FastAPI()

    @mock.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        is_streaming = body.get("stream", False)

        if is_streaming:
            content = "Hello from mock!"
            chunks = make_streaming_chunks(content)

            async def generate():
                for chunk in chunks:
                    yield f"{chunk}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        return JSONResponse(content=make_chat_response())

    @mock.get("/v1/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": "gpt-4o-mini", "object": "model"}],
        }

    return mock


@pytest.fixture
def mock_upstream():
    return create_mock_upstream()


@pytest.fixture
def config() -> GuardConfig:
    return GuardConfig()


@pytest.fixture
def app(config: GuardConfig) -> FastAPI:
    return create_app(config)
