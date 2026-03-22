from __future__ import annotations


import pytest
from httpx import ASGITransport, AsyncClient

from llm_guard.config import GuardConfig
from llm_guard.server import create_app
from tests.conftest import create_mock_upstream


@pytest.fixture
def upstream_app():
    return create_mock_upstream()


@pytest.fixture
async def proxy_client(upstream_app):
    config = GuardConfig()
    config.upstream.base_url = "http://mock-upstream"
    proxy_app = create_app(config)
    proxy_transport = ASGITransport(app=proxy_app)
    async with AsyncClient(transport=proxy_transport, base_url="http://localhost:8400") as client:
        yield client


@pytest.mark.asyncio
async def test_health():
    config = GuardConfig()
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_config_defaults():
    config = GuardConfig()
    assert config.server.port == 8400
    assert config.upstream.base_url == "https://api.openai.com"
