from __future__ import annotations

import tempfile

from llm_guard.config import GuardConfig, load_config


def test_default_config():
    config = GuardConfig()
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8400
    assert config.upstream.base_url == "https://api.openai.com"
    assert config.analyzers.confidence.enabled is True
    assert config.analyzers.conflict.enabled is True
    assert config.analyzers.verification.enabled is False
    assert config.output.mode == "header"
    assert config.analyzers.confidence.fallback_enabled is False
    assert config.analyzers.conflict.llm_fallback_enabled is False


def test_load_config_no_file():
    config = load_config(None)
    assert config.server.port == 8400


def test_load_config_missing_file():
    config = load_config("/nonexistent/config.yaml")
    assert config.server.port == 8400


def test_load_config_from_yaml():
    yaml_content = """
server:
  host: "0.0.0.0"
  port: 9090
upstream:
  base_url: "http://localhost:11434"
analyzers:
  confidence:
    enabled: false
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9090
    assert config.upstream.base_url == "http://localhost:11434"
    assert config.analyzers.confidence.enabled is False
    assert config.analyzers.conflict.enabled is True  # default


def test_load_config_empty_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        config = load_config(f.name)

    assert config.server.port == 8400
