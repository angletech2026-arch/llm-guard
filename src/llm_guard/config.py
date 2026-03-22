from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("llm_guard")


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8400
    log_level: str = "info"


class UpstreamConfig(BaseModel):
    base_url: str = "https://api.openai.com"
    api_key: str = ""
    timeout: int = 120
    verify_ssl: bool = True


class ConfidenceConfig(BaseModel):
    enabled: bool = True
    low_threshold: float = -3.0
    medium_threshold: float = -1.0
    aggregate_method: str = "p10"
    min_consecutive_low: int = 3
    # Multi-sample fallback when logprobs unavailable
    fallback_enabled: bool = False
    fallback_samples: int = 3
    fallback_temperature: float = 0.8
    fallback_model: str = "gpt-4o-mini"


class ConflictRule(BaseModel):
    pattern_a: str
    pattern_b: str
    message: str = ""


class ConflictConfig(BaseModel):
    enabled: bool = True
    priority: list[str] = Field(default_factory=lambda: ["system", "user", "context"])
    check_language_conflict: bool = True
    check_instruction_conflict: bool = True
    custom_rules: list[ConflictRule] = Field(default_factory=list)
    # LLM-based fallback when regex finds nothing
    llm_fallback_enabled: bool = False
    llm_fallback_model: str = "gpt-4o-mini"
    llm_fallback_max_tokens: int = 256


class VerificationConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o-mini"
    mode: str = "self_check"
    samples: int = 3
    max_tokens: int = 1024
    min_response_length: int = 200


class AnalyzersConfig(BaseModel):
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    conflict: ConflictConfig = Field(default_factory=ConflictConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)


class OutputConfig(BaseModel):
    mode: str = "header"  # "header" (default, safest) | "metadata" | "both"
    verbose_confidence: bool = False
    streaming_analysis: str = "none"  # "none" (default) | "sse_chunk"


class GuardConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    analyzers: AnalyzersConfig = Field(default_factory=AnalyzersConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path | None = None) -> GuardConfig:
    if path is None:
        logger.info("No config file specified, using defaults")
        return GuardConfig()

    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file %s not found, using defaults", config_path)
        return GuardConfig()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return GuardConfig(**raw)
