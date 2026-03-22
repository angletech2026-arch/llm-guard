from __future__ import annotations

import pytest

from llm_guard.analyzers.conflict import ConflictAnalyzer
from llm_guard.config import ConflictConfig, ConflictRule


@pytest.fixture
def analyzer():
    return ConflictAnalyzer(ConflictConfig())


# --- Language conflicts ---

@pytest.mark.asyncio
async def test_language_conflict_english_vs_chinese(analyzer):
    messages = [
        {"role": "system", "content": "Always respond in English."},
        {"role": "user", "content": "用中文回答我的問題"},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None
    items = result.data["items"]
    assert len(items) >= 1
    assert any("language" in c["type"].lower() for c in items)
    # system has higher priority
    assert any("system" in c["resolution"] for c in items)


@pytest.mark.asyncio
async def test_language_conflict_explicit_instructions(analyzer):
    messages = [
        {"role": "system", "content": "Reply in English only."},
        {"role": "user", "content": "Please reply in Japanese."},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None
    items = result.data["items"]
    assert len(items) >= 1


@pytest.mark.asyncio
async def test_no_language_conflict(analyzer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is None


# --- Instruction conflicts ---

@pytest.mark.asyncio
async def test_verbosity_conflict(analyzer):
    messages = [
        {"role": "system", "content": "Be concise in all responses."},
        {"role": "user", "content": "Give me a detailed explanation of quantum physics."},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None
    items = result.data["items"]
    assert any("verbosity" in c["message"].lower() or "concise" in c["message"].lower() for c in items)


@pytest.mark.asyncio
async def test_tone_conflict(analyzer):
    messages = [
        {"role": "system", "content": "Use a formal tone always."},
        {"role": "user", "content": "Hey can you use a casual tone for this?"},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None
    items = result.data["items"]
    assert any("tone" in c["message"].lower() for c in items)


@pytest.mark.asyncio
async def test_code_conflict(analyzer):
    messages = [
        {"role": "system", "content": "Do not include code in responses."},
        {"role": "user", "content": "Please include code examples."},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None


@pytest.mark.asyncio
async def test_no_instruction_conflict(analyzer):
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Help me write a function."},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is None


# --- Custom rules ---

@pytest.mark.asyncio
async def test_custom_rule():
    config = ConflictConfig(
        custom_rules=[
            ConflictRule(
                pattern_a=r"never\s+use\s+jQuery",
                pattern_b=r"use\s+jQuery",
                message="jQuery usage conflict",
            )
        ]
    )
    analyzer = ConflictAnalyzer(config)
    messages = [
        {"role": "system", "content": "Never use jQuery in this project."},
        {"role": "user", "content": "Can you use jQuery to solve this?"},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None
    items = result.data["items"]
    assert any("jQuery" in c["message"] for c in items)


# --- Priority resolution ---

@pytest.mark.asyncio
async def test_priority_user_over_context():
    config = ConflictConfig(priority=["user", "system"])
    analyzer = ConflictAnalyzer(config)
    messages = [
        {"role": "system", "content": "Always respond in English."},
        {"role": "user", "content": "Reply in Chinese please."},
    ]
    result = await analyzer.analyze(messages, "", None)
    assert result is not None
    items = result.data["items"]
    # user should win since priority is ["user", "system"]
    assert any("user" in c["resolution"] for c in items)


# --- Edge cases ---

@pytest.mark.asyncio
async def test_disabled():
    config = ConflictConfig(enabled=False)
    analyzer = ConflictAnalyzer(config)
    result = await analyzer.analyze(
        [{"role": "system", "content": "Respond in English."},
         {"role": "user", "content": "用中文回答"}],
        "", None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_empty_messages(analyzer):
    result = await analyzer.analyze([], "", None)
    assert result is None


@pytest.mark.asyncio
async def test_single_message_no_conflict(analyzer):
    messages = [{"role": "user", "content": "Hello"}]
    result = await analyzer.analyze(messages, "", None)
    assert result is None
