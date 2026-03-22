# LLM-Guard

[![CI](https://github.com/angletech2026-arch/llm-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/angletech2026-arch/llm-guard/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Local proxy server that improves LLM reasoning quality via confidence calibration, instruction conflict detection, and optional verification.

## How It Works

```
Your App  -->  LLM-Guard (localhost:8400)  -->  LLM API (OpenAI, Ollama, etc.)
                    |
                    v
              Analyzes responses:
              - Confidence scoring (logprobs)
              - Instruction conflict detection
              - Reasoning verification (optional)
```

Change one line (`base_url`) and every LLM response gets analyzed automatically.

## Quick Start

```bash
pip install -e .
llm-guard --port 8400
```

Then point your LLM client to `http://127.0.0.1:8400`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8400/v1")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response)  # includes llm_guard analysis metadata
```

## Features

### 1. Confidence Calibration (zero extra cost)

Uses OpenAI's `logprobs` to score how confident the model is about each part of its response.

```json
{
  "llm_guard": {
    "confidence": {
      "overall": "HIGH",
      "overall_score": 0.87,
      "low_confidence_segments": [
        {
          "text": "The population of Mars is approximately 7 billion",
          "score": 0.12,
          "level": "LOW"
        }
      ]
    }
  }
}
```

### 2. Instruction Conflict Detection (zero extra cost)

Detects contradictions between system prompt and user messages before the LLM responds.

```json
{
  "llm_guard": {
    "conflicts": {
      "items": [
        {
          "type": "language_conflict",
          "severity": "HIGH",
          "message": "system requests 'english' but user requests 'chinese'",
          "resolution": "Following system (higher priority): english"
        }
      ]
    }
  }
}
```

Detects: language mismatches, verbosity conflicts, tone conflicts, code inclusion conflicts, format conflicts, and custom rules.

### 3. Reasoning Verification (optional, extra LLM call)

Off by default. When enabled, runs a second LLM call to verify the response for logical errors.

Two modes:
- **self_check**: Asks a model to review the response for errors
- **multi_sample**: Generates multiple responses and checks consistency

## Configuration

Copy `config.example.yaml` to `config.yaml` and customize:

```bash
cp config.example.yaml config.yaml
llm-guard --config config.yaml
```

Key settings:

```yaml
upstream:
  base_url: "https://api.openai.com"  # or http://localhost:11434 for Ollama
  api_key: "sk-..."

analyzers:
  confidence:
    enabled: true
  conflict:
    enabled: true
  verification:
    enabled: false  # opt-in, costs extra
```

## Compatible With

Any tool that supports OpenAI-compatible APIs:
- OpenAI, Azure OpenAI
- Ollama, LM Studio
- DeepSeek, Groq, Together AI
- Cursor, aider, Continue
- LangChain, LlamaIndex
- Any custom app using the OpenAI SDK

## Docker

```bash
docker build -t llm-guard .
docker run -p 8400:8400 llm-guard
```

## Benchmarks

### Conflict Detection

```
Precision: 100%  |  Recall: 100%  |  F1: 100%  |  Accuracy: 100%
(15 test cases: 8 true conflicts, 7 non-conflicts)
```

Run benchmarks yourself:

```bash
python eval/benchmark_conflict.py                    # no API key needed
OPENAI_API_KEY=sk-... python eval/benchmark_confidence.py  # needs API key
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v              # 58 tests
ruff check src/ tests/        # linter
mypy src/llm_guard/           # type checker
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT
