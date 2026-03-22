# Contributing to LLM-Guard

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-guard.git
cd llm-guard
pip install -e ".[dev]"
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/llm_guard/ --ignore-missing-imports

# Run all checks
pytest tests/ -v && ruff check src/ tests/ && mypy src/llm_guard/ --ignore-missing-imports
```

## Adding a New Analyzer

1. Create `src/llm_guard/analyzers/your_analyzer.py`
2. Extend `BaseAnalyzer` from `analyzers/base.py`
3. Implement the `analyze()` method
4. Register it in `proxy.py` `ProxyHandler.startup()`
5. Add config in `config.py`
6. Write tests in `tests/test_your_analyzer.py`
7. Add eval benchmark in `eval/`

## Adding Conflict Patterns

Edit `INSTRUCTION_CONFLICT_PATTERNS` in `src/llm_guard/analyzers/conflict.py`.

Each pattern is a tuple: `(pattern_a_regex, pattern_b_regex, description)`.

Run `python eval/benchmark_conflict.py` to verify no regressions.

## Pull Requests

- All tests must pass
- Ruff and mypy must pass
- Include tests for new functionality
- Update eval benchmarks if applicable
