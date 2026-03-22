from llm_guard.utils.consistency import compute_consistency


def test_identical():
    result = compute_consistency("the sky is blue", ["the sky is blue", "the sky is blue"])
    assert result["score"] == 1.0
    assert result["divergent"] == 0


def test_completely_different():
    result = compute_consistency(
        "the sky is blue",
        ["bananas are yellow tropical fruit", "cats sleep eighteen hours daily"],
    )
    assert result["score"] < 0.3
    assert result["divergent"] == 2


def test_partial_overlap():
    result = compute_consistency(
        "the capital of France is Paris",
        ["Paris is the capital of France"],
    )
    assert result["score"] > 0.5


def test_empty_samples():
    result = compute_consistency("hello", [])
    assert result["score"] == 0.0


def test_empty_original():
    result = compute_consistency("", ["hello world"])
    assert result["divergent"] == 1
