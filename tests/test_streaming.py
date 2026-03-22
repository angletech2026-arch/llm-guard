from __future__ import annotations

import json

from tests.conftest import make_streaming_chunks


def test_make_streaming_chunks():
    chunks = make_streaming_chunks("Hi")
    assert len(chunks) == 4  # 'H', 'i', finish, [DONE]
    assert chunks[-1] == "data: [DONE]"

    first = json.loads(chunks[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["content"] == "H"

    second = json.loads(chunks[1].removeprefix("data: "))
    assert second["choices"][0]["delta"]["content"] == "i"
