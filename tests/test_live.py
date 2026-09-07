"""End-to-end tests against a running server.

    MOBILEMOE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live

Needs only ``pytest`` and ``httpx``; the ``openai`` SDK test is skipped when the
package is not installed. Assertions are about shape and behaviour, not the
exact words a model produces.
"""

from __future__ import annotations

import json
import os

import pytest

BASE_URL = os.environ.get("MOBILEMOE_TEST_BASE_URL")
if not BASE_URL:
    pytest.skip("set MOBILEMOE_TEST_BASE_URL to run the live suite", allow_module_level=True)

import httpx

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=BASE_URL, timeout=300.0) as client:
        yield client


def _events(text: str) -> list:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            out.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return out


def test_health(http):
    body = http.get("/health").json()
    assert body["status"] == "ok"
    assert body["context_length"] > 0


def test_models(http):
    body = http.get("/v1/models").json()
    assert body["data"][0]["object"] == "model"


def test_chat_completion(http):
    response = http.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Reply with one short sentence: hello!"}],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip()
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    assert body["usage"]["completion_tokens"] >= 1
    assert body["usage"]["completion_tokens"] <= 32


def test_chat_stream_reassembles(http):
    with http.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Count from one to five in words."}],
            "max_tokens": 24,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        events = _events(response.read().decode())
    assert events[-1] == "[DONE]"
    chunks = [e for e in events[:-1] if e["choices"]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks).strip()
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    assert events[-2]["usage"]["completion_tokens"] >= 1


def test_stop_string_is_honoured(http):
    body = http.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Write the alphabet as letters separated by spaces."}
            ],
            "max_tokens": 40,
            "stop": ["E"],
        },
    ).json()
    assert "E" not in body["choices"][0]["message"]["content"]


def test_greedy_is_deterministic(http):
    payload = {
        "messages": [{"role": "user", "content": "Name three colours."}],
        "max_tokens": 16,
        "temperature": 0,
    }
    first = http.post("/v1/chat/completions", json=payload).json()
    second = http.post("/v1/chat/completions", json=payload).json()
    assert first["choices"][0]["message"]["content"] == second["choices"][0]["message"]["content"]


def test_completions_endpoint(http):
    body = http.post(
        "/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 8}
    ).json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"]


def test_context_overflow_is_a_400(http):
    body = http.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10_000_000},
    )
    assert body.status_code == 400
    assert body.json()["error"]["code"] == "context_length_exceeded"


def test_openai_sdk_round_trip():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=16,
    )
    assert response.choices[0].message.content
    pieces = []
    for chunk in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=16,
        stream=True,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            pieces.append(chunk.choices[0].delta.content)
    assert "".join(pieces)
