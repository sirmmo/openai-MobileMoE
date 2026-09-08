"""HTTP-level tests, driven through a fake engine."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from openai_mobilemoe.config import Settings
from openai_mobilemoe.engine import EngineError, PromptTooLong
from openai_mobilemoe.server import create_app


def chat(client, **overrides):
    payload = {
        "model": "MobileMoE-S-QAT",
        "messages": [{"role": "user", "content": "hello there"}],
    }
    payload.update(overrides)
    return client.post("/v1/chat/completions", json=payload)


def _events(text: str) -> list:
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        out.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return out


# -- discovery -------------------------------------------------------------


def test_list_models(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    card = body["data"][0]
    assert card["id"] == "MobileMoE-S-QAT"
    assert card["owned_by"] == "facebook"
    assert card["x_mobilemoe"]["repo"] == "facebook/MobileMoE-S-QAT"
    assert card["x_mobilemoe"]["context_length"] == 64


def test_get_model_echoes_requested_id(client):
    assert client.get("/v1/models/mobilemoe").json()["id"] == "mobilemoe"


def test_health_reports_engine_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["queue_depth"] == 0
    assert body["device"] == "cpu"


def test_root_lists_endpoints(client):
    assert "/v1/chat/completions" in client.get("/").json()["endpoints"]


# -- chat completions ------------------------------------------------------


def test_chat_round_trip(client):
    response = chat(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "MobileMoE-S-QAT"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello from MobileMoE, how can I help?"
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert "tokens_per_second" in body["x_mobilemoe"]


def test_prompt_is_rendered_through_the_chat_template(client):
    chat(
        client,
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
    )
    prompt = client.engine.calls[-1]["prompt"]
    assert prompt.startswith("<s>")
    assert "<|system|>be terse<|end|>" in prompt
    assert prompt.endswith("<|assistant|>")
    # BOS came from the template and must not be doubled.
    assert client.engine.calls[-1]["prompt_ids"].count(0) == 1


def test_max_tokens_truncates_and_reports_length(client):
    body = chat(client, max_tokens=3).json()
    assert body["choices"][0]["message"]["content"] == "Hello from MobileMoE,"
    assert body["choices"][0]["finish_reason"] == "length"
    assert client.engine.calls[-1]["params"].max_new_tokens == 3


def test_default_max_tokens_is_clamped_to_the_context(client):
    chat(client)
    params = client.engine.calls[-1]["params"]
    prompt_tokens = len(client.engine.calls[-1]["prompt_ids"])
    assert params.max_new_tokens == 64 - prompt_tokens


def test_stop_strings_cut_the_output(client):
    body = chat(client, stop=["MobileMoE"]).json()
    assert body["choices"][0]["message"]["content"] == "Hello from "
    assert client.engine.calls[-1]["params"].stop == ["MobileMoE"]


def test_sampling_parameters_are_forwarded(client):
    chat(client, temperature=0.7, top_p=0.5, seed=42, top_k=20)
    params = client.engine.calls[-1]["params"]
    assert (params.temperature, params.top_p, params.seed, params.top_k) == (0.7, 0.5, 42, 20)
    assert params.do_sample is True


def test_omitted_temperature_means_greedy(client):
    chat(client)
    assert client.engine.calls[-1]["params"].do_sample is False


def test_ignored_parameters_are_reported_not_rejected(client):
    body = chat(client, tools=[{"type": "function", "function": {"name": "f"}}], logprobs=True)
    assert body.status_code == 200
    warnings = body.json()["x_mobilemoe"]["warnings"]
    assert any("tools" in w for w in warnings)
    assert any("logprobs" in w for w in warnings)


def test_json_mode_hints_the_prompt_and_warns(client):
    body = chat(client, response_format={"type": "json_object"}).json()
    assert "JSON" in client.engine.calls[-1]["prompt"]
    assert any("prompt-guided" in w for w in body["x_mobilemoe"]["warnings"])


def test_n_produces_multiple_choices(client):
    body = chat(client, n=2).json()
    assert [c["index"] for c in body["choices"]] == [0, 1]
    assert body["usage"]["completion_tokens"] == 14
    assert len(client.engine.calls) == 2


def test_n_with_seed_varies_the_seed_per_choice(client):
    chat(client, n=3, seed=10, temperature=1.0)
    assert [c["params"].seed for c in client.engine.calls[-3:]] == [10, 11, 12]


def test_tool_role_falls_back_when_the_template_rejects_it(client):
    client.engine.rejects_tool_role = True
    response = chat(
        client,
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "tool", "content": "sunny", "name": "get_weather"},
        ],
    )
    assert response.status_code == 200, response.text
    assert "Tool result (get_weather)" in client.engine.calls[-1]["prompt"]


def test_server_default_repetition_penalty(engine):
    app = create_app(settings=Settings(default_repetition_penalty=1.15), engine=engine)
    with TestClient(app) as client:
        chat(client)
        assert engine.calls[-1]["params"].repetition_penalty == 1.15
        chat(client, repetition_penalty=1.0)
        assert engine.calls[-1]["params"].repetition_penalty == 1.0


def test_extras_can_be_disabled(engine):
    app = create_app(settings=Settings(expose_extras=False), engine=engine)
    with TestClient(app) as client:
        assert "x_mobilemoe" not in chat(client).json()


# -- streaming -------------------------------------------------------------


def test_streaming_emits_a_valid_chunk_sequence(client):
    response = chat(client, stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == "Hello from MobileMoE, how can I help?"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert "x_mobilemoe" in chunks[-1]
    # Content arrives as the model produces it: more than one content chunk.
    assert sum(1 for c in chunks if c["choices"][0]["delta"].get("content")) > 1


def test_streaming_include_usage_appends_a_usage_chunk(client):
    events = _events(chat(client, stream=True, stream_options={"include_usage": True}).text)
    usage_chunk = events[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] == 7


def test_streaming_n_interleaves_choice_indices(client):
    events = _events(chat(client, stream=True, n=2).text)[:-1]
    indices = {c["choices"][0]["index"] for c in events}
    assert indices == {0, 1}
    finals = [c for c in events if c["choices"][0]["finish_reason"]]
    assert len(finals) == 2


def test_streaming_engine_error_is_sent_as_an_error_event(client):
    client.engine.raise_on_generate = EngineError("boom")
    events = _events(chat(client, stream=True).text)
    assert any(isinstance(e, dict) and "error" in e for e in events)


# -- legacy completions ----------------------------------------------------


def test_completions_round_trip(client):
    response = client.post(
        "/v1/completions", json={"model": "MobileMoE-S-QAT", "prompt": "Once upon a"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"].startswith("Hello")
    assert client.engine.calls[-1]["prompt"] == "<s> Once upon a"


def test_completions_accept_a_list_of_prompts(client):
    body = client.post("/v1/completions", json={"prompt": ["a", "b"], "n": 2}).json()
    assert [c["index"] for c in body["choices"]] == [0, 1, 2, 3]


def test_completions_accept_token_arrays(client):
    body = client.post("/v1/completions", json={"prompt": [0, 1, 2]}).json()
    assert body["choices"][0]["text"]
    assert client.engine.calls[-1]["prompt_ids"] == [0, 1, 2]


def test_completions_stream(client):
    events = _events(client.post("/v1/completions", json={"prompt": "x", "stream": True}).text)
    assert events[-1] == "[DONE]"
    text = "".join(e["choices"][0]["text"] for e in events[:-1])
    assert text == "Hello from MobileMoE, how can I help?"
    assert events[-2]["choices"][0]["finish_reason"] == "stop"


def test_completions_reject_empty_prompt(client):
    response = client.post("/v1/completions", json={"prompt": ""})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "prompt"


def test_base_model_without_template_rejects_chat_but_serves_completions(client):
    client.engine.has_chat_template = False
    response = chat(client)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_chat_template"
    assert client.post("/v1/completions", json={"prompt": "x"}).status_code == 200


# -- errors ----------------------------------------------------------------


def test_empty_messages_is_a_clear_400(client):
    response = chat(client, messages=[])
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages"


def test_image_content_is_a_400(client):
    response = chat(
        client,
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_content"


def test_invalid_json_body_is_a_400(client):
    response = client.post(
        "/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_pydantic_validation_error_is_a_400(client):
    response = client.post("/v1/chat/completions", json={"messages": "not a list"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages"


def test_explicit_max_tokens_over_context_is_a_400(client):
    response = chat(client, max_tokens=1000)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_prompt_longer_than_context_is_a_400(client):
    response = chat(client, messages=[{"role": "user", "content": " ".join(["w"] * 100)}])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_engine_prompt_too_long_maps_to_400(client):
    client.engine.raise_on_generate = PromptTooLong(70, 64)
    response = chat(client)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_engine_overload_maps_to_429(client):
    client.engine.overloaded = True
    response = chat(client)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "engine_busy"


def test_engine_error_maps_to_503(client):
    client.engine.raise_on_generate = EngineError("cuda fell over")
    assert chat(client).status_code == 503


def test_engine_not_ready_maps_to_503(engine):
    engine.ready = False
    with TestClient(create_app(settings=Settings(), engine=engine)) as client:
        assert chat(client).status_code == 503
        assert client.get("/health").json()["status"] == "loading"


def test_api_key_is_enforced_when_configured(engine):
    app = create_app(settings=Settings(api_key="secret"), engine=engine)
    with TestClient(app) as client:
        assert chat(client).status_code == 401
        assert client.get("/health").status_code == 200
        ok = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret"},
        )
        assert ok.status_code == 200


def test_unknown_model_is_accepted_by_default(client):
    assert chat(client, model="gpt-4o-mini").status_code == 200


def test_strict_model_rejects_unknown_names(engine):
    app = create_app(settings=Settings(strict_model=True), engine=engine)
    with TestClient(app) as client:
        assert chat(client, model="gpt-4o-mini").status_code == 404
        assert chat(client, model="facebook/MobileMoE-S-QAT").status_code == 200
        assert chat(client, model="mobilemoe").status_code == 200
