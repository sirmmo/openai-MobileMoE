"""Unit tests for the pure mapping layer and the incremental decoder."""

from __future__ import annotations

import pytest

from openai_mobilemoe import translate
from openai_mobilemoe.engine import GenerationResult, IncrementalDecoder
from openai_mobilemoe.translate import TranslationError

# -- messages --------------------------------------------------------------


def test_normalize_keeps_roles_and_text():
    out = translate.normalize_messages(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "how "}, {"type": "text", "text": "are you"}],
            },
        ]
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[-1]["content"] == "how are you"


def test_developer_role_becomes_system():
    out = translate.normalize_messages(
        [{"role": "developer", "content": "x"}, {"role": "user", "content": "y"}]
    )
    assert out[0]["role"] == "system"


def test_assistant_tool_calls_are_kept_as_text():
    out = translate.normalize_messages(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny", "name": "get_weather"},
        ]
    )
    assert "get_weather" in out[1]["content"]
    assert out[2] == {"role": "tool", "content": "sunny", "name": "get_weather"}


def test_tool_history_as_context_drops_call_turns_and_relabels_results():
    msgs = translate.normalize_messages(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "w"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny", "name": "w"},
            {"role": "assistant", "content": "It is sunny.", "tool_calls": [{"id": "c2"}]},
        ]
    )
    out = translate.tool_history_as_context(msgs)
    assert [m["role"] for m in out] == ["user", "user", "assistant"]
    assert out[1]["content"] == "Tool result (w):\nsunny"
    assert out[2]["content"].startswith("It is sunny.")  # content-bearing turns survive
    assert all(not k.startswith("_") for m in out for k in m)
    assert all(not k.startswith("_") for m in translate.strip_markers(msgs) for k in m)


def test_image_parts_are_rejected():
    with pytest.raises(TranslationError) as exc:
        translate.normalize_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        )
    assert exc.value.code == "unsupported_content"


def test_unknown_role_is_rejected():
    with pytest.raises(TranslationError):
        translate.normalize_messages([{"role": "narrator", "content": "x"}])


def test_system_only_conversation_is_rejected():
    with pytest.raises(TranslationError):
        translate.normalize_messages([{"role": "system", "content": "x"}])


def test_empty_messages_are_rejected():
    with pytest.raises(TranslationError):
        translate.normalize_messages([])


def test_tool_messages_fold_into_user_turns():
    out = translate.tool_messages_to_user(
        [{"role": "tool", "content": "42", "name": "calc"}, {"role": "user", "content": "ok"}]
    )
    assert out[0]["role"] == "user"
    assert "Tool result (calc)" in out[0]["content"]


# -- response_format -------------------------------------------------------


def test_json_object_hint_is_appended_to_system():
    hint = translate.response_format_hint({"type": "json_object"})
    out = translate.with_system_hint(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "x"}], hint
    )
    assert out[0]["content"].startswith("be brief")
    assert "JSON" in out[0]["content"]
    assert len(out) == 2


def test_json_schema_hint_prepends_system_when_missing():
    hint = translate.response_format_hint(
        {"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object"}}}
    )
    out = translate.with_system_hint([{"role": "user", "content": "x"}], hint)
    assert out[0]["role"] == "system"
    assert '"type":"object"' in out[0]["content"]


def test_text_response_format_is_a_noop():
    assert translate.response_format_hint({"type": "text"}) is None
    assert translate.response_format_hint(None) is None


def test_unknown_response_format_is_rejected():
    with pytest.raises(TranslationError):
        translate.response_format_hint({"type": "xml"})


# -- parameters ------------------------------------------------------------


def test_warnings_name_ignored_fields():
    warnings = translate.collect_warnings(
        {"tools": [{"type": "function"}], "logprobs": True, "temperature": 0.5}
    )
    assert any("tools" in w for w in warnings)
    assert any("logprobs" in w for w in warnings)
    assert not any("temperature" in w for w in warnings)


def test_json_mode_is_flagged_as_prompt_guided():
    warnings = translate.collect_warnings({"response_format": {"type": "json_object"}})
    assert any("prompt-guided" in w for w in warnings)


def test_sampling_defaults_are_greedy():
    params = translate.sampling_params({}, default_max_new_tokens=100, default_temperature=0.0)
    assert params.max_new_tokens == 100
    assert params.temperature == 0.0
    assert params.do_sample is False
    assert params.stop == []


def test_sampling_reads_openai_fields():
    params = translate.sampling_params(
        {
            "max_completion_tokens": 7,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 3,
            "stop": "END",
            "repetition_penalty": 1.1,
        },
        default_max_new_tokens=100,
        default_temperature=0.0,
    )
    assert params.max_new_tokens == 7
    assert params.do_sample is True
    assert (params.top_p, params.top_k, params.seed) == (0.9, 40, 3)
    assert params.stop == ["END"]
    assert params.repetition_penalty == 1.1


@pytest.mark.parametrize(
    "payload",
    [
        {"max_tokens": 0},
        {"max_tokens": "10"},
        {"temperature": 3},
        {"top_p": -0.1},
        {"top_k": -1},
        {"stop": [1, 2]},
        {"seed": "x"},
    ],
)
def test_invalid_sampling_values_are_rejected(payload):
    with pytest.raises(TranslationError):
        translate.sampling_params(payload, default_max_new_tokens=10, default_temperature=0.0)


def test_choice_count():
    assert translate.choice_count({}) == 1
    assert translate.choice_count({"n": 3}) == 3
    with pytest.raises(TranslationError):
        translate.choice_count({"n": 0})
    with pytest.raises(TranslationError):
        translate.choice_count({"n": 99})


def test_default_repetition_penalty_applies_only_when_omitted():
    kw = {"default_max_new_tokens": 10, "default_temperature": 0.0}
    assert translate.sampling_params({}, **kw).repetition_penalty == 1.0
    assert (
        translate.sampling_params({}, default_repetition_penalty=1.15, **kw).repetition_penalty
        == 1.15
    )
    assert (
        translate.sampling_params(
            {"repetition_penalty": 1.0}, default_repetition_penalty=1.15, **kw
        ).repetition_penalty
        == 1.0
    )


def test_choice_params_offset_the_seed():
    base = translate.sampling_params({"seed": 5}, default_max_new_tokens=1, default_temperature=1)
    assert translate.choice_params(base, 0) is base
    assert translate.choice_params(base, 2).seed == 7
    unseeded = translate.sampling_params({}, default_max_new_tokens=1, default_temperature=1)
    assert translate.choice_params(unseeded, 3) is unseeded


def test_budget_check_mirrors_openai():
    translate.check_budget(10, 100, None)
    translate.check_budget(10, 100, 90)
    with pytest.raises(TranslationError) as exc:
        translate.check_budget(10, 100, 91)
    assert exc.value.code == "context_length_exceeded"
    with pytest.raises(TranslationError):
        translate.check_budget(100, 100, None)


# -- responses -------------------------------------------------------------


def _result(text="hi", prompt=3, completion=2, finish="stop"):
    return GenerationResult(
        text=text,
        prompt_tokens=prompt,
        completion_tokens=completion,
        finish_reason=finish,
        compute_seconds=0.5,
    )


def test_chat_completion_shape():
    body = translate.build_chat_completion(
        [_result(), _result(text="yo")], "m", fingerprint="fp", warnings=["w"]
    )
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert [c["index"] for c in body["choices"]] == [0, 1]
    assert body["choices"][1]["message"] == {"role": "assistant", "content": "yo", "refusal": None}
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert body["x_mobilemoe"]["warnings"] == ["w"]
    assert body["x_mobilemoe"]["tokens_per_second"] == 4.0


def test_text_completion_shape():
    body = translate.build_text_completion([_result()], "m", fingerprint="fp", include_extras=False)
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hi"
    assert "x_mobilemoe" not in body


# -- incremental decoder ---------------------------------------------------


class CharTokenizer:
    """Each id is one string; ``overrides`` emulate multi-byte partial decodes."""

    def __init__(self, vocab, overrides=None):
        self.vocab = vocab
        self.overrides = overrides or {}

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        self.last_kwargs = kwargs
        key = tuple(ids)
        if key in self.overrides:
            return self.overrides[key]
        return "".join(self.vocab[i] for i in ids)


def _run(vocab, ids, stop=None, overrides=None):
    pieces = []
    dec = IncrementalDecoder(CharTokenizer(vocab, overrides), stop, pieces.append)
    for i in ids:
        dec.push([i])
    dec.finish()
    return dec, pieces


def test_decoder_streams_every_token():
    dec, pieces = _run(["a", "b", "c"], [0, 1, 2])
    assert pieces == ["a", "b", "c"]
    assert dec.text == "abc"


def test_decoder_holds_back_partial_stop_string_then_releases():
    dec, pieces = _run(["a", "E", "N", "x", "D"], [0, 1, 2, 3])
    assert dec.text == "aENx"
    dec2, _ = _run(["a", "E", "N", "x", "D"], [0, 1, 2, 3], stop=["END"])
    assert dec2.text == "aENx"
    assert dec2.stop_hit is False


def test_decoder_cuts_at_stop_string():
    dec, pieces = _run(["a", "E", "N", "D", "z"], [0, 1, 2, 3, 4], stop=["END"])
    assert dec.text == "a"
    assert dec.stop_hit is True
    assert "".join(pieces) == "a"


def test_decoder_disables_tokenizer_space_cleanup():
    # Llama-3 style tokenizers default to clean_up_tokenization_spaces=True, which
    # rewrites "SELECT ?s" as "SELECT?s". The decoder must opt out explicitly.
    tokenizer = CharTokenizer(["a", "b"])
    dec = IncrementalDecoder(tokenizer, None, None)
    dec.push([0])
    assert tokenizer.last_kwargs == {"clean_up_tokenization_spaces": False}


def test_decoder_waits_for_complete_multibyte_char():
    # id 1 alone decodes to a replacement char; with id 2 it becomes "é".
    vocab = ["a", "�", "", "b"]
    dec, pieces = _run(
        vocab, [0, 1, 2, 3], overrides={(0, 1, 2): "aé", (1, 2): "é", (1, 2, 3): "éb"}
    )
    assert dec.text == "aéb"
    assert "�" not in "".join(pieces)
