"""Pure OpenAI <-> MobileMoE mapping.

Every decision about how an OpenAI request becomes a prompt and sampling
parameters, and how a finished generation becomes a response body, lives here
as a function with no I/O. That is what lets the whole HTTP surface be tested
without loading a model.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
import time
from typing import Any

from .engine import GenerationParams, GenerationResult

#: Request fields the server accepts but cannot honour. Each maps to the
#: warning reported in ``x_mobilemoe.warnings``.
IGNORED_PARAMS: dict[str, str] = {
    "tools": "MobileMoE is not trained for tool calling; 'tools' were ignored",
    "functions": "MobileMoE is not trained for tool calling; 'functions' were ignored",
    "tool_choice": "'tool_choice' ignored: no tool calling",
    "function_call": "'function_call' ignored: no tool calling",
    "logprobs": "'logprobs' ignored: token log-probabilities are not exposed",
    "top_logprobs": "'top_logprobs' ignored: token log-probabilities are not exposed",
    "logit_bias": "'logit_bias' ignored: per-token biasing is not supported",
    "presence_penalty": (
        "'presence_penalty' ignored: use the non-standard 'repetition_penalty' instead"
    ),
    "frequency_penalty": (
        "'frequency_penalty' ignored: use the non-standard 'repetition_penalty' instead"
    ),
    "echo": "'echo' ignored: the prompt is never echoed back",
    "suffix": "'suffix' ignored: fill-in-the-middle is not supported",
    "best_of": "'best_of' ignored: choices are not re-ranked",
    "parallel_tool_calls": "'parallel_tool_calls' ignored: no tool calling",
}

_ROLE_ALIASES = {"developer": "system"}
_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function"}


class TranslationError(Exception):
    """A request the server understands but cannot serve. Becomes a 400."""

    def __init__(self, message: str, param: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(12)}"


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def _content_to_text(content: Any, index: int) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                raise TranslationError(
                    f"messages[{index}].content contains an unsupported part",
                    param=f"messages[{index}].content",
                )
            kind = part.get("type")
            if kind in ("text", "input_text"):
                parts.append(str(part.get("text", "")))
            else:
                raise TranslationError(
                    f"messages[{index}].content part type {kind!r} is not supported; "
                    "MobileMoE is text-only",
                    param=f"messages[{index}].content",
                    code="unsupported_content",
                )
        return "".join(parts)
    raise TranslationError(
        f"messages[{index}].content must be a string or a list of text parts",
        param=f"messages[{index}].content",
    )


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce OpenAI messages to ``{"role", "content"}`` dicts the chat template accepts.

    Tool-call turns from an earlier (different) model are kept in the history as
    plain text so a transcript that mixes models still renders coherently.
    """
    if not messages:
        raise TranslationError("'messages' must contain at least one message", param="messages")
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        role = _ROLE_ALIASES.get(role, role)
        if role not in _ALLOWED_ROLES:
            raise TranslationError(
                f"messages[{index}].role {role!r} is not one of "
                "system, developer, user, assistant, tool",
                param=f"messages[{index}].role",
            )
        text = _content_to_text(message.get("content"), index)
        tool_calls_only = False
        if role == "assistant" and message.get("tool_calls"):
            calls = json.dumps(message["tool_calls"], separators=(",", ":"))
            tool_calls_only = not text
            text = f"{text}\n{calls}".strip() if text else calls
        if role == "function":
            role = "tool"
        entry: dict[str, Any] = {"role": role, "content": text}
        if tool_calls_only:
            entry["_tool_calls_only"] = True
        if role == "tool" and message.get("name"):
            entry["name"] = str(message["name"])
        out.append(entry)
    if not any(m["role"] in ("user", "tool") for m in out):
        raise TranslationError(
            "'messages' needs at least one user message for the model to answer",
            param="messages",
        )
    return out


def response_format_hint(response_format: Any) -> str | None:
    """Turn ``response_format`` into a system-prompt instruction, or ``None``.

    MobileMoE has no grammar-constrained decoding, so JSON modes are guided by
    prompt only. The caller reports this as a warning.
    """
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return "Respond with a single valid JSON object and nothing else."
    if kind == "json_schema":
        spec = response_format.get("json_schema") or {}
        schema = spec.get("schema") if isinstance(spec, dict) else None
        if schema is None:
            raise TranslationError(
                "response_format.json_schema.schema is required", param="response_format"
            )
        return (
            "Respond with a single valid JSON object and nothing else. It must conform "
            f"to this JSON Schema:\n{json.dumps(schema, separators=(',', ':'))}"
        )
    if kind in (None, "text"):
        return None
    raise TranslationError(
        f"response_format.type {kind!r} is not supported", param="response_format"
    )


def with_system_hint(messages: list[dict[str, Any]], hint: str | None) -> list[dict[str, Any]]:
    """Append ``hint`` to the first system message, or prepend one."""
    if not hint:
        return messages
    out = [dict(m) for m in messages]
    for m in out:
        if m["role"] == "system":
            m["content"] = f"{m['content']}\n\n{hint}".strip()
            return out
    return [{"role": "system", "content": hint}, *out]


def tool_history_as_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render tool exchanges as plain context for a model with no tool training.

    Assistant turns that only carried ``tool_calls`` (already serialized to JSON
    by :func:`normalize_messages`) are dropped: shown that JSON as its own last
    turn, MobileMoE imitates the format instead of answering. Each ``tool``
    result becomes a user turn labelled with the tool name, so the retrieved
    material reads as something the user supplied.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("_tool_calls_only"):
            continue
        if m["role"] == "tool":
            name = m.get("name")
            label = f"Tool result ({name})" if name else "Tool result"
            out.append({"role": "user", "content": f"{label}:\n{m['content']}"})
            continue
        out.append({"role": m["role"], "content": m["content"]})
    return out


def strip_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the private ``_tool_calls_only`` marker before a template sees the messages."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def tool_messages_to_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fallback for templates that reject the ``tool`` role."""
    out = []
    for m in messages:
        if m["role"] == "tool":
            name = m.get("name")
            label = f"Tool result ({name})" if name else "Tool result"
            out.append({"role": "user", "content": f"{label}:\n{m['content']}"})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def collect_warnings(payload: dict[str, Any]) -> list[str]:
    """Name the request fields that were accepted but had no effect."""
    warnings: list[str] = []
    for key, message in IGNORED_PARAMS.items():
        value = payload.get(key)
        if value in (None, False, [], {}, 0, 0.0, "", "none"):
            continue
        warnings.append(message)
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    ):
        warnings.append(
            "response_format is prompt-guided only: MobileMoE has no constrained "
            "decoding, so validate the JSON you get back"
        )
    return warnings


def _number(payload: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranslationError(f"'{key}' must be a number", param=key)
    if not lo <= value <= hi:
        raise TranslationError(f"'{key}' must be between {lo:g} and {hi:g}", param=key)
    return float(value)


def sampling_params(
    payload: dict[str, Any],
    *,
    default_max_new_tokens: int,
    default_temperature: float,
    default_repetition_penalty: float = 1.0,
) -> GenerationParams:
    """Read OpenAI sampling fields (plus a few common extensions) into engine params."""
    max_tokens = payload.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        max_tokens = default_max_new_tokens
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise TranslationError("'max_tokens' must be a positive integer", param="max_tokens")

    temperature = _number(payload, "temperature", default_temperature, 0.0, 2.0)
    top_p = _number(payload, "top_p", 1.0, 0.0, 1.0)
    top_k = payload.get("top_k", 0)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise TranslationError("'top_k' must be a non-negative integer", param="top_k")
    repetition_penalty = _number(
        payload, "repetition_penalty", default_repetition_penalty, 0.0, 10.0
    )

    stop_raw = payload.get("stop")
    if stop_raw is None:
        stop: list[str] = []
    elif isinstance(stop_raw, str):
        stop = [stop_raw]
    elif isinstance(stop_raw, list) and all(isinstance(s, str) for s in stop_raw):
        stop = list(stop_raw)
    else:
        raise TranslationError("'stop' must be a string or a list of strings", param="stop")

    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TranslationError("'seed' must be an integer", param="seed")

    return GenerationParams(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop=stop,
        seed=seed,
    )


def choice_count(payload: dict[str, Any]) -> int:
    n = payload.get("n")
    if n is None:
        return 1
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise TranslationError("'n' must be a positive integer", param="n")
    if n > 8:
        raise TranslationError("'n' larger than 8 is not allowed", param="n")
    return n


def choice_params(params: GenerationParams, index: int) -> GenerationParams:
    """Parameters for the ``index``-th of ``n`` choices.

    A fixed ``seed`` would otherwise make every choice identical; offsetting it
    per choice keeps the request reproducible while still giving ``n``
    different samples.
    """
    if index == 0 or params.seed is None:
        return params
    return dataclasses.replace(params, seed=params.seed + index)


def check_budget(prompt_tokens: int, max_context: int, requested: int | None) -> None:
    """Mirror OpenAI's context check for an explicit ``max_tokens``."""
    if prompt_tokens >= max_context:
        raise TranslationError(
            f"This model's maximum context length is {max_context} tokens. However, "
            f"your messages resulted in {prompt_tokens} tokens. Please reduce the "
            "length of the messages.",
            param="messages",
            code="context_length_exceeded",
        )
    if requested is not None and prompt_tokens + requested > max_context:
        raise TranslationError(
            f"This model's maximum context length is {max_context} tokens. However, "
            f"you requested {prompt_tokens + requested} tokens ({prompt_tokens} in the "
            f"messages, {requested} in the completion). Please reduce the length of "
            "the messages or completion.",
            param="max_tokens",
            code="context_length_exceeded",
        )


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------


def usage_of(results: list[GenerationResult]) -> dict[str, int]:
    prompt = results[0].prompt_tokens if results else 0
    completion = sum(r.completion_tokens for r in results)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def extras(results: list[GenerationResult], warnings: list[str] | None = None) -> dict[str, Any]:
    compute = sum(r.compute_seconds for r in results)
    generated = sum(r.completion_tokens for r in results)
    block: dict[str, Any] = {
        "compute_seconds": round(compute, 4),
        "queue_wait_seconds": round(max((r.queue_wait_seconds for r in results), default=0), 4),
        "tokens_per_second": round(generated / compute, 2) if compute > 0 else 0.0,
    }
    if warnings:
        block["warnings"] = list(warnings)
    return block


def build_chat_completion(
    results: list[GenerationResult],
    model: str,
    *,
    fingerprint: str,
    include_extras: bool = True,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": r.text, "refusal": None},
                "logprobs": None,
                "finish_reason": r.finish_reason,
            }
            for i, r in enumerate(results)
        ],
        "usage": usage_of(results),
    }
    if include_extras:
        body["x_mobilemoe"] = extras(results, warnings)
    return body


def build_text_completion(
    results: list[GenerationResult],
    model: str,
    *,
    fingerprint: str,
    include_extras: bool = True,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": new_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [
            {"index": i, "text": r.text, "logprobs": None, "finish_reason": r.finish_reason}
            for i, r in enumerate(results)
        ],
        "usage": usage_of(results),
    }
    if include_extras:
        body["x_mobilemoe"] = extras(results, warnings)
    return body
