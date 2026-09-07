"""Server-sent-event rendering for ``stream: true``.

Unlike a synthesized stream, every content chunk here corresponds to text the
model has actually produced so far: the engine pushes decoded deltas into an
``asyncio.Queue`` from the generation thread and the route drains it.
"""

from __future__ import annotations

import json
from typing import Any


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


DONE = "data: [DONE]\n\n"


def chat_chunk(
    completion_id: str,
    created: int,
    model: str,
    index: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    *,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": index, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        ],
    }
    if fingerprint:
        body["system_fingerprint"] = fingerprint
    return body


def text_chunk(
    completion_id: str,
    created: int,
    model: str,
    index: int,
    text: str,
    finish_reason: str | None = None,
    *,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": index, "text": text, "logprobs": None, "finish_reason": finish_reason}
        ],
    }
    if fingerprint:
        body["system_fingerprint"] = fingerprint
    return body


def usage_chunk(
    completion_id: str,
    created: int,
    model: str,
    object_type: str,
    usage: dict[str, int],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": completion_id,
        "object": object_type,
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    }
    if extras is not None:
        body["x_mobilemoe"] = extras
    return body
