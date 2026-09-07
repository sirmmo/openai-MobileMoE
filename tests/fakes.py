"""A stand-in for the torch engine, so the HTTP surface is testable offline.

Kept out of ``conftest.py`` on purpose: this module imports ``openai_mobilemoe``,
and the live suite runs against a container with only ``pytest`` and ``httpx``
installed.

The fake "tokenizes" on whitespace, so token counts in tests are word counts,
and it renders a tiny chat template of its own so the prompt the server built
can be inspected as text.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from openai_mobilemoe.engine import (
    EngineError,
    EngineOverloaded,
    GenerationParams,
    GenerationResult,
)

BOS = "<s>"


class FakeEngine:
    """Stands in for :class:`MobileMoEEngine`, recording what it was asked to do."""

    def __init__(self, response: str = "Hello from MobileMoE, how can I help?") -> None:
        self.ready = True
        self.model_name = "facebook/MobileMoE-S-QAT"
        self.device = "cpu"
        self.dtype = "float32"
        self.max_context = 64
        self.has_chat_template = True
        self.queue_depth = 0
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.overloaded = False
        self.raise_on_generate: Exception | None = None
        self.rejects_tool_role = False
        self.stream_delay = 0.0
        self._vocab: dict[str, int] = {}
        self._words: list[str] = []

    # -- prompt handling ---------------------------------------------------

    def render_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if not self.has_chat_template:
            raise EngineError("no chat template")
        parts = [BOS]
        for m in messages:
            if m["role"] == "tool" and self.rejects_tool_role:
                raise ValueError("Unknown role: tool")
            parts.append(f"<|{m['role']}|>{m['content']}<|end|>")
        parts.append("<|assistant|>")
        return "".join(parts)

    def encode(self, text: str, *, rendered: bool = False) -> list[int]:
        words = text.split()
        if not rendered or not text.startswith(BOS):
            words = [BOS, *words]
        return [self._id(w) for w in words]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._words[i] if i < len(self._words) else f"<{i}>" for i in ids)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._words)
            self._words.append(word)
        return self._vocab[word]

    # -- generation --------------------------------------------------------

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        if self.overloaded:
            raise EngineOverloaded("engine queue is full (16 requests waiting); retry shortly")
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # surfaced to the caller via the future
            future.set_exception(exc)
        return future

    def generate(
        self,
        prompt_ids: list[int],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> GenerationResult:
        self.calls.append(
            {"prompt": self.decode(prompt_ids), "prompt_ids": list(prompt_ids), "params": params}
        )
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        text = self.response
        finish = "stop"
        for stop in params.stop:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
        words = text.split(" ")
        if len(words) > params.max_new_tokens:
            words = words[: params.max_new_tokens]
            finish = "length"
        emitted = ""
        for i, word in enumerate(words):
            piece = word if i == 0 else f" {word}"
            emitted += piece
            if on_text is not None:
                on_text(piece)
            if self.stream_delay:
                time.sleep(self.stream_delay)
        return GenerationResult(
            text=emitted,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(words),
            finish_reason=finish,
            compute_seconds=0.01,
        )

    def shutdown(self) -> None:
        pass
