"""Model loading and token generation.

Everything that touches torch lives here. Generation runs on a single worker
thread: Transformers' ``generate`` is not safe to call concurrently on one
model, and on CPU a second concurrent request would only slow both down.
Requests queue in front of that thread and are rejected past
``Settings.max_queue_depth``.

Streaming is genuine. A :class:`CallbackStreamer` receives every token as it is
sampled, an :class:`IncrementalDecoder` turns the growing id list into text
deltas without splitting multi-byte characters, and stop strings are matched on
the decoded text while holding back any suffix that could still grow into one.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)

#: Tokens that end an assistant turn in the Llama-3 family and its relatives.
#: Only the ones that actually exist in the loaded tokenizer are used.
_TURN_END_TOKENS = ("<|eot|>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>")


class EngineError(RuntimeError):
    """Base class for failures the HTTP layer maps to 5xx responses."""


class EngineOverloaded(EngineError):
    """Raised by :meth:`MobileMoEEngine.submit` when the queue is full."""


class PromptTooLong(EngineError):
    """The prompt alone does not fit the context window."""

    def __init__(self, prompt_tokens: int, limit: int) -> None:
        super().__init__(
            f"This model's maximum context length is {limit} tokens. However, your "
            f"messages resulted in {prompt_tokens} tokens."
        )
        self.prompt_tokens = prompt_tokens
        self.limit = limit


@dataclass
class GenerationParams:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    #: ``stop`` when the model emitted an end-of-turn token or a stop string
    #: matched, ``length`` when ``max_new_tokens`` was reached.
    finish_reason: str
    compute_seconds: float
    queue_wait_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.compute_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.compute_seconds


class IncrementalDecoder:
    """Turn a growing list of token ids into text deltas.

    Decoding token by token would split multi-byte characters (a single emoji
    can span three BPE tokens). Instead the decoder re-decodes a short window
    of trailing ids each time and only releases text once it no longer ends in
    a replacement character. Stop strings are matched on the released text;
    any trailing text that is a prefix of a stop string is held back until it
    either completes the match or diverges.
    """

    def __init__(
        self,
        tokenizer: Any,
        stop: list[str] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._stops = [s for s in (stop or []) if s]
        self._on_text = on_text
        self._ids: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0
        self._pending = ""
        self.text = ""
        self.stop_hit = False

    def push(self, token_ids: list[int]) -> None:
        if self.stop_hit:
            return
        self._ids.extend(token_ids)
        prefix_text = self._decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = self._decode(self._ids[self._prefix_offset :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)
            self._absorb(new_text[len(prefix_text) :])

    def finish(self) -> None:
        """Release whatever is still buffered once generation has ended."""
        if not self.stop_hit:
            prefix_text = self._decode(self._ids[self._prefix_offset : self._read_offset])
            new_text = self._decode(self._ids[self._prefix_offset :]).rstrip("�")
            if len(new_text) > len(prefix_text):
                self._absorb(new_text[len(prefix_text) :])
        if not self.stop_hit and self._pending:
            self._emit(self._pending)
        self._pending = ""

    # -- internals -------------------------------------------------------

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        # clean_up_tokenization_spaces (on by default for the MobileMoE tokenizer)
        # deletes the space before ? . , ! and turns `SELECT ?s` into `SELECT?s`,
        # which silently corrupts code, SPARQL and anything punctuation-sensitive.
        return self._tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _absorb(self, delta: str) -> None:
        self._pending += delta
        if not self._stops:
            self._emit(self._pending)
            self._pending = ""
            return
        hits = [i for i in (self._pending.find(s) for s in self._stops) if i >= 0]
        if hits:
            self._emit(self._pending[: min(hits)])
            self._pending = ""
            self.stop_hit = True
            return
        hold = 0
        for stop in self._stops:
            for k in range(min(len(stop) - 1, len(self._pending)), 0, -1):
                if self._pending.endswith(stop[:k]):
                    hold = max(hold, k)
                    break
        release = len(self._pending) - hold
        if release > 0:
            self._emit(self._pending[:release])
            self._pending = self._pending[release:]

    def _emit(self, piece: str) -> None:
        if not piece:
            return
        self.text += piece
        if self._on_text is not None:
            self._on_text(piece)


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(name: str, device: str) -> Any:
    import torch

    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise EngineError(f"unknown dtype {name!r}; use one of {sorted(table)}") from exc


class MobileMoEEngine:
    """Owns the model, the tokenizer and the single generation thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.model
        self.ready = False
        self.device = "unloaded"
        self.dtype = "unloaded"
        self.max_context = settings.max_context or 0
        self.stop_token_ids: list[int] = []
        self.pad_token_id: int | None = None
        self.has_chat_template = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mobilemoe")
        self._pending = 0
        self._pending_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._pending

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def start(self) -> None:
        """Download (if needed), load and warm up the model. Blocking."""
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self.settings
        major = int(transformers.__version__.split(".")[0])
        if major >= 5 and s.trust_remote_code:
            log.warning(
                "transformers %s builds models on the meta device; MobileMoE's remote "
                "code fails there (Tensor.item() in the rotary embedding). Install "
                "'transformers>=4.57,<5'.",
                transformers.__version__,
            )
        if s.threads:
            torch.set_num_threads(s.threads)
        device = _resolve_device(s.device)
        dtype = _resolve_dtype(s.dtype, device)
        log.info("loading %s on %s as %s", s.model, device, str(dtype).replace("torch.", ""))

        self._tokenizer = AutoTokenizer.from_pretrained(
            s.model, trust_remote_code=s.trust_remote_code
        )
        kwargs: dict[str, Any] = {"trust_remote_code": s.trust_remote_code, "dtype": dtype}
        if s.attn_implementation:
            kwargs["attn_implementation"] = s.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(s.model, **kwargs)
        model.to(device)
        model.eval()
        self._model = model
        self.device = device
        self.dtype = str(dtype).replace("torch.", "")

        self.stop_token_ids = self._collect_stop_ids()
        pad = self._tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.stop_token_ids[0]
        self.max_context = s.max_context or self._detect_context()
        self.has_chat_template = bool(getattr(self._tokenizer, "chat_template", None))

        # The QAT checkpoints keep INT4 weights on disk and materialize them on
        # the first forward pass. Doing that here keeps it off the first request.
        bos = self._tokenizer.bos_token_id
        probe = bos if bos is not None else self.stop_token_ids[0]
        with torch.inference_mode():
            model(input_ids=torch.tensor([[probe]], device=device))

        self.ready = True
        log.info(
            "ready: context=%d stop_ids=%s chat_template=%s",
            self.max_context,
            self.stop_token_ids,
            self.has_chat_template,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Queue ``fn`` on the generation thread; reject when the queue is full."""
        with self._pending_lock:
            if self._pending >= self.settings.max_queue_depth:
                raise EngineOverloaded(
                    f"engine queue is full ({self._pending} requests waiting); retry shortly"
                )
            self._pending += 1

        def run() -> Any:
            try:
                return fn(*args, **kwargs)
            finally:
                with self._pending_lock:
                    self._pending -= 1

        return self._executor.submit(run)

    # -- prompt handling ---------------------------------------------------

    def render_chat(self, messages: list[dict[str, Any]], **template_kwargs: Any) -> str:
        """Apply the checkpoint's chat template with the generation prompt appended."""
        if not self.has_chat_template:
            raise EngineError(
                f"{self.model_name} has no chat template; it is a base model. Use "
                "/v1/completions, or serve an SFT / QAT checkpoint for chat."
            )
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )

    def encode(self, text: str, *, rendered: bool = False) -> list[int]:
        """Tokenize. ``rendered`` prompts came out of the chat template, which
        already spells out the BOS token, so it must not be added twice."""
        add_special = True
        bos = getattr(self._tokenizer, "bos_token", None)
        if rendered and bos and text.startswith(bos):
            add_special = False
        return self._tokenizer(text, add_special_tokens=add_special)["input_ids"]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    # -- generation --------------------------------------------------------

    def generate(
        self,
        prompt_ids: list[int],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> GenerationResult:
        """Run one completion. Meant to be called via :meth:`submit`."""
        import torch
        from transformers import StoppingCriteriaList

        if len(prompt_ids) >= self.max_context:
            raise PromptTooLong(len(prompt_ids), self.max_context)
        max_new = max(1, min(params.max_new_tokens, self.max_context - len(prompt_ids)))

        stop_event = cancel if cancel is not None else threading.Event()
        decoder = IncrementalDecoder(self._tokenizer, params.stop, on_text)
        streamer = CallbackStreamer(decoder, stop_event)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.pad_token_id,
            "do_sample": params.do_sample,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_EventStop(stop_event)]),
            "use_cache": True,
        }
        if params.do_sample:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p
            gen_kwargs["top_k"] = params.top_k if params.top_k > 0 else None
        else:
            # Silence the "temperature is set but do_sample is False" warnings
            # that the checkpoint's generation_config would otherwise trigger.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        if params.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = params.repetition_penalty
        if params.seed is not None:
            torch.manual_seed(params.seed)

        input_ids = torch.tensor([prompt_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        started = time.monotonic()
        with torch.inference_mode():
            output = self._model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
        elapsed = time.monotonic() - started

        generated = output[0, len(prompt_ids) :].tolist()
        decoder.finish()
        hit_eos = bool(generated) and generated[-1] in self.stop_token_ids
        if hit_eos or decoder.stop_hit or stop_event.is_set():
            finish_reason = "stop"
        elif len(generated) >= max_new:
            finish_reason = "length"
        else:
            finish_reason = "stop"
        return GenerationResult(
            text=decoder.text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            compute_seconds=elapsed,
        )

    # -- internals ---------------------------------------------------------

    def _collect_stop_ids(self) -> list[int]:
        tok = self._tokenizer
        ids: list[int] = []

        def add(value: Any) -> None:
            if isinstance(value, int) and value >= 0 and value not in ids:
                ids.append(value)

        gen_eos = getattr(self._model.generation_config, "eos_token_id", None)
        for v in gen_eos if isinstance(gen_eos, list) else [gen_eos]:
            add(v)
        add(tok.eos_token_id)
        unk = tok.unk_token_id
        for token in _TURN_END_TOKENS:
            candidate = tok.convert_tokens_to_ids(token)
            if candidate is not None and candidate != unk:
                add(candidate)
        if not ids:
            raise EngineError("tokenizer defines no end-of-sequence token")
        return ids

    def _detect_context(self) -> int:
        cfg = self._model.config
        for attr in ("max_position_embeddings", "max_sequence_length", "n_positions"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        limit = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(limit, int) and 0 < limit < 1_000_000:
            return limit
        return 8192


def _base_streamer_class() -> type:
    from transformers.generation.streamers import BaseStreamer

    return BaseStreamer


class CallbackStreamer(_base_streamer_class()):  # type: ignore[misc]
    """Feed generated ids into an :class:`IncrementalDecoder` as they appear."""

    def __init__(self, decoder: IncrementalDecoder, stop_event: threading.Event) -> None:
        self.decoder = decoder
        self.stop_event = stop_event
        self._skip_next = True  # generate() puts the prompt first

    def put(self, value: Any) -> None:
        if self._skip_next:
            self._skip_next = False
            return
        self.decoder.push(value.reshape(-1).tolist())
        if self.decoder.stop_hit:
            self.stop_event.set()

    def end(self) -> None:
        pass


def _stopping_criteria_class() -> type:
    from transformers import StoppingCriteria

    return StoppingCriteria


class _EventStop(_stopping_criteria_class()):  # type: ignore[misc]
    """Stop generation once ``event`` is set (stop string matched, client gone)."""

    def __init__(self, event: threading.Event) -> None:
        super().__init__()
        self.event = event

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        import torch

        return torch.full(
            (input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device
        )
