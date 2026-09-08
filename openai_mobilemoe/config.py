"""Runtime configuration, read from the environment (``MOBILEMOE_*``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "facebook/MobileMoE-S-QAT"

#: Every checkpoint Meta publishes under the MobileMoE name. All share one
#: custom architecture (``model_type: mobilemoe``) and one Llama-3 tokenizer,
#: so the server treats them interchangeably: set ``MOBILEMOE_MODEL`` to any of
#: them. Only the SFT and QAT rows carry a chat template; the Base rows are raw
#: language models and are best driven through ``/v1/completions``.
MODEL_FAMILY: tuple[str, ...] = (
    "facebook/MobileMoE-S-Base",
    "facebook/MobileMoE-S-SFT",
    "facebook/MobileMoE-S-QAT",
    "facebook/MobileMoE-M-Base",
    "facebook/MobileMoE-M-SFT",
    "facebook/MobileMoE-M-QAT",
    "facebook/MobileMoE-L-Base",
    "facebook/MobileMoE-L-SFT",
    "facebook/MobileMoE-L-QAT",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000

    #: HuggingFace repo id (or local directory) of the checkpoint to serve.
    model: str = DEFAULT_MODEL

    #: Model id advertised by ``GET /v1/models``. Defaults to the last path
    #: segment of :attr:`model`, e.g. ``MobileMoE-S-QAT``.
    model_id: str | None = None

    #: Reject requests naming a model we do not serve. Off by default because
    #: many OpenAI clients hard-code names like "gpt-4o-mini".
    strict_model: bool = False

    #: When set, require ``Authorization: Bearer <key>``.
    api_key: str | None = None

    #: ``auto`` picks CUDA when available and falls back to CPU. Anything torch
    #: accepts (``cpu``, ``cuda``, ``cuda:1``, ``mps``) is passed through.
    device: str = "auto"

    #: ``auto`` is bfloat16 on CUDA and float32 on CPU. The INT4 QAT weights are
    #: dequantized into this dtype on the first forward pass.
    dtype: str = "auto"

    #: Forwarded to ``from_pretrained(attn_implementation=...)`` when set
    #: (``sdpa`` / ``eager``). Unset lets Transformers choose.
    attn_implementation: str | None = None

    #: MobileMoE ships its own modeling code, so this must stay on for the
    #: official checkpoints. Exposed so a converted / upstreamed copy can turn
    #: it off.
    trust_remote_code: bool = True

    #: ``torch.set_num_threads`` for CPU inference. Unset keeps torch's default.
    threads: int | None = None

    #: Default cap on generated tokens when the request omits ``max_tokens``.
    max_new_tokens: int = 512

    #: Context window. Unset reads ``max_position_embeddings`` from the model
    #: config (8192 for every MobileMoE checkpoint).
    max_context: int | None = None

    #: Temperature used when the request omits one. The model card recommends
    #: greedy decoding, so the default is 0 rather than OpenAI's 1.0.
    default_temperature: float = 0.0

    #: Transformers' multiplicative repetition penalty applied when the request
    #: omits ``repetition_penalty``. 1.0 = off. Small MobileMoE checkpoints can
    #: loop on repetitive input (a list of near-identical rows); 1.1 to 1.2 stops
    #: that for prose, but penalises legitimate repetition in code and queries,
    #: so it is off unless a deployment serving one fixed client turns it on.
    default_repetition_penalty: float = 1.0

    #: Reject a request outright once this many are already waiting for the
    #: engine. Generation serializes through one thread, so an unbounded queue
    #: just converts load into timeouts.
    max_queue_depth: int = 16

    #: Seconds a single request may spend queued + generating before giving up.
    request_timeout: float = 600.0

    #: How tool exchanges in the history reach a model that was never trained on
    #: them. ``context``: assistant turns that only carry ``tool_calls`` are
    #: dropped and each ``tool`` result becomes a user turn ("Tool result (name):
    #: ..."), so the model sees retrieved material as plain context and answers
    #: in prose. ``template``: pass the ``tool`` role and the serialized calls to
    #: the chat template as-is, which makes MobileMoE imitate the JSON instead.
    tool_history: str = "context"

    #: Include timings and warnings as an ``x_mobilemoe`` object on responses.
    expose_extras: bool = True

    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def served_model_id(self) -> str:
        return self.model_id or self.model.rstrip("/").split("/")[-1]

    def served_names(self) -> set[str]:
        """Every name a request may use to address the loaded model."""
        served = self.served_model_id
        names = {served, self.model, "mobilemoe"}
        names |= {n.lower() for n in list(names)}
        return names

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("MOBILEMOE_ALLOWED_ORIGINS", "*")
        return cls(
            host=os.environ.get("MOBILEMOE_HOST", "0.0.0.0"),
            port=_env_int("MOBILEMOE_PORT", 8000) or 8000,
            model=os.environ.get("MOBILEMOE_MODEL") or DEFAULT_MODEL,
            model_id=os.environ.get("MOBILEMOE_MODEL_ID") or None,
            strict_model=_env_bool("MOBILEMOE_STRICT_MODEL", False),
            api_key=os.environ.get("MOBILEMOE_API_KEY") or None,
            device=os.environ.get("MOBILEMOE_DEVICE") or "auto",
            dtype=os.environ.get("MOBILEMOE_DTYPE") or "auto",
            attn_implementation=os.environ.get("MOBILEMOE_ATTN_IMPLEMENTATION") or None,
            trust_remote_code=_env_bool("MOBILEMOE_TRUST_REMOTE_CODE", True),
            threads=_env_int("MOBILEMOE_THREADS", None),
            max_new_tokens=_env_int("MOBILEMOE_MAX_NEW_TOKENS", 512) or 512,
            max_context=_env_int("MOBILEMOE_MAX_CONTEXT", None),
            default_temperature=_env_float("MOBILEMOE_DEFAULT_TEMPERATURE", 0.0),
            default_repetition_penalty=_env_float("MOBILEMOE_DEFAULT_REPETITION_PENALTY", 1.0),
            max_queue_depth=_env_int("MOBILEMOE_MAX_QUEUE_DEPTH", 16) or 16,
            request_timeout=_env_float("MOBILEMOE_REQUEST_TIMEOUT", 600.0),
            tool_history=(os.environ.get("MOBILEMOE_TOOL_HISTORY") or "context").lower(),
            expose_extras=_env_bool("MOBILEMOE_EXPOSE_EXTRAS", True),
            allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )
