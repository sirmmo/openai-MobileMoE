# openai-MobileMoE

[![CI](https://github.com/sirmmo/openai-MobileMoE/actions/workflows/ci.yml/badge.svg)](https://github.com/sirmmo/openai-MobileMoE/actions/workflows/ci.yml)
[![Docs](https://github.com/sirmmo/openai-MobileMoE/actions/workflows/docs.yml/badge.svg)](https://ingmmo.com/openai-MobileMoE/)
[![PyPI](https://img.shields.io/pypi/v/openai-mobilemoe)](https://pypi.org/project/openai-mobilemoe/)
[![Python](https://img.shields.io/pypi/pyversions/openai-mobilemoe)](https://pypi.org/project/openai-mobilemoe/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An OpenAI-compatible HTTP API in front of Meta's
[MobileMoE](https://huggingface.co/facebook/MobileMoE-S-QAT) — sparse
mixture-of-experts language models with 272M–922M *active* parameters, designed
to run on phones ([paper](https://arxiv.org/abs/2605.27358)). Point any OpenAI
client at it and chat with a model whose INT4 weights fit in 0.7 GB. Serves
every checkpoint in the family, on CPU or CUDA.

📖 **[Full documentation](https://ingmmo.com/openai-MobileMoE/)**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="MobileMoE-S-QAT",
    messages=[{"role": "user", "content": "Why are open-source on-device language models great?"}],
)
print(response.choices[0].message.content)

for chunk in client.chat.completions.create(
    model="MobileMoE-S-QAT",
    messages=[{"role": "user", "content": "Count to ten in words."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

> [!IMPORTANT]
> The weights are **gated** on HuggingFace behind Meta's
> [FAIR Noncommercial Research License](https://huggingface.co/facebook/MobileMoE-S-QAT/blob/main/LICENSE).
> Accept it once on the model page, create a
> [read token](https://huggingface.co/settings/tokens), and pass it as
> `HF_TOKEN`. This server is MIT; the weights are not, and they are
> noncommercial.

## Quickstart

```bash
docker run -d -p 8000:8000 -e HF_TOKEN=hf_... -v mobilemoe-cache:/cache \
  ghcr.io/sirmmo/openai-mobilemoe:latest
curl -s localhost:8000/health
```

Or from a clone, after copying `.env.example` to `.env` and filling in
`HF_TOKEN`:

```bash
docker compose up -d
```

Or as a Python package (needs Python ≥3.10):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build; skip on a CUDA box
pip install openai-mobilemoe
HF_TOKEN=hf_... openai-mobilemoe --port 8000
```

First start downloads the checkpoint into `/cache` (0.7 GB for S-QAT) and
dequantizes it, which takes a minute or two on CPU; the container reports
`"loading"` on `/health` until then, and the weights are cached across
restarts. On a GPU use `ghcr.io/sirmmo/openai-mobilemoe:latest-cuda` with
`--gpus all`.

Runnable examples: [`examples/openai_client.py`](examples/openai_client.py) and
[`examples/curl.sh`](examples/curl.sh).

## The family

Set `MOBILEMOE_MODEL` (or `--model`) to any of the nine checkpoints. They share
one architecture and one tokenizer; only size and training stage differ.

| Checkpoint | Active / total params | Weights on disk | Chat | RAM at float32 / bfloat16 |
| --- | --- | --- | --- | --- |
| `facebook/MobileMoE-S-QAT` (default) | 272M / 1.3B | 0.7 GB INT4 | yes | ~5.5 / ~3 GB |
| `facebook/MobileMoE-S-SFT` | 272M / 1.3B | 2.6 GB bf16 | yes | ~5.5 / ~3 GB |
| `facebook/MobileMoE-S-Base` | 272M / 1.3B | 2.6 GB bf16 | no | ~5.5 / ~3 GB |
| `facebook/MobileMoE-M-QAT` | 528M / 2.8B | 1.6 GB INT4 | yes | ~11.5 / ~6 GB |
| `facebook/MobileMoE-M-SFT` | 528M / 2.8B | 5.6 GB bf16 | yes | ~11.5 / ~6 GB |
| `facebook/MobileMoE-M-Base` | 528M / 2.8B | 5.6 GB bf16 | no | ~11.5 / ~6 GB |
| `facebook/MobileMoE-L-QAT` | 922M / 5.3B | 3.0 GB INT4 | yes | ~21.5 / ~11 GB |
| `facebook/MobileMoE-L-SFT` | 922M / 5.3B | 10.6 GB bf16 | yes | ~21.5 / ~11 GB |
| `facebook/MobileMoE-L-Base` | 922M / 5.3B | 10.6 GB bf16 | no | ~21.5 / ~11 GB |

The QAT rows are the ones built for devices: the INT4 weights are dequantized
into the serving dtype on load, so their RAM footprint matches the SFT rows
even though the download is four times smaller. All checkpoints have an 8,192
token context. The Base rows have no chat template and serve
`/v1/completions` only.

## What works

| | |
| --- | --- |
| **Chat** | `/v1/chat/completions` with system / user / assistant history. |
| **Real streaming** | Tokens are sent as the model produces them, not synthesized after the fact. |
| **Sampling** | `temperature`, `top_p`, `seed`, `stop`, `n`, `max_tokens`, plus `top_k` and `repetition_penalty`. |
| **Exact usage** | Prompt and completion token counts from the model's own tokenizer. |
| **Legacy completions** | Strings, lists of strings, or raw token arrays. |
| **Any checkpoint** | All nine MobileMoE repos, or a local directory. |
| **CPU or CUDA** | CPU images for amd64 and arm64, a CUDA image, or `pip`. |
| **Auth** | Optional bearer token; `/health` stays open. |

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Streaming, `n`, stop strings, sampling. |
| `POST /v1/completions` | Legacy text completion, streaming, token-array prompts. |
| `GET /v1/models`, `GET /v1/models/{id}` | Model discovery. |
| `GET /health` | Readiness, device, dtype, context length, queue depth. |

Full details in the [API reference](https://ingmmo.com/openai-MobileMoE/api-reference/).

## Key things to know

A short version of the [fidelity notes](https://ingmmo.com/openai-MobileMoE/fidelity/):

- **No tool calling.** MobileMoE is a plain chat model. `tools` and
  `functions` are accepted, ignored, and reported in `x_mobilemoe.warnings`;
  no `tool_calls` are ever returned.
- **JSON modes are prompt-guided only.** `response_format` adds an instruction
  to the system prompt; there is no constrained decoding, so validate what you
  get back.
- **Greedy by default.** The model card recommends `do_sample=False`, so an
  omitted `temperature` means 0, not OpenAI's 1.0. Send a temperature to sample.
- **One request at a time per container.** Generation serializes through a
  single thread (CPU inference would not go faster in parallel anyway).
  Requests queue up to `MOBILEMOE_MAX_QUEUE_DEPTH`, then get a 429. Scale by
  running more containers.
- **`logprobs`, `logit_bias`, `presence_penalty` and `frequency_penalty`** are
  accepted and ignored, with a warning.
- **Transformers 4.57 to 4.x only.** Transformers 5 builds models on the meta
  device and Meta's remote rotary-embedding code fails there; the package pins
  `transformers<5` until the upstream repo is updated.
- **Noncommercial.** Read Meta's license before deploying.

## Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest httpx ruff
pytest -m "not live"      # 72 tests, ~7s, no model download (fake engine)
```

```bash
docker compose up -d      # live suite against the real model (needs HF_TOKEN)
MOBILEMOE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live
```

Without an `HF_TOKEN`, an ungated Llama-style chat model exercises the same
code path: `MOBILEMOE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct openai-mobilemoe`.
CI does exactly that when the secret is absent.

See [Development](https://ingmmo.com/openai-MobileMoE/development/) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). The MobileMoE weights are published by Meta under
the [FAIR Noncommercial Research License](https://huggingface.co/facebook/MobileMoE-S-QAT/blob/main/LICENSE).
