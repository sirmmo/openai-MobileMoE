# Configuration

Every setting is an environment variable with a `MOBILEMOE_` prefix. CLI flags
override the environment. `.env.example` in the repository lists them all with
comments.

## Model

| Variable | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `MOBILEMOE_MODEL` | `--model` | `facebook/MobileMoE-S-QAT` | HuggingFace repo id or local directory. Any of the nine checkpoints. |
| `MOBILEMOE_MODEL_ID` | `--model-id` | last segment of the repo | Name advertised by `/v1/models`. |
| `MOBILEMOE_STRICT_MODEL` | `--strict-model` | `false` | 404 on requests naming another model. |
| `MOBILEMOE_TRUST_REMOTE_CODE` | — | `true` | MobileMoE ships its own modeling code, so this must stay on for the official repos. |
| `HF_TOKEN` | — | — | Read token for the gated repos. Read by `huggingface_hub`. |
| `HF_HOME` | — | `/cache/huggingface` in the image | Where checkpoints are cached. Mount a volume at `/cache`. |

## Hardware

| Variable | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `MOBILEMOE_DEVICE` | `--device` | `auto` | `auto` = CUDA if available, else CPU. Or `cpu`, `cuda`, `cuda:1`, `mps`. |
| `MOBILEMOE_DTYPE` | `--dtype` | `auto` | `auto` = `bfloat16` on CUDA, `float32` on CPU. Also `float16`. |
| `MOBILEMOE_ATTN_IMPLEMENTATION` | `--attn-implementation` | unset | `sdpa` or `eager`; unset lets Transformers choose. |
| `MOBILEMOE_THREADS` | `--threads` | torch default | `torch.set_num_threads` for CPU inference. |

`bfloat16` on CPU halves memory but is only fast on CPUs with native bf16
support (recent Xeon / EPYC, Apple silicon via `mps`). On older x86 it is
markedly slower than `float32`, which is why `auto` picks `float32` there.

## Generation

| Variable | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `MOBILEMOE_MAX_NEW_TOKENS` | `--max-new-tokens` | `512` | Cap when a request omits `max_tokens`. Clamped to the remaining context. |
| `MOBILEMOE_MAX_CONTEXT` | `--max-context` | from the model (8192) | Override the context window, for example to bound prompt size. |
| `MOBILEMOE_DEFAULT_TEMPERATURE` | — | `0` | Temperature when a request omits one. `0` = greedy. |
| `MOBILEMOE_DEFAULT_REPETITION_PENALTY` | `--repetition-penalty` | `1.0` | Applied when a request omits `repetition_penalty`. `1.0` = off. See [fidelity notes](fidelity.md#things-about-the-checkpoints). |

## Server

| Variable | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `MOBILEMOE_HOST` | `--host` | `0.0.0.0` | Bind address. |
| `MOBILEMOE_PORT` | `--port` | `8000` | Bind port. |
| `MOBILEMOE_API_KEY` | `--api-key` | unset | Require `Authorization: Bearer <key>` on `/v1/*`. |
| `MOBILEMOE_MAX_QUEUE_DEPTH` | `--max-queue-depth` | `16` | Requests waiting for the generation thread before new ones get 429. |
| `MOBILEMOE_REQUEST_TIMEOUT` | `--request-timeout` | `600` | Seconds a request may spend queued + generating. |
| `MOBILEMOE_EXPOSE_EXTRAS` | `--no-extras` | `true` | Include `x_mobilemoe` on responses. |
| `MOBILEMOE_ALLOWED_ORIGINS` | — | `*` | CORS origins, comma-separated. |
| — | `--log-level` | `info` | uvicorn / application log level. |

## Memory

The QAT checkpoints store INT4 weights but dequantize them into the serving
dtype on load, so at run time they cost the same as the SFT/Base rows. Figures
are the parameter count times the dtype width plus roughly 20% headroom for
activations, the KV cache and Python; measure on your own hardware before
sizing a container.

| Size | Total params | `float32` | `bfloat16` / `float16` |
| --- | --- | --- | --- |
| S | 1.3B | ~5.5 GB | ~3 GB |
| M | 2.8B | ~11.5 GB | ~6 GB |
| L | 5.3B | ~21.5 GB | ~11 GB |

`docker-compose.yml` sets `mem_limit: 8g`, enough for S at `float32`. Raise it
for M and L, or switch to `MOBILEMOE_DTYPE=bfloat16`.

### Measured: M-QAT on an old CPU

One data point, `facebook/MobileMoE-M-QAT` in the CPU container on two Xeon
E5-2640 v4 (2016, AVX2 only, no AVX-512, no native bf16), `float32`,
20 threads, otherwise idle:

| | |
| --- | --- |
| Load + INT4 dequantisation | ~70 s |
| Resident memory | 10.9 GiB |
| Prefill | ~27 tok/s |
| Decode | 2.6–4 tok/s |
| Time to first streamed token (25-token prompt) | ~3 s |

Decode is far slower than prefill because Meta's reference MoE layer loops over
all 60 experts in Python for every token; on hardware this old that overhead
dominates. Expect several times these numbers on a current desktop CPU, and
tens of tokens per second on any CUDA GPU. Thread oversubscription hurts
decode badly: on a loaded host, set `MOBILEMOE_THREADS` to the number of cores
you can actually dedicate.

## Serving a converted checkpoint

The model card shows how to export a QAT checkpoint to plain BF16 safetensors
for runtimes that lack INT4 support. The result loads here unchanged:

```bash
MOBILEMOE_MODEL=/models/MobileMoE-S-QAT-bf16 openai-mobilemoe
```

Mount the directory into the container and point `MOBILEMOE_MODEL` at it.
`MOBILEMOE_TRUST_REMOTE_CODE` stays on: the export still uses the custom
architecture code copied alongside the weights.
