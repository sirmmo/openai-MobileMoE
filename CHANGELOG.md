# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MOBILEMOE_TOOL_HISTORY` / `--tool-history` (default `context`): tool
  exchanges in the history are rendered as plain context, because a transcript
  with serialized `tool_calls` made the model imitate the JSON instead of
  answering. Found driving MobileMoE-S-QAT as the answer model behind aiproxy.
- `MOBILEMOE_DEFAULT_REPETITION_PENALTY` / `--repetition-penalty`: a server-side
  default for requests that omit `repetition_penalty`, for fixed clients that
  cannot send the extension. Off (1.0) by default.

## [0.1.2] - 2026-09-08

### Fixed

- Decode with `clean_up_tokenization_spaces=False`. The MobileMoE tokenizer
  enables the cleanup by default, which deletes the space before `?`, `.`, `,`
  and `!`, so `SELECT ?s ?label` came back as `SELECT?s?label`. Found while
  replaying a text-to-SPARQL loop; it corrupts any code-like output.

## [0.1.1] - 2026-09-08

First run against the real weights (`facebook/MobileMoE-M-QAT`, CPU).

### Fixed

- Pin `transformers>=4.57,<5`. Transformers 5 initialises models on the meta
  device, and the rotary embedding in Meta's remote modeling code calls
  `Tensor.item()` in `__init__`, so every MobileMoE checkpoint failed to load
  with `Tensor.item() cannot be called on meta tensors`. The 0.1.0 wheel and
  image resolved 5.x and could not serve the model they were built for.
- The engine logs a clear warning when it finds transformers 5 with
  `trust_remote_code` on.

### Added

- Measured CPU numbers for M-QAT in the configuration docs.

## [0.1.0] - 2026-09-07

First release.

### Added

- `POST /v1/chat/completions` with genuine token streaming, `n`, `stop`,
  `temperature`, `top_p`, `seed`, `max_tokens` / `max_completion_tokens`, and
  the non-standard `top_k` and `repetition_penalty`.
- `POST /v1/completions` accepting a string, a list of strings, or token
  arrays, streaming included.
- `GET /v1/models`, `GET /v1/models/{id}`, `GET /health`, `GET /`.
- Exact `usage` counts from the model's own tokenizer.
- `x_mobilemoe` extension block with timings and a list of accepted-but-ignored
  parameters.
- Serve any of the nine `facebook/MobileMoE-{S,M,L}-{Base,SFT,QAT}` checkpoints
  through `MOBILEMOE_MODEL`; base checkpoints serve `/v1/completions` only.
- CPU (amd64 + arm64) and CUDA container images, `docker-compose.yml`,
  optional bearer-token auth, CORS.
- Offline unit suite through a fake engine and a live suite that runs against a
  container in CI, using an ungated stand-in model when no `HF_TOKEN` secret is
  configured.
