# Development

## Setup

```bash
git clone https://github.com/sirmmo/openai-MobileMoE
cd openai-MobileMoE
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest httpx ruff openai
```

## Tests

```bash
pytest -m "not live"          # unit suite: fake engine, ~7s, no download
ruff check . && ruff format --check .
```

The unit suite drives the whole HTTP surface through `tests/fakes.py`, which
"tokenizes" on whitespace and renders a toy chat template, so it can assert on
the exact prompt the server built. `translate.py` and `IncrementalDecoder` are
tested directly.

### Live suite

`tests/test_live.py` talks to a running server over HTTP and needs only
`pytest`, `httpx` and optionally `openai`. It asserts on shape and behaviour,
not on the model's wording.

```bash
# the real thing (needs HF_TOKEN after accepting the license)
docker compose up -d
MOBILEMOE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live
```

Without a token, an ungated Llama-style instruct model goes through exactly the
same engine code — chat template, incremental decoding, stop strings,
cancellation — and is small enough for CI:

```bash
MOBILEMOE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct openai-mobilemoe --port 8765
MOBILEMOE_TEST_BASE_URL=http://127.0.0.1:8765 pytest -m live
```

CI builds the CPU image and runs the live suite against the real checkpoint
when the repository has an `HF_TOKEN` secret, and against the stand-in
otherwise.

## Docker

```bash
docker build --target cpu -t openai-mobilemoe:dev .
docker build --target cuda -t openai-mobilemoe:dev-cuda .
docker run --rm -p 8000:8000 -e HF_TOKEN=hf_... openai-mobilemoe:dev
```

The CPU stage installs torch from the CPU wheel index before
`requirements.txt` so the resolver never pulls the CUDA build. The CUDA stage
starts from `pytorch/pytorch:*-cudnn9-runtime`.

## Layout

```
openai_mobilemoe/
  __init__.py      version
  __main__.py      CLI
  config.py        Settings, MODEL_FAMILY
  engine.py        MobileMoEEngine, IncrementalDecoder, CallbackStreamer
  translate.py     pure OpenAI <-> prompt/params/response mapping
  schemas.py       pydantic request models, error envelope
  streaming.py     SSE chunk builders
  server.py        FastAPI app
tests/
  fakes.py         FakeEngine
  test_translate.py
  test_server.py
  test_live.py     -m live
docs/              this site (mkdocs-material)
examples/          openai_client.py, curl.sh
```

## Releasing

1. Update `CHANGELOG.md`, bump `version` in `pyproject.toml`,
   `openai_mobilemoe/__init__.py` and `docker-compose.yml`.
2. Tag and push: `git tag v0.2.0 && git push --tags`.
3. `release.yml` builds the sdist/wheel, publishes to PyPI through trusted
   publishing, pushes `ghcr.io/sirmmo/openai-mobilemoe:{version,major.minor,latest}`
   (CPU, amd64 + arm64) and the `-cuda` variants, and creates the GitHub
   release.

PyPI trusted publishing has to be configured once on the PyPI project
(owner `sirmmo`, repository `openai-MobileMoE`, workflow `release.yml`,
environment `pypi`); until then the PyPI job fails and the container and
release jobs still succeed.
