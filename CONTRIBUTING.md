# Contributing

Thanks for looking. This is a small, focused project: an OpenAI-compatible
translation layer over Meta's MobileMoE checkpoints. Contributions that keep it
small and honest are very welcome.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest httpx ruff
pytest -m "not live"      # 70 tests, no model download needed
ruff check . && ruff format --check .
```

## Ground rules

**Keep `translate.py` pure.** Every OpenAI ↔ prompt/parameter mapping decision
lives there as a function with no I/O, which is why the whole HTTP surface can
be tested without loading a model. New mapping logic belongs there with a unit
test, not inline in a route handler.

**Run the live suite when you touch `engine.py`.**

```bash
docker compose up -d
MOBILEMOE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live
```

The unit tests drive a fake engine and cannot catch a regression in incremental
decoding, stop-string handling or cancellation on the real tokenizer. Without an
`HF_TOKEN` you can still run it against an ungated stand-in:

```bash
MOBILEMOE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct openai-mobilemoe
```

**Document limitations rather than hiding them.** If MobileMoE cannot do
something an OpenAI client expects, the right outcome is a clear 400 or a
warning in `x_mobilemoe.warnings`, plus an entry in
[`docs/fidelity.md`](https://ingmmo.com/openai-MobileMoE/fidelity/). Silently
accepting a parameter that does nothing is the thing we are trying to avoid.

## Pull requests

- One concern per PR.
- Lint and unit tests must pass; CI runs them on Python 3.10–3.13.
- Update `CHANGELOG.md` under "Unreleased".
- If you change or add an endpoint, update `docs/api-reference.md`.

## Reporting bugs

Please include the request body, the response (including the `x_mobilemoe`
block if present), the output of `GET /health`, and which checkpoint you were
serving. For load or crash reports, `docker logs` matters.
