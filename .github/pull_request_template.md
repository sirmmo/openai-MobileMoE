## What

<!-- One or two sentences. Link the issue if there is one. -->

## Why

## Checklist

- [ ] `ruff check . && ruff format --check .` pass
- [ ] `pytest -m "not live"` passes
- [ ] Ran the live suite if `engine.py` changed
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] `docs/api-reference.md` updated if an endpoint changed
