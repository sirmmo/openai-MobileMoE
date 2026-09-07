"""Shared fixtures.

Every import of ``openai_mobilemoe`` or ``fastapi`` happens *inside* a fixture.
``conftest.py`` is loaded for every session, including the live suite, which
runs against a container with only ``pytest`` and ``httpx`` available -- a
module-level import here would make torch mandatory for that suite too.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def engine():
    from fakes import FakeEngine

    return FakeEngine()


@pytest.fixture
def settings():
    from openai_mobilemoe.config import Settings

    # Deterministic settings, independent of the caller's environment.
    return Settings(
        model="facebook/MobileMoE-S-QAT",
        api_key=None,
        expose_extras=True,
        max_new_tokens=64,
        request_timeout=5.0,
    )


@pytest.fixture
def client(settings, engine):
    from fastapi.testclient import TestClient

    from openai_mobilemoe.server import create_app

    app = create_app(settings=settings, engine=engine)
    with TestClient(app) as test_client:
        test_client.engine = engine  # type: ignore[attr-defined]
        yield test_client
