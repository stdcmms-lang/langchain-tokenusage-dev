from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import urllib.request
from urllib.parse import urlsplit

import pytest

from langchain_tokenusage_dev import _client

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Allow marked production API integration tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(
        reason="Production API test; enable with --run-integration."
    )
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, request):
    _client._CACHE.clear()

    def denied(*args, **kwargs):
        raise AssertionError("Unexpected network access in an offline test")

    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setattr(urllib.request, "urlopen", denied)
    yield
    _client._CACHE.clear()


class FakeHTTP:
    def __init__(self):
        self.clock = 0.0
        self.calls: list[tuple[Any, float]] = []
        self.error: Exception | None = None
        self.raw: bytes | None = None
        self.bodies = {
            "/api/resets": json.loads(
                (FIXTURES / "resets.json").read_text(encoding="utf-8")
            ),
            "/api/nerfwatch": json.loads(
                (FIXTURES / "nerfwatch.json").read_text(encoding="utf-8")
            ),
        }

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        raw = self.raw
        if raw is None:
            path = urlsplit(request.full_url).path
            raw = json.dumps(self.bodies[path]).encode("utf-8")
        return io.BytesIO(raw)


@pytest.fixture
def http(isolate, monkeypatch):
    state = FakeHTTP()
    monkeypatch.setattr(_client, "monotonic", lambda: state.clock)
    monkeypatch.setattr(urllib.request, "urlopen", state.open)
    return state
