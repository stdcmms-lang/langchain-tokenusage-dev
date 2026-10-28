from concurrent.futures import ThreadPoolExecutor
import io
from threading import Event
import urllib.error
import urllib.request

import pytest

from langchain_tokenusage_dev import (
    GetCodexResetAnnouncementsTool as Announcements,
    GetCodexResetCalendarTool as Calendar,
    GetNerfwatchScoresTool as Scores,
    TokenUsageToolkit,
)


def test_canonical_gets_identity_and_timeout(http):
    tools = TokenUsageToolkit(timeout=2.5).get_tools()
    tools[0].invoke({"limit": 1})
    tools[1].invoke({"month": "2026-09"})
    tools[2].invoke({})
    tools[3].invoke({"model": "alpha"})
    assert [request.full_url for request, _ in http.calls] == [
        "https://tokenusage.dev/api/resets",
        "https://tokenusage.dev/api/nerfwatch",
    ]
    for request, timeout in http.calls:
        assert request.get_method() == "GET"
        assert timeout == 2.5
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers == {
            "accept": "application/json",
            "user-agent": (
                "langchain-tokenusage-dev/0.1.0 "
                "(+https://github.com/stdcmms-lang/langchain-tokenusage-dev)"
            ),
        }


@pytest.mark.parametrize("tool_class,ttl", [
    (Announcements, 15.0), (Scores, 300.0),
])
def test_exact_cache_expiry(http, tool_class, ttl):
    tool_class().invoke({})
    http.clock = ttl - 0.001
    tool_class().invoke({})
    assert len(http.calls) == 1
    http.clock = ttl
    tool_class().invoke({})
    assert len(http.calls) == 2


def test_endpoint_caches_expire_independently(http):
    Announcements().invoke({})
    Scores().invoke({})
    http.clock = 15.0
    Calendar().invoke({})
    Scores().invoke({})
    assert [request.full_url for request, _ in http.calls] == [
        "https://tokenusage.dev/api/resets",
        "https://tokenusage.dev/api/nerfwatch",
        "https://tokenusage.dev/api/resets",
    ]


def test_concurrent_reset_tools_share_one_read(http, monkeypatch):
    entered = Event()
    release = Event()
    original = http.open

    def blocked(request, timeout):
        entered.set()
        assert release.wait(5)
        return original(request, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(Announcements().invoke, {})
        assert entered.wait(5)
        second = executor.submit(Calendar().invoke, {})
        release.set()
        assert first.result(timeout=5)["resets"][0]["postId"] == "301"
        assert second.result(timeout=5)["days"][0]["date"] == "2026-10-07"
    assert len(http.calls) == 1


@pytest.mark.parametrize("status", [404, 429, 500, 503])
def test_http_errors_are_strings_and_cached(http, status):
    http.error = urllib.error.HTTPError(
        "https://tokenusage.dev/api/resets",
        status,
        "failure",
        {},
        io.BytesIO(b'{"message":"PRIVATE_ERROR_BODY"}'),
    )
    expected = (
        f"tokenusage.dev HTTP {status} for /api/resets; try again later."
    )
    assert Announcements().invoke({}) == expected
    http.error = None
    assert Calendar().invoke({}) == expected
    assert len(http.calls) == 1
    http.clock = 15.0
    assert isinstance(Calendar().invoke({}), dict)
    assert len(http.calls) == 2


@pytest.mark.parametrize("wrapped", [False, True])
def test_timeout_errors(http, wrapped):
    error = TimeoutError("PRIVATE_TIMEOUT_DETAIL")
    http.error = urllib.error.URLError(error) if wrapped else error
    assert Announcements().invoke({}) == (
        "tokenusage.dev request timed out for /api/resets after 10s."
    )


def test_network_error(http):
    http.error = urllib.error.URLError("PRIVATE_NETWORK_DETAIL")
    assert Scores().invoke({}) == (
        "tokenusage.dev network error for /api/nerfwatch; try again later."
    )


@pytest.mark.parametrize("raw", [
    b"{", b"\xff", b"[]", b"null", b"NaN",
])
def test_invalid_json(http, raw):
    http.raw = raw
    assert Announcements().invoke({}) == (
        "tokenusage.dev returned an invalid JSON object for /api/resets."
    )


def test_invalid_response_shape(http):
    http.bodies["/api/resets"] = {}
    assert Announcements().invoke({}) == (
        "tokenusage.dev returned an invalid response for /api/resets."
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeouts(timeout):
    with pytest.raises(ValueError):
        TokenUsageToolkit(timeout=timeout)
