"""Standard-library HTTP and shared process-local endpoint caches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from threading import Lock
from time import monotonic
from typing import Any
import urllib.error
import urllib.request

try:
    VERSION = version("langchain-tokenusage-dev")
except PackageNotFoundError:
    VERSION = "0.1.0"

ORIGIN = "https://tokenusage.dev"
USER_AGENT = (
    f"langchain-tokenusage-dev/{VERSION} "
    "(+https://github.com/stdcmms-lang/langchain-tokenusage-dev)"
)
_TTLS = {"/api/resets": 15.0, "/api/nerfwatch": 300.0}
_LOCKS = {path: Lock() for path in _TTLS}


class TokenUsageAPIError(Exception):
    """A public API request could not produce a JSON object."""


@dataclass(frozen=True)
class _Entry:
    started_at: float
    value: dict[str, Any] | None = None
    error: str | None = None


_CACHE: dict[str, _Entry] = {}


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _read_json(path: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        ORIGIN + path,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    timeout_message = (
        f"tokenusage.dev request timed out for {path} after {timeout:g}s."
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        raise TokenUsageAPIError(
            f"tokenusage.dev HTTP {status} for {path}; try again later."
        ) from None
    except TimeoutError:
        raise TokenUsageAPIError(timeout_message) from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TokenUsageAPIError(timeout_message) from None
        raise TokenUsageAPIError(
            f"tokenusage.dev network error for {path}; try again later."
        ) from None
    except OSError:
        raise TokenUsageAPIError(
            f"tokenusage.dev network error for {path}; try again later."
        ) from None

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError):
        raise TokenUsageAPIError(
            f"tokenusage.dev returned an invalid JSON object for {path}."
        ) from None
    if not isinstance(value, dict):
        raise TokenUsageAPIError(
            f"tokenusage.dev returned an invalid JSON object for {path}."
        )
    return value


def get_json(path: str, timeout: float) -> dict[str, Any]:
    """Read an allowed endpoint, sharing successes and errors for its TTL."""
    if path not in _TTLS:
        raise ValueError("Unsupported tokenusage.dev endpoint")

    with _LOCKS[path]:
        now = monotonic()
        entry = _CACHE.get(path)
        if entry is None or now >= entry.started_at + _TTLS[path]:
            try:
                entry = _Entry(now, value=_read_json(path, timeout))
            except TokenUsageAPIError as error:
                entry = _Entry(now, error=str(error))
            _CACHE[path] = entry

        if entry.error is not None:
            raise TokenUsageAPIError(entry.error)
        assert entry.value is not None
        return deepcopy(entry.value)
