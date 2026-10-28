# langchain-tokenusage-dev

Four read-only LangChain tools for tokenusage.dev's public Codex reset
announcements and Nerf Watch scores and history. No tokenusage.dev API key,
account or subscription is required.

Python 3.10 or newer is required. The only direct runtime dependency is
`langchain-core>=1.0,<2`; HTTP uses Python's standard library.

## Install

```bash
pip install langchain-tokenusage-dev
```

## Example

This example calls all four tools without a model or model-provider credentials.

```python
import json

from langchain_tokenusage_dev import TokenUsageToolkit

tools = {tool.name: tool for tool in TokenUsageToolkit().get_tools()}


def show(result):
    if isinstance(result, str):
        raise RuntimeError(result)
    print(json.dumps(result, indent=2))
    print(f"[Data: tokenusage.dev]({result['source']})")
    if "qualifier" in result:
        print(result["qualifier"])
    return result


show(tools["get_codex_reset_announcements"].invoke({"limit": 3}))
show(tools["get_codex_reset_calendar"].invoke({"month": "2026-10"}))
scores = show(tools["get_nerfwatch_scores"].invoke({}))

if scores["models"]:
    show(
        tools["get_nerfwatch_history"].invoke(
            {"model": scores["models"][0]["model"]}
        )
    )
```

Individual tools can also be constructed directly:

```python
from langchain_tokenusage_dev import GetCodexResetAnnouncementsTool

tool = GetCodexResetAnnouncementsTool(timeout=10.0)
result = tool.invoke({"limit": 3})
```

Pass the toolkit's `BaseTool` instances to your configured LangChain agent.
An agent's model package and provider credentials are separate from these tools.

## Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `get_codex_reset_announcements` | Optional integer `limit`, 1–50, default 10 | Recorded announcements, newest first |
| `get_codex_reset_calendar` | Optional UTC `month`, `YYYY-MM` | Recorded announcements grouped by UTC date, newest dates first |
| `get_nerfwatch_scores` | None | Latest scores, previous runs, changes and baselines |
| `get_nerfwatch_history` | Required `model` id from the scores tool | Retained history, oldest first |

All argument objects reject extra keys. Explicit `null` is not a valid argument.
Model ids contain 1–64 lowercase letters, digits, dots, underscores or hyphens,
starting with a letter or digit.

Successful calls return JSON-serialisable dictionaries. Each contains `source`.
Nerf Watch calls also contain `qualifier`. HTTP failures, timeouts, invalid
responses, invalid arguments and unknown model ids return tool error strings
through `invoke` and `ainvoke`.

The tools preserve the MCP projections and values. The sole additional result
field is `qualifier` on the two Nerf Watch results. Only stable public fields
are projected; private evidence, test questions and answers are excluded.

## Caching and scope

Tools share one in-process cache per canonical endpoint: 15 seconds for resets,
300 seconds for Nerf Watch. TTL starts when a request starts. Failed reads are
cached for the same interval. Concurrent calls share the read. There is no
background polling or automatic retry.

Please poll no faster than these intervals and use canonical URLs without
cache-busting query strings. Requests identify this package and its repository
in the User-Agent. The default HTTP timeout is 10 seconds.

Only announcements recorded from @thsottiaux on X are available, at most the
50 newest. This is not a complete history. A date without an entry does not
mean no reset was announced. A banked reset grant saves a reset for eligible
users to apply later and does not itself refill limits. An announcement is not
proof that any account's limit was reset; open its source post for wording
and eligibility.

Nerf Watch preserves missing days and suite versions. Scores and changes are
already rounded by the API; this package does not recalculate them. Read the
returned baseline and retention settings rather than assuming fixed values.

## Reuse and credit

When showing reset data to other people, display
[Data: tokenusage.dev](https://tokenusage.dev/resets) on the same screen or in
the same message.

When showing Nerf Watch scores, changes, history or metadata to other people,
display [Data: tokenusage.dev](https://tokenusage.dev/nerfwatch) on the same
screen or in the same message. Personal use needs no visible credit.

API and other relays must retain each dataset's source URL and pass on its
credit condition. Anyone displaying the data must show the applicable credit.

Retain this qualification when displaying Nerf Watch scores or changes:

> A lower score is a lower score on this suite, not proof that a provider changed a model: runs are noisy, and a provider's defaults, the endpoint or the runner itself can change.

The MIT licence covers this package's code. Reused service data remains subject
to the credit conditions above.

tokenusage.dev is unofficial and not affiliated with OpenAI or Anthropic.

## Links

- [LangChain guide](https://tokenusage.dev/langchain)
- [Public API reference](https://tokenusage.dev/developers)
- [Codex Reset Tracker](https://tokenusage.dev/resets)
- [Nerf Watch](https://tokenusage.dev/nerfwatch)
- [Nerf Watch method](https://tokenusage.dev/nerfwatch/method)
- [Reuse terms](https://tokenusage.dev/terms)
- [Source and issues](https://github.com/stdcmms-lang/langchain-tokenusage-dev)

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m build
python -m twine check --strict dist/*
```

Unit tests use local JSON fixtures and mock HTTP. Production integration tests
are marked and skipped unless explicitly enabled:

```bash
python -m pytest --run-integration -m integration tests/integration_tests
```
