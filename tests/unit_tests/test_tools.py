from __future__ import annotations

import asyncio
from copy import deepcopy
import json

from langchain_core.tools import BaseTool, BaseToolkit
import pytest

from langchain_tokenusage_dev import (
    GetCodexResetAnnouncementsTool as Announcements,
    GetCodexResetCalendarTool as Calendar,
    GetNerfwatchHistoryTool as History,
    GetNerfwatchScoresTool as Scores,
    TokenUsageToolkit,
)

RESET_SOURCE = "https://tokenusage.dev/resets"
NERFWATCH_SOURCE = "https://tokenusage.dev/nerfwatch"
GENERATED = "2026-10-09T12:00:00.000Z"
UPDATED = "2026-10-09T11:59:00.000Z"
POLLED = "2026-10-09T11:59:30.000Z"
QUALIFIER = (
    "A lower score is a lower score on this suite, not proof that a provider "
    "changed a model: runs are noisy, and a provider's defaults, the endpoint "
    "or the runner itself can change."
)
COVERAGE = (
    "Only announcements this monitor recorded from @thsottiaux are shown, "
    "at most the 50 newest; this is not a complete history. "
    "A date without an entry does not mean no reset was announced."
)
LIMIT_ERROR = (
    "Invalid arguments: limit must be an integer from 1 to 50, "
    "and no other argument is accepted."
)
MONTH_ERROR = (
    "Invalid arguments: month must be a UTC month written YYYY-MM, "
    "and no other argument is accepted."
)
MODEL_ERROR = (
    "Invalid arguments: model must be a model id of 1 to 64 lowercase "
    "letters, digits, dots, underscores or hyphens, starting with a letter "
    "or digit; no other argument is accepted."
)
UNKNOWN = "Unknown model. Call get_nerfwatch_scores for available model ids."

EXPECTED_ANNOUNCEMENTS = {
    "source": RESET_SOURCE,
    "generatedAt": GENERATED,
    "lastPolledAt": POLLED,
    "resets": [
        {
            "postId": "301",
            "url": "https://x.com/i/status/301",
            "postedAt": "2026-10-07T19:50:07.000Z",
            "detectedAt": "2026-10-07T19:51:00.000Z",
            "resetType": "banked",
        },
        {
            "postId": "201",
            "url": "https://x.com/i/status/201",
            "postedAt": "2026-10-07T04:05:59.000Z",
            "detectedAt": "2026-10-07T04:06:30.000Z",
            "resetType": "automatic",
        },
        {
            "postId": "101",
            "url": "https://x.com/i/status/101",
            "postedAt": "2026-09-30T16:00:00.000Z",
            "detectedAt": "2026-10-07T04:06:30.000Z",
            "resetType": None,
        },
    ],
}
EXPECTED_CALENDAR = {
    "source": RESET_SOURCE,
    "generatedAt": GENERATED,
    "lastPolledAt": POLLED,
    "timeZone": "UTC",
    "coverage": COVERAGE,
    "month": None,
    "days": [
        {
            "date": "2026-10-07",
            "resets": [
                {
                    "postId": "301",
                    "url": "https://x.com/i/status/301",
                    "postedAt": "2026-10-07T19:50:07.000Z",
                    "resetType": "banked",
                },
                {
                    "postId": "201",
                    "url": "https://x.com/i/status/201",
                    "postedAt": "2026-10-07T04:05:59.000Z",
                    "resetType": "automatic",
                },
            ],
        },
        {
            "date": "2026-09-30",
            "resets": [
                {
                    "postId": "101",
                    "url": "https://x.com/i/status/101",
                    "postedAt": "2026-09-30T16:00:00.000Z",
                    "resetType": None,
                },
            ],
        },
    ],
}


def point(day, score, suite="suite.1"):
    return {
        "day": f"2026-10-{day:02d}",
        "score": score,
        "suiteVersion": suite,
    }


EXPECTED_SCORES = {
    "source": NERFWATCH_SOURCE,
    "generatedAt": GENERATED,
    "dataUpdatedAt": UPDATED,
    "baselineDays": 7,
    "minBaselineRuns": 3,
    "flatPoints": 2,
    "models": [
        {
            "model": "alpha", "name": "Alpha", "provider": "Anthropic",
            "latest": point(9, 60), "previous": point(8, 78),
            "change": -18, "baseline": 84, "baselineRuns": 3,
            "changeVsBaseline": -24, "trend": "down",
        },
        {
            "model": "beta", "name": "Beta", "provider": "Anthropic",
            "latest": point(9, 44), "previous": point(8, 44),
            "change": 0, "baseline": 42, "baselineRuns": 3,
            "changeVsBaseline": 2, "trend": "flat",
        },
        {
            "model": "gamma", "name": "Gamma", "provider": "OpenAI",
            "latest": point(9, 54), "previous": point(8, 50),
            "change": 4, "baseline": 50, "baselineRuns": 3,
            "changeVsBaseline": 4, "trend": "up",
        },
        {
            "model": "shift", "name": "Shift", "provider": "OpenAI",
            "latest": point(9, 10), "previous": point(8, 99, "suite.0"),
            "change": None, "baseline": None, "baselineRuns": 0,
            "changeVsBaseline": None, "trend": None,
        },
    ],
    "qualifier": QUALIFIER,
}
EXPECTED_ALPHA = {
    "source": NERFWATCH_SOURCE,
    "generatedAt": GENERATED,
    "dataUpdatedAt": UPDATED,
    "historyDays": 366,
    "model": "alpha",
    "name": "Alpha",
    "provider": "Anthropic",
    "history": [point(2, 90), point(4, 84), point(8, 78), point(9, 60)],
    "qualifier": QUALIFIER,
}
EXPECTED_SHIFT = {
    "source": NERFWATCH_SOURCE,
    "generatedAt": GENERATED,
    "dataUpdatedAt": UPDATED,
    "historyDays": 366,
    "model": "shift",
    "name": "Shift",
    "provider": "OpenAI",
    "history": [point(8, 99, "suite.0"), point(9, 10)],
    "qualifier": QUALIFIER,
}


def test_toolkit_and_argument_schemas():
    toolkit = TokenUsageToolkit(timeout=2.5)
    tools = toolkit.get_tools()
    assert isinstance(toolkit, BaseToolkit)
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert [tool.name for tool in tools] == [
        "get_codex_reset_announcements",
        "get_codex_reset_calendar",
        "get_nerfwatch_scores",
        "get_nerfwatch_history",
    ]
    assert all(tool.timeout == 2.5 for tool in tools)
    schemas = [tool.args_schema.model_json_schema() for tool in tools]
    assert all(schema["additionalProperties"] is False for schema in schemas)
    assert schemas[0]["properties"]["limit"]["type"] == "integer"
    assert schemas[0]["properties"]["limit"]["minimum"] == 1
    assert schemas[0]["properties"]["limit"]["maximum"] == 50
    assert schemas[0]["properties"]["limit"]["default"] == 10
    month = schemas[1]["properties"]["month"]
    assert month["type"] == "string"
    assert month["pattern"] == r"^\d{4}-(0[1-9]|1[0-2])$"
    assert "anyOf" not in month and "default" not in month
    assert schemas[2]["properties"] == {}
    assert schemas[3]["required"] == ["model"]
    assert schemas[3]["properties"]["model"]["pattern"] == (
        r"^[a-z0-9][a-z0-9._-]{0,63}$"
    )
    for tool in tools:
        source = RESET_SOURCE if "codex" in tool.name else NERFWATCH_SOURCE
        assert tool.description.endswith("Source: " + source)


def test_exact_outputs(http):
    assert Announcements().invoke({}) == EXPECTED_ANNOUNCEMENTS
    assert Calendar().invoke({}) == EXPECTED_CALENDAR
    assert Scores().invoke({}) == EXPECTED_SCORES
    assert History().invoke({"model": "alpha"}) == EXPECTED_ALPHA
    assert History().invoke({"model": "shift"}) == EXPECTED_SHIFT
    assert len(http.calls) == 2


@pytest.mark.parametrize("limit,ids", [
    (1, ["301"]), (2, ["301", "201"]), (50, ["301", "201", "101"]),
])
def test_announcement_limits(http, limit, ids):
    result = Announcements().invoke({"limit": limit})
    assert [item["postId"] for item in result["resets"]] == ids


def test_default_ten_and_upper_bound_fifty(http):
    template = http.bodies["/api/resets"]["resets"][0]
    http.bodies["/api/resets"]["resets"] = [
        {**template, "postId": str(index)}
        for index in range(60)
    ]
    assert [
        item["postId"] for item in Announcements().invoke({})["resets"]
    ] == [str(index) for index in range(10)]
    assert len(Announcements().invoke({"limit": 50})["resets"]) == 50


@pytest.mark.parametrize("month,indices", [
    ("2026-10", [0]), ("2026-09", [1]), ("2026-08", []),
    ("0000-01", []), ("9999-12", []),
])
def test_calendar_months(http, month, indices):
    expected = {
        **EXPECTED_CALENDAR,
        "month": month,
        "days": [EXPECTED_CALENDAR["days"][index] for index in indices],
    }
    assert Calendar().invoke({"month": month}) == expected


def test_calendar_preserves_source_order_within_dates(http):
    http.bodies["/api/resets"]["resets"].reverse()
    result = Calendar().invoke({})
    assert [day["date"] for day in result["days"]] == [
        "2026-10-07", "2026-09-30"
    ]
    assert [
        item["postId"] for item in result["days"][0]["resets"]
    ] == ["201", "301"]


def test_calendar_legacy_type_and_null_poll_time(http):
    payload = http.bodies["/api/resets"]
    payload["lastPolledAt"] = None
    del payload["resets"][0]["resetType"]
    result = Calendar().invoke({})
    assert result["lastPolledAt"] is None
    assert result["days"][0]["resets"][0]["resetType"] is None


@pytest.mark.parametrize("tool_class,args,error", [
    (Announcements, {"limit": 0}, LIMIT_ERROR),
    (Announcements, {"limit": 51}, LIMIT_ERROR),
    (Announcements, {"limit": 1.0}, LIMIT_ERROR),
    (Announcements, {"limit": 1.5}, LIMIT_ERROR),
    (Announcements, {"limit": True}, LIMIT_ERROR),
    (Announcements, {"limit": "2"}, LIMIT_ERROR),
    (Announcements, {"limit": None}, LIMIT_ERROR),
    (Announcements, {"month": "2026-10"}, LIMIT_ERROR),
    (Announcements, {"limit": 1, "extra": 1}, LIMIT_ERROR),
    (Calendar, {"month": None}, MONTH_ERROR),
    (Calendar, {"month": ""}, MONTH_ERROR),
    (Calendar, {"month": "2026-1"}, MONTH_ERROR),
    (Calendar, {"month": "2026-00"}, MONTH_ERROR),
    (Calendar, {"month": "2026-13"}, MONTH_ERROR),
    (Calendar, {"month": "2026-10\n"}, MONTH_ERROR),
    (Calendar, {"month": "２０２６-１０"}, MONTH_ERROR),
    (Calendar, {"month": 202610}, MONTH_ERROR),
    (Calendar, {"month": True}, MONTH_ERROR),
    (Calendar, {"limit": 1}, MONTH_ERROR),
    (Calendar, {"month": "2026-10", "extra": 1}, MONTH_ERROR),
    (History, {}, MODEL_ERROR),
    (History, {"model": None}, MODEL_ERROR),
    (History, {"model": ""}, MODEL_ERROR),
    (History, {"model": "Alpha"}, MODEL_ERROR),
    (History, {"model": "-alpha"}, MODEL_ERROR),
    (History, {"model": "a/b"}, MODEL_ERROR),
    (History, {"model": "alpha\n"}, MODEL_ERROR),
    (History, {"model": "a" * 65}, MODEL_ERROR),
    (History, {"model": 1}, MODEL_ERROR),
    (History, {"model": "alpha", "extra": 1}, MODEL_ERROR),
])
def test_invalid_arguments_do_not_read(http, tool_class, args, error):
    assert tool_class().invoke(args) == error
    assert http.calls == []


@pytest.mark.parametrize("args", [{"limit": 1}, {"model": "alpha"}])
def test_scores_ignores_input_like_any_no_argument_tool(http, args):
    # langchain-core passes no arguments to a tool whose schema has no fields,
    # so stray input neither errors nor changes the result.
    assert Scores().invoke(args) == Scores().invoke({})
    assert len(http.calls) == 1


@pytest.mark.parametrize("model", ["a", "a.b_c-9", "a" * 64])
def test_valid_unknown_model_ids(http, model):
    assert History().invoke({"model": model}) == UNKNOWN
    assert len(http.calls) == 1


def test_unknown_models_share_cache(http):
    assert History().invoke({"model": "missing"}) == UNKNOWN
    assert History().invoke({"model": "also-missing"}) == UNKNOWN
    assert len(http.calls) == 1


def test_private_and_future_fields_are_projected_out(http):
    private_keys = [
        "evidenceAvailable", "latestAnswer", "checkLabel",
        "question", "tests", "futureField",
    ]

    def inject(value):
        if isinstance(value, dict):
            for child in list(value.values()):
                inject(child)
            value.update({key: "PRIVATE_SENTINEL" for key in private_keys})
        elif isinstance(value, list):
            for child in value:
                inject(child)

    for body in http.bodies.values():
        inject(body)

    results = [
        Announcements().invoke({}),
        Calendar().invoke({}),
        Scores().invoke({}),
        History().invoke({"model": "alpha"}),
    ]
    assert results == [
        EXPECTED_ANNOUNCEMENTS,
        EXPECTED_CALENDAR,
        EXPECTED_SCORES,
        EXPECTED_ALPHA,
    ]
    encoded = json.dumps(results, allow_nan=False)
    assert "PRIVATE_SENTINEL" not in encoded
    for key in private_keys:
        assert f'"{key}"' not in encoded


def test_settings_are_read_from_the_response(http):
    payload = http.bodies["/api/nerfwatch"]
    payload.update(
        historyDays=42, baselineDays=14, minBaselineRuns=5, flatPoints=1.25
    )
    scores = Scores().invoke({})
    assert (
        scores["baselineDays"], scores["minBaselineRuns"], scores["flatPoints"]
    ) == (14, 5, 1.25)
    assert History().invoke({"model": "alpha"})["historyDays"] == 42


def test_empty_datasets(http):
    http.bodies["/api/resets"].update(lastPolledAt=None, resets=[])
    http.bodies["/api/nerfwatch"].update(dataUpdatedAt=None, models=[])
    assert Announcements().invoke({}) == {
        **EXPECTED_ANNOUNCEMENTS, "lastPolledAt": None, "resets": []
    }
    assert Calendar().invoke({}) == {
        **EXPECTED_CALENDAR, "lastPolledAt": None, "days": []
    }
    assert Scores().invoke({}) == {
        **EXPECTED_SCORES, "dataUpdatedAt": None, "models": []
    }
    assert History().invoke({"model": "alpha"}) == UNKNOWN


def test_null_runs_are_preserved(http):
    model = http.bodies["/api/nerfwatch"]["models"][0]
    model.update(
        latest=None, previous=None, change=None, baseline=None,
        baselineRuns=0, changeVsBaseline=None, trend=None, history=[],
    )
    assert Scores().invoke({})["models"][0] == {
        "model": "alpha", "name": "Alpha", "provider": "Anthropic",
        "latest": None, "previous": None, "change": None,
        "baseline": None, "baselineRuns": 0,
        "changeVsBaseline": None, "trend": None,
    }
    assert History().invoke({"model": "alpha"})["history"] == []


def test_results_do_not_mutate_cached_data(http):
    first = Announcements().invoke({})
    first["resets"][0]["postId"] = "changed"
    assert Announcements().invoke({}) == EXPECTED_ANNOUNCEMENTS
    scores = Scores().invoke({})
    scores["models"][0]["latest"]["score"] = 0
    assert Scores().invoke({}) == EXPECTED_SCORES
    assert len(http.calls) == 2


def test_async_outputs_match_sync(http):
    async def run():
        return await asyncio.gather(
            Announcements().ainvoke({}),
            Calendar().ainvoke({}),
            Scores().ainvoke({}),
            History().ainvoke({"model": "alpha"}),
        )

    assert asyncio.run(run()) == [
        EXPECTED_ANNOUNCEMENTS,
        EXPECTED_CALENDAR,
        EXPECTED_SCORES,
        EXPECTED_ALPHA,
    ]
    assert len(http.calls) == 2


def test_tool_call_errors_have_error_status(http):
    result = Announcements().invoke({
        "type": "tool_call",
        "id": "invalid-limit",
        "name": "get_codex_reset_announcements",
        "args": {"limit": 0},
    })
    assert result.status == "error"
    assert result.content == LIMIT_ERROR
    assert http.calls == []
