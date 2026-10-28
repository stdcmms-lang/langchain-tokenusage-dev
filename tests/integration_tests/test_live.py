import json

import pytest

from langchain_tokenusage_dev import TokenUsageToolkit

pytestmark = pytest.mark.integration


def test_production_public_tools():
    tools = {tool.name: tool for tool in TokenUsageToolkit().get_tools()}
    announcements = tools["get_codex_reset_announcements"].invoke({"limit": 1})
    calendar = tools["get_codex_reset_calendar"].invoke({})
    scores = tools["get_nerfwatch_scores"].invoke({})

    for result in [announcements, calendar, scores]:
        assert isinstance(result, dict), result
        json.dumps(result, allow_nan=False)

    assert announcements["source"] == "https://tokenusage.dev/resets"
    assert len(announcements["resets"]) <= 1
    assert calendar["source"] == "https://tokenusage.dev/resets"
    assert calendar["timeZone"] == "UTC"
    assert scores["source"] == "https://tokenusage.dev/nerfwatch"
    assert len(scores["models"]) <= 100
    assert "qualifier" in scores

    if scores["models"]:
        result = tools["get_nerfwatch_history"].invoke({
            "model": scores["models"][0]["model"],
        })
        assert isinstance(result, dict), result
        assert result["source"] == scores["source"]
        assert result["qualifier"] == scores["qualifier"]
        json.dumps(result, allow_nan=False)
