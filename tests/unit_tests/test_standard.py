from langchain_tests.unit_tests import ToolsUnitTests
import pytest

from langchain_tokenusage_dev import (
    GetCodexResetAnnouncementsTool,
    GetCodexResetCalendarTool,
    GetNerfwatchHistoryTool,
    GetNerfwatchScoresTool,
)

pytestmark = pytest.mark.usefixtures("http")


class _ToolTests(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return self.tool_type

    @property
    def tool_constructor_params(self):
        return {}

    @property
    def tool_invoke_params_example(self):
        return self.invoke_args

    @property
    def tool_has_pydantic_input_schema(self):
        return True


class TestAnnouncementsStandard(_ToolTests):
    tool_type = GetCodexResetAnnouncementsTool
    invoke_args = {"limit": 1}


class TestCalendarStandard(_ToolTests):
    tool_type = GetCodexResetCalendarTool
    invoke_args = {"month": "2026-10"}


class TestScoresStandard(_ToolTests):
    tool_type = GetNerfwatchScoresTool
    invoke_args = {}


class TestHistoryStandard(_ToolTests):
    tool_type = GetNerfwatchHistoryTool
    invoke_args = {"model": "alpha"}
