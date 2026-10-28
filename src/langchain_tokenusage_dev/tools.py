"""Four tools preserving tokenusage.dev's public MCP projections."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, ClassVar

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._client import TokenUsageAPIError, get_json

RESET_SOURCE = "https://tokenusage.dev/resets"
NERFWATCH_SOURCE = "https://tokenusage.dev/nerfwatch"
RESET_CALENDAR_COVERAGE = (
    "Only announcements this monitor recorded from @thsottiaux are shown, "
    "at most the 50 newest; this is not a complete history. "
    "A date without an entry does not mean no reset was announced."
)
NERFWATCH_QUALIFIER = (
    "A lower score is a lower score on this suite, not proof that a provider "
    "changed a model: runs are noisy, and a provider's defaults, the endpoint "
    "or the runner itself can change."
)
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
MODEL_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"

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
UNKNOWN_MODEL_ERROR = (
    "Unknown model. Call get_nerfwatch_scores for available model ids."
)


class _Arguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=False,
    )


class ResetAnnouncementsInput(_Arguments):
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "How many announcements to return, newest first: "
            "1 to 50, default 10."
        ),
    )


def _month_json_schema(schema: dict[str, Any]) -> None:
    # Omission is allowed, but an explicit JSON null is not.
    schema.pop("anyOf", None)
    schema.pop("default", None)
    schema["type"] = "string"
    schema["pattern"] = MONTH_PATTERN


class ResetCalendarInput(_Arguments):
    month: str | None = Field(
        default=None,
        description=(
            "An optional UTC calendar month, YYYY-MM. Omit it to return "
            "every recorded date in the cached response."
        ),
        json_schema_extra=_month_json_schema,
    )

    @field_validator("month", mode="before")
    @classmethod
    def valid_month(cls, value: Any) -> str:
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", value) is None
        ):
            raise ValueError(MONTH_ERROR)
        return value


class NerfwatchScoresInput(_Arguments):
    pass


class NerfwatchHistoryInput(_Arguments):
    model: str = Field(
        pattern=MODEL_PATTERN,
        description=(
            "A model id from get_nerfwatch_scores: 1 to 64 lowercase "
            "letters, digits, dots, underscores or hyphens, starting "
            "with a letter or digit."
        ),
    )

    @field_validator("model", mode="before")
    @classmethod
    def valid_model(cls, value: Any) -> str:
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value) is None
        ):
            raise ValueError(MODEL_ERROR)
        return value


def _pick(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in keys}


def _point(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _pick(value, ("day", "score", "suiteVersion"))


class _TokenUsageTool(BaseTool):
    timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    handle_tool_error: bool = True
    _path: ClassVar[str]

    def _project(self, response: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            result = self._project(
                get_json(self._path, self.timeout),
                **kwargs,
            )
            json.dumps(result, allow_nan=False)
            return result
        except TokenUsageAPIError as error:
            raise ToolException(str(error)) from error
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ToolException(
                f"tokenusage.dev returned an invalid response for {self._path}."
            ) from error

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._run, **kwargs)


class GetCodexResetAnnouncementsTool(_TokenUsageTool):
    name: str = "get_codex_reset_announcements"
    description: str = (
        "Codex usage-limit reset announcements recorded by tokenusage.dev's "
        "Reset Alerts from @thsottiaux's public posts on X, newest first "
        "(at most 50): each post's link, when it was posted, when this site "
        "recorded it and resetType, plus when the monitor last checked X "
        "successfully. resetType is automatic, banked or both; null means "
        "recorded before Reset Alerts recorded reset types. A banked reset "
        "grant saves a reset for eligible users to apply later and does not "
        "itself refill limits. An announcement may describe a forthcoming "
        "event or one already completed; a post is not proof that any "
        "account's limit was reset. Open its url for the exact wording and "
        "eligibility. Only @thsottiaux is watched. Unofficial, not affiliated "
        "with OpenAI. Source: https://tokenusage.dev/resets"
    )
    args_schema: type[BaseModel] = ResetAnnouncementsInput
    handle_validation_error: bool | str = LIMIT_ERROR
    _path: ClassVar[str] = "/api/resets"

    def _project(
        self,
        response: dict[str, Any],
        limit: int = 10,
    ) -> dict[str, Any]:
        return {
            "source": RESET_SOURCE,
            "generatedAt": response["generatedAt"],
            "lastPolledAt": response["lastPolledAt"],
            "resets": [
                _pick(
                    item,
                    ("postId", "url", "postedAt", "detectedAt", "resetType"),
                )
                for item in response["resets"][:limit]
            ],
        }


class GetCodexResetCalendarTool(_TokenUsageTool):
    name: str = "get_codex_reset_calendar"
    description: str = (
        "Codex usage-limit reset announcements recorded by tokenusage.dev's "
        "Reset Alerts from @thsottiaux's public posts on X, grouped by UTC "
        "calendar date, newest dates first, with posts within each date "
        "newest first (at most 50 announcements in total). Optional month "
        "selects a UTC month written YYYY-MM. Dates are UTC; only dates with "
        "recorded announcements are returned. Only announcements this "
        "monitor recorded from @thsottiaux are shown, at most the 50 newest; "
        "this is not a complete history. A date without an entry does not "
        "mean no reset was announced. Each item has postId, url, postedAt "
        "and resetType. resetType is automatic, banked or both; null means "
        "recorded before Reset Alerts recorded reset types. A banked reset "
        "grant saves a reset for eligible users to apply later and does not "
        "itself refill limits. An announcement may describe a forthcoming "
        "event or one already completed; a post is not proof that any "
        "account's limit was reset. Open its url for the exact wording and "
        "eligibility. Unofficial, not affiliated with OpenAI. "
        "Source: https://tokenusage.dev/resets"
    )
    args_schema: type[BaseModel] = ResetCalendarInput
    handle_validation_error: bool | str = MONTH_ERROR
    _path: ClassVar[str] = "/api/resets"

    def _project(
        self,
        response: dict[str, Any],
        month: str | None = None,
    ) -> dict[str, Any]:
        by_date: dict[str, list[dict[str, Any]]] = {}
        for item in response["resets"]:
            date = item["postedAt"][:10]
            if month is not None and not date.startswith(month + "-"):
                continue
            by_date.setdefault(date, []).append({
                "postId": item["postId"],
                "url": item["url"],
                "postedAt": item["postedAt"],
                "resetType": item.get("resetType"),
            })
        return {
            "source": RESET_SOURCE,
            "generatedAt": response["generatedAt"],
            "lastPolledAt": response["lastPolledAt"],
            "timeZone": "UTC",
            "coverage": RESET_CALENDAR_COVERAGE,
            "month": month,
            "days": [
                {"date": date, "resets": by_date[date]}
                for date in sorted(by_date, reverse=True)
            ],
        }


class GetNerfwatchScoresTool(_TokenUsageTool):
    name: str = "get_nerfwatch_scores"
    description: str = (
        "Latest public Nerf Watch scores, previous runs, changes and "
        "baselines for at most 100 models, ordered by provider, name and "
        "model id. No arguments. Histories are available through "
        "get_nerfwatch_history. Scores and changes are rounded to two "
        "decimals; comparisons are within one suite version. A lower score "
        "is a lower score on this suite, not proof that a provider changed "
        "a model. Unofficial, not affiliated with OpenAI or Anthropic. "
        "Source: https://tokenusage.dev/nerfwatch"
    )
    args_schema: type[BaseModel] = NerfwatchScoresInput
    _path: ClassVar[str] = "/api/nerfwatch"

    def _project(self, response: dict[str, Any]) -> dict[str, Any]:
        models = []
        for model in response["models"]:
            models.append({
                "model": model["model"],
                "name": model["name"],
                "provider": model["provider"],
                "latest": _point(model["latest"]),
                "previous": _point(model["previous"]),
                "change": model["change"],
                "baseline": model["baseline"],
                "baselineRuns": model["baselineRuns"],
                "changeVsBaseline": model["changeVsBaseline"],
                "trend": model["trend"],
            })
        return {
            "source": NERFWATCH_SOURCE,
            "generatedAt": response["generatedAt"],
            "dataUpdatedAt": response["dataUpdatedAt"],
            "baselineDays": response["baselineDays"],
            "minBaselineRuns": response["minBaselineRuns"],
            "flatPoints": response["flatPoints"],
            "models": models,
            "qualifier": NERFWATCH_QUALIFIER,
        }


class GetNerfwatchHistoryTool(_TokenUsageTool):
    name: str = "get_nerfwatch_history"
    description: str = (
        "Retained public Nerf Watch score history for one model, oldest "
        "first, including each point's suite version. The required model "
        "id is available from get_nerfwatch_scores. Missing days are "
        "missing, not zero; comparisons are within one suite version. "
        "A lower score is a lower score on this suite, not proof that a "
        "provider changed a model. Unofficial, not affiliated with OpenAI "
        "or Anthropic. Source: https://tokenusage.dev/nerfwatch"
    )
    args_schema: type[BaseModel] = NerfwatchHistoryInput
    handle_validation_error: bool | str = MODEL_ERROR
    _path: ClassVar[str] = "/api/nerfwatch"

    def _project(
        self,
        response: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        selected = next(
            (item for item in response["models"] if item["model"] == model),
            None,
        )
        if selected is None:
            raise ToolException(UNKNOWN_MODEL_ERROR)
        return {
            "source": NERFWATCH_SOURCE,
            "generatedAt": response["generatedAt"],
            "dataUpdatedAt": response["dataUpdatedAt"],
            "historyDays": response["historyDays"],
            "model": selected["model"],
            "name": selected["name"],
            "provider": selected["provider"],
            "history": [
                _pick(point, ("day", "score", "suiteVersion"))
                for point in selected["history"]
            ],
            "qualifier": NERFWATCH_QUALIFIER,
        }


ALL_TOOL_CLASSES = (
    GetCodexResetAnnouncementsTool,
    GetCodexResetCalendarTool,
    GetNerfwatchScoresTool,
    GetNerfwatchHistoryTool,
)
