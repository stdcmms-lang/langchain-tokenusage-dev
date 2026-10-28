"""Toolkit exposing all four public-data tools."""

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field

from .tools import ALL_TOOL_CLASSES


class TokenUsageToolkit(BaseToolkit):
    """Four read-only tokenusage.dev tools sharing process-local caches."""

    timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)

    def get_tools(self) -> list[BaseTool]:
        return [
            tool_class(timeout=self.timeout)
            for tool_class in ALL_TOOL_CLASSES
        ]
