"""Tool calls, results and parse failures (contract file, SPEC.md Appendix A)."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, str | int]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(name=d["name"], arguments=dict(d["arguments"]))


class ErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    BAD_ARGUMENTS = "bad_arguments"
    NOT_FOUND = "not_found"
    INVALID_STATUS = "invalid_status"
    INVALID_AMOUNT = "invalid_amount"
    EXCEEDS_AMOUNT_DUE = "exceeds_amount_due"
    NO_CHANGE = "no_change"
    FIELD_NOT_EDITABLE = "field_not_editable"
    INVALID_VALUE = "invalid_value"
    HAS_ACTIVITY = "has_activity"


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    error: ErrorCode | None = None
    message: str = ""


class ParseError(StrEnum):
    NO_TOOL_CALL = "no_tool_call"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    MALFORMED = "malformed"
    UNKNOWN_TOOL = "unknown_tool"
    MISSING_ARGUMENT = "missing_argument"
    UNEXPECTED_ARGUMENT = "unexpected_argument"
    BAD_ARGUMENT_TYPE = "bad_argument_type"


@dataclass(frozen=True, slots=True)
class ParseFailure:
    reason: ParseError
    detail: str = ""
