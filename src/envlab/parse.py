"""Parse a model completion into one tool call (SPEC §10 "Parsing", §11.5).

Accepts Qwen3.5's native XML-parameter form and a JSON fallback:

    <tool_call>                         <tool_call>
    <function=NAME>                     {"name": NAME, "arguments": {...}}
    <parameter=ARG>                     </tool_call>
    VALUE
    </parameter>
    </function>
    </tool_call>
"""

import json
import re
from typing import Any

from envlab.calls import ParseError, ParseFailure, ToolCall
from envlab.sim.schemas import PARAM_TYPES, TOOL_NAMES

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([A-Za-z_]\w*)>(.*)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"\s*<parameter=([A-Za-z_]\w*)>(.*?)</parameter>", re.DOTALL)
_INT_RE = re.compile(r"-?\d+")


class _Fail(Exception):
    def __init__(self, reason: ParseError, detail: str = "") -> None:
        super().__init__(reason, detail)
        self.failure = ParseFailure(reason=reason, detail=detail)


def _strip_one_newline(value: str) -> str:
    """Drop one leading and one trailing newline (the template's layout)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _parse_native(inner: str) -> tuple[str, dict[str, Any]]:
    m = _FUNCTION_RE.fullmatch(inner)
    if m is None:
        raise _Fail(ParseError.MALFORMED, "expected <function=NAME>…</function>")
    name, body = m.group(1), m.group(2)
    args: dict[str, Any] = {}
    pos = 0
    while (pm := _PARAM_RE.match(body, pos)) is not None:
        key = pm.group(1)
        if key in args:
            raise _Fail(ParseError.MALFORMED, f"duplicate parameter {key!r}")
        args[key] = _strip_one_newline(pm.group(2))
        pos = pm.end()
    if body[pos:].strip():
        raise _Fail(ParseError.MALFORMED, "unexpected text between parameters")
    return name, args


def _parse_json(inner: str) -> tuple[str, dict[str, Any]]:
    try:
        obj = json.loads(inner)
    except (ValueError, RecursionError) as exc:
        raise _Fail(ParseError.MALFORMED, f"invalid JSON: {type(exc).__name__}") from None
    # Reading of "any other shape": exactly the two keys, name a string, arguments an object.
    if (
        not isinstance(obj, dict)
        or set(obj) != {"name", "arguments"}
        or not isinstance(obj["name"], str)
        or not isinstance(obj["arguments"], dict)
    ):
        raise _Fail(ParseError.MALFORMED, 'expected {"name": str, "arguments": object}')
    return obj["name"], obj["arguments"]


def _coerce(key: str, typ: type, value: Any) -> str | int:
    if typ is int:
        if type(value) is int:  # JSON int; bool is rejected
            return value
        if isinstance(value, str) and _INT_RE.fullmatch(value.strip()):
            return int(value.strip())
        raise _Fail(ParseError.BAD_ARGUMENT_TYPE, f"{key} must be an integer")
    if isinstance(value, str):
        return value.strip()
    raise _Fail(ParseError.BAD_ARGUMENT_TYPE, f"{key} must be a string")


def _parse(text: str) -> ToolCall:
    text = _THINK_RE.sub("", text)
    blocks = _BLOCK_RE.findall(text)
    if not blocks:
        if "<tool_call>" in text:
            raise _Fail(ParseError.MALFORMED, "unterminated <tool_call>")
        raise _Fail(ParseError.NO_TOOL_CALL)
    if len(blocks) > 1:
        raise _Fail(ParseError.MULTIPLE_TOOL_CALLS, f"{len(blocks)} blocks")

    inner = blocks[0].strip()
    name, raw_args = _parse_json(inner) if inner.startswith("{") else _parse_native(inner)

    if name not in TOOL_NAMES:
        raise _Fail(ParseError.UNKNOWN_TOOL, name)
    schema = PARAM_TYPES[name]
    missing = [k for k in schema if k not in raw_args]
    if missing:
        raise _Fail(ParseError.MISSING_ARGUMENT, ",".join(missing))
    extra = [k for k in raw_args if k not in schema]
    if extra:
        raise _Fail(ParseError.UNEXPECTED_ARGUMENT, ",".join(extra))
    return ToolCall(name=name, arguments={k: _coerce(k, t, raw_args[k]) for k, t in schema.items()})


def parse_tool_call(text: str) -> ToolCall | ParseFailure:
    """Parse a completion per SPEC §10; returns a `ParseFailure` instead of raising."""
    if not isinstance(text, str):
        return ParseFailure(ParseError.NO_TOOL_CALL, "completion is not a string")
    try:
        return _parse(text)
    except _Fail as fail:
        return fail.failure
