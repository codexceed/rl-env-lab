"""Prompt construction in Qwen3.5's native chat format (SPEC §10, §11.8).

The tokenizer is never imported here: ``render_prompt`` receives one from GPU-side code, and
``estimate_prompt_tokens`` is a CI-safe proxy for the real count.
"""

from __future__ import annotations

import json
from typing import Any

from envlab.calls import ToolCall
from envlab.sim.entities import State
from envlab.sim.schemas import PARAM_TYPES, TOOL_SCHEMAS
from envlab.tasks.instance import TaskInstance

SYSTEM_PROMPT = (
    "You are a billing back-office agent. Resolve the request with exactly one tool call. "
    "All money values are integer cents. Use only ids that appear in the records. "
    "Reply with the tool call only."
)

TOOLS: list[dict[str, Any]] = list(TOOL_SCHEMAS)

# §10 budget: real prompt <= PROMPT_TOKEN_LIMIT; the generator keeps the estimate <=
# ESTIMATE_BUDGET. The estimate is 850 + ceil(len(user_message) / 2.5).
PROMPT_TOKEN_LIMIT = 1536
ESTIMATE_BUDGET = 1500
ESTIMATE_OVERHEAD = 850

# Record groups shown to the model, in order. The ledger is never shown.
_GROUPS = ("customers", "invoices", "payments", "cases")


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop ``note == ""``, ``resolution is None`` and ``credit_balance == 0`` (§10)."""
    out = {}
    for k, v in d.items():
        if (k == "note" and v == "") or (k == "resolution" and v is None) or (k == "credit_balance" and v == 0):
            continue
        out[k] = v
    return out


def record_line(d: dict[str, Any]) -> str:
    """One record as compact JSON, fields in dataclass order."""
    return json.dumps(_compact(d), separators=(",", ":"), ensure_ascii=False)


def format_user_message(request: str, state: State) -> str:
    """The §10 user message for a request and a state (records in state order)."""
    sd = state.to_dict()
    parts = [f"Request: {request}", "", "Records:"]
    for group in _GROUPS:
        if sd[group]:
            parts.append(f"{group}:")
            parts.extend(record_line(r) for r in sd[group])
    return "\n".join(parts)


def user_message(instance: TaskInstance) -> str:
    return format_user_message(instance.request, instance.initial_state)


def build_messages(instance: TaskInstance) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(instance)},
    ]


def render_prompt(tokenizer: Any, instance: TaskInstance) -> str:
    """Full prompt text via the model's own chat template. Thinking is always off (§10)."""
    return tokenizer.apply_chat_template(
        build_messages(instance),
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


def render_call(call: ToolCall) -> str:
    """A tool call in Qwen3.5's native XML-parameter form; args in ``PARAM_TYPES`` order."""
    order = list(PARAM_TYPES.get(call.name, {}))
    names = [k for k in order if k in call.arguments]
    names += [k for k in call.arguments if k not in order]
    lines = ["<tool_call>", f"<function={call.name}>"]
    for k in names:
        lines += [f"<parameter={k}>", str(call.arguments[k]), "</parameter>"]
    lines += ["</function>", "</tool_call>"]
    return "\n".join(lines)


def estimate_tokens(request: str, state: State) -> int:
    """``850 + ceil(len(user_message) / 2.5)``, in integer arithmetic."""
    n = len(format_user_message(request, state))
    return ESTIMATE_OVERHEAD + (2 * n + 4) // 5


def estimate_prompt_tokens(instance: TaskInstance) -> int:
    return estimate_tokens(instance.request, instance.initial_state)
