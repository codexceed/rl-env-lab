"""Tool schemas in HF/OpenAI function format (contract file, SPEC.md Appendix A)."""

from typing import Any


def _tool(name: str, description: str, params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": params, "required": list(params)},
        },
    }


_STR = {"type": "string"}
_CENTS = {"type": "integer", "description": "cents"}

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _tool(
        "apply_credit",
        "Credit an amount against an invoice's amount due.",
        {"invoice_id": _STR, "amount": _CENTS, "reason": _STR},
    ),
    _tool("refund_payment", "Refund a payment in full.", {"payment_id": _STR, "reason": _STR}),
    _tool(
        "adjust_invoice",
        "Set an invoice's billed total.",
        {"invoice_id": _STR, "new_amount": _CENTS, "reason": _STR},
    ),
    _tool(
        "update_customer",
        "Change one contact field of a customer.",
        {
            "customer_id": _STR,
            "field": {"type": "string", "enum": ["name", "email", "phone", "address"]},
            "value": _STR,
        },
    ),
    _tool(
        "resolve_case",
        "Close a dispute case. The adjustment is credited to the case's invoice.",
        {"case_id": _STR, "adjustment": _CENTS, "note": _STR},
    ),
    _tool(
        "void_invoice",
        "Void an open invoice that has no payments or credits.",
        {"invoice_id": _STR, "reason": _STR},
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOL_SCHEMAS)

# Parameter name -> Python type, in schema order (render and parse use this order).
PARAM_TYPES: dict[str, dict[str, type]] = {
    t["function"]["name"]: {
        k: int if v["type"] == "integer" else str for k, v in t["function"]["parameters"]["properties"].items()
    }
    for t in TOOL_SCHEMAS
}

# The argument naming the record a tool acts on.
TARGET_ARG: dict[str, str] = {
    "apply_credit": "invoice_id",
    "refund_payment": "payment_id",
    "adjust_invoice": "invoice_id",
    "update_customer": "customer_id",
    "resolve_case": "case_id",
    "void_invoice": "invoice_id",
}

# The free-text argument of a tool (stored as a note or memo). update_customer has none.
TEXT_ARG: dict[str, str | None] = {
    "apply_credit": "reason",
    "refund_payment": "reason",
    "adjust_invoice": "reason",
    "update_customer": None,
    "resolve_case": "note",
    "void_invoice": "reason",
}
