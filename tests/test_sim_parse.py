"""Tool-call parser (SPEC §10 "Parsing")."""

import json

import pytest

from envlab.calls import ParseError, ParseFailure, ToolCall
from envlab.parse import parse_tool_call
from envlab.sim.schemas import PARAM_TYPES

P = ParseError

SPEC_EXAMPLE = """<tool_call>
<function=apply_credit>
<parameter=invoice_id>
INV-48213
</parameter>
<parameter=amount>
1250
</parameter>
<parameter=reason>
Credit overcharge versus contract rate
</parameter>
</function>
</tool_call>"""

EXPECTED = ToolCall(
    "apply_credit",
    {"invoice_id": "INV-48213", "amount": 1250, "reason": "Credit overcharge versus contract rate"},
)


def native(name: str, **params) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def as_json(name, arguments) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def fails(text: str, reason: ParseError) -> None:
    got = parse_tool_call(text)
    assert isinstance(got, ParseFailure), got
    assert got.reason == reason, got


# --- valid native ---------------------------------------------------------------------------


def test_spec_example():
    got = parse_tool_call(SPEC_EXAMPLE)
    assert got == EXPECTED
    assert list(got.arguments) == list(PARAM_TYPES["apply_credit"])
    assert type(got.arguments["amount"]) is int


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("refund_payment", {"payment_id": "PAY-00002", "reason": "Refund"}),
        ("adjust_invoice", {"invoice_id": "INV-1", "new_amount": "0", "reason": "x"}),
        ("update_customer", {"customer_id": "C1", "field": "email", "value": "a@b.co"}),
        ("resolve_case", {"case_id": "CS-1", "adjustment": "-3", "note": "n"}),
        ("void_invoice", {"invoice_id": "INV-1", "reason": "r"}),
    ],
)
def test_native_every_tool(name, params):
    got = parse_tool_call(native(name, **params))
    assert isinstance(got, ToolCall) and got.name == name
    for k, typ in PARAM_TYPES[name].items():
        assert got.arguments[k] == typ(params[k])


def test_extra_text_around_call_is_ignored():
    text = "Sure, here you go:\n" + SPEC_EXAMPLE + "\nLet me know if you need anything else."
    assert parse_tool_call(text) == EXPECTED


def test_think_block_is_removed():
    text = "<think>\nmaybe <tool_call>x</tool_call> no\n</think>\n\n" + SPEC_EXAMPLE
    assert parse_tool_call(text) == EXPECTED


def test_multiple_think_blocks_removed():
    text = "<think>a</think>" + SPEC_EXAMPLE + "<think><tool_call>b</tool_call></think>"
    assert parse_tool_call(text) == EXPECTED


def test_compact_native_without_newlines():
    text = (
        "<tool_call><function=void_invoice><parameter=invoice_id>INV-1</parameter>"
        "<parameter=reason>r</parameter></function></tool_call>"
    )
    assert parse_tool_call(text) == ToolCall("void_invoice", {"invoice_id": "INV-1", "reason": "r"})


def test_values_are_stripped_and_only_one_newline_dropped():
    text = native("void_invoice", invoice_id="  INV-1 ", reason="\n\nline one\nline two\n")
    got = parse_tool_call(text)
    assert got == ToolCall("void_invoice", {"invoice_id": "INV-1", "reason": "line one\nline two"})


def test_int_with_surrounding_whitespace():
    got = parse_tool_call(native("apply_credit", invoice_id="I", amount="  42 ", reason=""))
    assert got.arguments["amount"] == 42
    assert got.arguments["reason"] == ""


def test_parameter_order_does_not_matter():
    text = native("apply_credit", reason="r", amount="5", invoice_id="I")
    got = parse_tool_call(text)
    assert got == ToolCall("apply_credit", {"invoice_id": "I", "amount": 5, "reason": "r"})
    assert list(got.arguments) == ["invoice_id", "amount", "reason"]


# --- valid JSON fallback --------------------------------------------------------------------


def test_json_fallback():
    args = {
        "invoice_id": "INV-48213",
        "amount": 1250,
        "reason": "Credit overcharge versus contract rate",
    }
    assert parse_tool_call(as_json("apply_credit", args)) == EXPECTED


def test_json_fallback_numeric_string_and_whitespace():
    args = {
        "invoice_id": " INV-48213 ",
        "amount": " 1250 ",
        "reason": "Credit overcharge versus contract rate",
    }
    assert parse_tool_call(as_json("apply_credit", args)) == EXPECTED


def test_json_fallback_with_surrounding_text():
    args = {"payment_id": "PAY-1", "reason": "r"}
    got = parse_tool_call("ok " + as_json("refund_payment", args) + " done")
    assert got == ToolCall("refund_payment", args)


# --- failures -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", P.NO_TOOL_CALL),
        ("I would credit the invoice.", P.NO_TOOL_CALL),
        ('{"name": "apply_credit", "arguments": {}}', P.NO_TOOL_CALL),  # no tags
        ("<think><tool_call>x</tool_call></think>", P.NO_TOOL_CALL),  # only inside think
        ("<tool_call>\n<function=void_invoice>", P.MALFORMED),  # unterminated
        ("</tool_call><tool_call>", P.MALFORMED),
        (SPEC_EXAMPLE + SPEC_EXAMPLE, P.MULTIPLE_TOOL_CALLS),
        ("<tool_call></tool_call>", P.MALFORMED),
        ("<tool_call>apply_credit(invoice_id='x')</tool_call>", P.MALFORMED),
        ("<tool_call><function=void_invoice></tool_call>", P.MALFORMED),
        ("<tool_call><function=void-invoice></function></tool_call>", P.MALFORMED),
        ("<tool_call><function=1abc></function></tool_call>", P.MALFORMED),
        ("<tool_call>x<function=void_invoice></function></tool_call>", P.MALFORMED),
        (
            "<tool_call><function=void_invoice><parameter=invoice_id>I</parameter>"
            " junk <parameter=reason>r</parameter></function></tool_call>",
            P.MALFORMED,
        ),
        (
            "<tool_call><function=void_invoice><parameter=invoice_id>I</parameter>"
            "<parameter=reason>r</parameter>trailing</function></tool_call>",
            P.MALFORMED,
        ),
        (
            "<tool_call><function=void_invoice><parameter=invoice_id>I</function></tool_call>",
            P.MALFORMED,
        ),
        (
            "<tool_call><function=void_invoice><parameter=invoice_id>I</parameter>"
            "<parameter=invoice_id>J</parameter><parameter=reason>r</parameter>"
            "</function></tool_call>",
            P.MALFORMED,
        ),
        ("<tool_call>{not json}</tool_call>", P.MALFORMED),
        ("<tool_call>" + '{"a":' * 100_000 + "</tool_call>", P.MALFORMED),  # too deep
        ('<tool_call>{"name": "void_invoice"}</tool_call>', P.MALFORMED),
        ('<tool_call>{"name": 3, "arguments": {}}</tool_call>', P.MALFORMED),
        ('<tool_call>{"name": "void_invoice", "arguments": "{}"}</tool_call>', P.MALFORMED),
        ('<tool_call>{"name": "void_invoice", "arguments": [1]}</tool_call>', P.MALFORMED),
        (
            '<tool_call>{"name": "void_invoice", "arguments": {}, "id": "x"}</tool_call>',
            P.MALFORMED,
        ),
        ('<tool_call>{"name": "void_invoice", "arguments": {}} extra</tool_call>', P.MALFORMED),
        ('<tool_call>["void_invoice", {}]</tool_call>', P.MALFORMED),
    ],
)
def test_parse_failures(text, reason):
    fails(text, reason)


@pytest.mark.parametrize("name", ["delete_customer", "Apply_credit", "issue_refund"])
def test_unknown_tool_native(name):
    fails(native(name, invoice_id="I"), P.UNKNOWN_TOOL)


def test_unknown_tool_json():
    fails(as_json("transfer_funds", {"amount": 1}), P.UNKNOWN_TOOL)


def test_missing_argument():
    fails(native("apply_credit", invoice_id="I", amount="1"), P.MISSING_ARGUMENT)
    fails(as_json("refund_payment", {"payment_id": "P"}), P.MISSING_ARGUMENT)
    fails(native("void_invoice"), P.MISSING_ARGUMENT)


def test_unexpected_argument():
    fails(native("void_invoice", invoice_id="I", reason="r", force="1"), P.UNEXPECTED_ARGUMENT)
    fails(
        as_json("void_invoice", {"invoice_id": "I", "reason": "r", "x": 1}), P.UNEXPECTED_ARGUMENT
    )


def test_missing_reported_before_unexpected():
    fails(native("void_invoice", invoice_id="I", why="r"), P.MISSING_ARGUMENT)


@pytest.mark.parametrize(
    "amount", ["12.50", "1,250", "$12", "+5", "twelve", "", "1e3", "0x10", "12 50", "--1"]
)
def test_bad_int_text(amount):
    fails(native("apply_credit", invoice_id="I", amount=amount, reason="r"), P.BAD_ARGUMENT_TYPE)


@pytest.mark.parametrize("amount", [12.5, 12.0, True, None, [1], {"v": 1}])
def test_bad_int_json(amount):
    args = {"invoice_id": "I", "amount": amount, "reason": "r"}
    fails(as_json("apply_credit", args), P.BAD_ARGUMENT_TYPE)


@pytest.mark.parametrize("value", [1, 1.5, True, None, ["a"], {"a": 1}])
def test_bad_str_json(value):
    fails(as_json("void_invoice", {"invoice_id": value, "reason": "r"}), P.BAD_ARGUMENT_TYPE)


def test_enum_not_checked_by_parser():
    got = parse_tool_call(native("update_customer", customer_id="C1", field="credit", value="9"))
    assert isinstance(got, ToolCall) and got.arguments["field"] == "credit"


def test_non_string_input_does_not_raise():
    assert isinstance(parse_tool_call(None), ParseFailure)  # type: ignore[arg-type]
