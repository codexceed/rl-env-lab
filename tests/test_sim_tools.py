"""Tool effects, error codes, purity (SPEC §3)."""

from dataclasses import replace

import pytest

from envlab.calls import ErrorCode, ToolCall, ToolResult
from envlab.sim.entities import (
    CaseStatus,
    Invoice,
    InvoiceStatus,
    LedgerEntry,
    LedgerKind,
    PaymentStatus,
    State,
)
from envlab.sim.ledger import check_invariants
from envlab.sim.schemas import PARAM_TYPES, TOOL_NAMES
from envlab.sim.state_hash import state_hash
from envlab.sim.tools import apply_tool
from sim_factory import make_state

E = ErrorCode


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name, arguments)


def ok(state: State, c: ToolCall) -> State:
    before = state_hash(state)
    new, res = apply_tool(state, c)
    assert res == ToolResult(ok=True), res
    assert new is not state
    assert state_hash(state) == before  # input never mutated
    assert check_invariants(new) == []
    return new


def err(state: State, c: ToolCall, code: ErrorCode) -> None:
    before = state_hash(state)
    new, res = apply_tool(state, c)
    assert new is state  # same object on error
    assert res.ok is False and res.error == code, res
    assert state_hash(state) == before


def invoice(state: State, invoice_id: str) -> Invoice:
    inv = state.invoice(invoice_id)
    assert inv is not None
    return inv


def changed(old: State, new: State) -> dict[str, list[int]]:
    """Positions of records that differ, per group (ledger: appended entries only)."""
    out = {}
    for group in ("customers", "invoices", "payments", "cases"):
        a, b = getattr(old, group), getattr(new, group)
        assert len(a) == len(b)
        idx = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        if idx:
            out[group] = idx
    assert new.ledger[: len(old.ledger)] == old.ledger
    return out


@pytest.fixture
def s() -> State:
    return make_state()


# --- apply_credit ---------------------------------------------------------------------------


def test_apply_credit_partial(s):
    new = ok(s, call("apply_credit", invoice_id="INV-00001", amount=1000, reason="  Overcharge "))
    inv = new.invoices[0]
    assert (inv.amount_billed, inv.amount_due, inv.status, inv.note) == (
        5999,
        4999,
        InvoiceStatus.OPEN,
        "Overcharge",
    )
    assert changed(s, new) == {"invoices": [0]}
    assert new.ledger[len(s.ledger) :] == (
        LedgerEntry("L0016", "C0001", "INV-00001", None, LedgerKind.CREDIT, -1000, "Overcharge"),
    )


def test_apply_credit_full_marks_paid(s):
    new = ok(s, call("apply_credit", invoice_id="INV-00001", amount=5999, reason=""))
    inv = new.invoices[0]
    assert (inv.amount_due, inv.status, inv.note) == (0, InvoiceStatus.PAID, "")
    assert new.ledger[-1].memo == ""


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"invoice_id": "INV-99999", "amount": 1}, E.NOT_FOUND),
        ({"invoice_id": "INV-00002", "amount": 1}, E.INVALID_STATUS),  # paid
        ({"invoice_id": "INV-00001", "amount": 0}, E.INVALID_AMOUNT),
        ({"invoice_id": "INV-00001", "amount": -5}, E.INVALID_AMOUNT),
        ({"invoice_id": "INV-00001", "amount": 6000}, E.EXCEEDS_AMOUNT_DUE),
        ({"invoice_id": "INV-00006", "amount": 6001}, E.EXCEEDS_AMOUNT_DUE),  # due < billed
        ({"invoice_id": "INV-00002", "amount": -1}, E.INVALID_STATUS),  # status checked first
    ],
)
def test_apply_credit_errors(s, args, code):
    err(s, call("apply_credit", reason="r", **args), code)


def test_apply_credit_void_invoice(s):
    s = ok(s, call("void_invoice", invoice_id="INV-00001", reason="r"))
    err(s, call("apply_credit", invoice_id="INV-00001", amount=1, reason="r"), E.INVALID_STATUS)


# --- refund_payment -------------------------------------------------------------------------


def test_refund_duplicate(s):
    new = ok(s, call("refund_payment", payment_id="PAY-00002", reason="Refund duplicate payment"))
    pay, inv = new.payments[0], new.invoices[1]
    assert (pay.status, pay.note) == (PaymentStatus.REFUNDED, "Refund duplicate payment")
    assert (inv.amount_due, inv.status, inv.note) == (0, InvoiceStatus.PAID, "")
    assert changed(s, new) == {"invoices": [1], "payments": [0]}
    assert new.ledger[len(s.ledger) :] == (
        LedgerEntry(
            "L0016",
            "C0001",
            "INV-00002",
            "PAY-00002",
            LedgerKind.REFUND,
            4999,
            "Refund duplicate payment",
        ),
    )


def test_refund_reopens_paid_invoice(s):
    new = ok(s, call("refund_payment", payment_id="PAY-00004", reason="r"))
    inv = invoice(new, "INV-00004")
    assert (inv.amount_due, inv.status) == (3000, InvoiceStatus.OPEN)


@pytest.mark.parametrize(
    ("payment_id", "code"),
    [
        ("PAY-99999", E.NOT_FOUND),
        ("PAY-00003", E.INVALID_STATUS),  # failed
        ("PAY-00006", E.INVALID_STATUS),  # already refunded
    ],
)
def test_refund_errors(s, payment_id, code):
    err(s, call("refund_payment", payment_id=payment_id, reason="r"), code)


def test_refund_payment_invoice_missing(s):
    s = replace(s, payments=(replace(s.payments[0], invoice_id="INV-99999"), *s.payments[1:]))
    err(s, call("refund_payment", payment_id="PAY-00002", reason="r"), E.NOT_FOUND)


def test_refund_on_void_invoice(s):
    inv = replace(s.invoices[1], status=InvoiceStatus.VOID, amount_due=0)
    s = replace(s, invoices=(s.invoices[0], inv, *s.invoices[2:]))
    err(s, call("refund_payment", payment_id="PAY-00002", reason="r"), E.INVALID_STATUS)


# --- adjust_invoice -------------------------------------------------------------------------


def test_adjust_down(s):
    new = ok(s, call("adjust_invoice", invoice_id="INV-00001", new_amount=4999, reason="Fix"))
    inv = new.invoices[0]
    assert (inv.amount_billed, inv.amount_due, inv.status, inv.note) == (
        4999,
        4999,
        InvoiceStatus.OPEN,
        "Fix",
    )
    assert inv.line_items == s.invoices[0].line_items
    assert changed(s, new) == {"invoices": [0]}
    assert new.ledger[-1] == LedgerEntry("L0016", "C0001", "INV-00001", None, LedgerKind.ADJUSTMENT, -1000, "Fix")


def test_adjust_paid_invoice_up_reopens(s):
    new = ok(s, call("adjust_invoice", invoice_id="INV-00004", new_amount=3500, reason="r"))
    inv = invoice(new, "INV-00004")
    assert (inv.amount_billed, inv.amount_due, inv.status) == (3500, 500, InvoiceStatus.OPEN)
    assert new.ledger[-1].amount == 500


def test_adjust_to_zero(s):
    new = ok(s, call("adjust_invoice", invoice_id="INV-00001", new_amount=0, reason="r"))
    inv = new.invoices[0]
    assert (inv.amount_billed, inv.amount_due, inv.status) == (0, 0, InvoiceStatus.PAID)


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"invoice_id": "INV-99999", "new_amount": 1}, E.NOT_FOUND),
        ({"invoice_id": "INV-00001", "new_amount": -1}, E.INVALID_AMOUNT),
        ({"invoice_id": "INV-00001", "new_amount": 5999}, E.NO_CHANGE),
    ],
)
def test_adjust_errors(s, args, code):
    err(s, call("adjust_invoice", reason="r", **args), code)


def test_adjust_void(s):
    s = ok(s, call("void_invoice", invoice_id="INV-00001", reason="r"))
    err(
        s,
        call("adjust_invoice", invoice_id="INV-00001", new_amount=1, reason="r"),
        E.INVALID_STATUS,
    )


# --- update_customer ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "stored"),
    [
        ("email", " alice.smith@example.org ", "alice.smith@example.org"),
        ("phone", "+1-415-555-0199", "+1-415-555-0199"),
        ("phone", "415 555 0199", "415 555 0199"),
        ("address", "9 Elm Rd, Springfield", "9 Elm Rd, Springfield"),
        ("name", "Alice Smith-Jones", "Alice Smith-Jones"),
    ],
)
def test_update_customer(s, field, value, stored):
    new = ok(s, call("update_customer", customer_id="C0001", field=field, value=value))
    assert getattr(new.customers[0], field) == stored
    assert new.customers[0] == replace(s.customers[0], **{field: stored})
    assert changed(s, new) == {"customers": [0]}
    assert new.ledger == s.ledger


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"customer_id": "C9999", "field": "email", "value": "a@b.co"}, E.NOT_FOUND),
        ({"customer_id": "C0001", "field": "contract_rate", "value": "1"}, E.FIELD_NOT_EDITABLE),
        ({"customer_id": "C0001", "field": "id", "value": "C0003"}, E.FIELD_NOT_EDITABLE),
        ({"customer_id": "C0001", "field": "Email", "value": "a@b.co"}, E.FIELD_NOT_EDITABLE),
        ({"customer_id": "C0001", "field": "name", "value": "   "}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "address", "value": ""}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "email", "value": "not-an-email"}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "email", "value": "a@b"}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "email", "value": "a b@c.de"}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "phone", "value": "12345"}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "phone", "value": "+1-415-555-01x9"}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "phone", "value": "-1-415-555-0199"}, E.INVALID_VALUE),
        ({"customer_id": "C0001", "field": "email", "value": "alice@example.com"}, E.NO_CHANGE),
        ({"customer_id": "C0001", "field": "name", "value": " Alice Smith "}, E.NO_CHANGE),
    ],
)
def test_update_customer_errors(s, args, code):
    err(s, call("update_customer", **args), code)


# --- resolve_case ---------------------------------------------------------------------------


def test_resolve_case_with_adjustment(s):
    new = ok(s, call("resolve_case", case_id="CS-0001", adjustment=1500, note="Approved"))
    case, inv = new.cases[0], new.invoices[2]
    assert (case.status, case.resolution, case.note) == (CaseStatus.RESOLVED, 1500, "Approved")
    assert (inv.amount_due, inv.status, inv.note) == (5000, InvoiceStatus.OPEN, "")
    assert changed(s, new) == {"invoices": [2], "cases": [0]}
    assert new.ledger[len(s.ledger) :] == (
        LedgerEntry("L0016", "C0002", "INV-00003", None, LedgerKind.DISPUTE_CREDIT, -1500, "Approved"),
    )


def test_resolve_case_zero_adjustment_no_ledger(s):
    new = ok(s, call("resolve_case", case_id="CS-0001", adjustment=0, note="Rejected"))
    assert (new.cases[0].status, new.cases[0].resolution) == (CaseStatus.RESOLVED, 0)
    assert changed(s, new) == {"cases": [0]}
    assert new.ledger == s.ledger


def test_resolve_case_zero_on_paid_invoice_allowed(s):
    new = ok(s, call("resolve_case", case_id="CS-0003", adjustment=0, note=""))
    assert new.cases[2].resolution == 0 and new.cases[2].note == ""


def test_resolve_case_full_amount_due(s):
    new = ok(s, call("resolve_case", case_id="CS-0001", adjustment=6500, note="n"))
    assert new.invoices[2].status == InvoiceStatus.PAID


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"case_id": "CS-9999", "adjustment": 1}, E.NOT_FOUND),
        ({"case_id": "CS-0002", "adjustment": 0}, E.INVALID_STATUS),  # already resolved
        ({"case_id": "CS-0003", "adjustment": 1}, E.INVALID_STATUS),  # invoice paid
        ({"case_id": "CS-0001", "adjustment": -1}, E.INVALID_AMOUNT),
        ({"case_id": "CS-0003", "adjustment": -1}, E.INVALID_AMOUNT),  # status ok when adj <= 0
        ({"case_id": "CS-0001", "adjustment": 6501}, E.EXCEEDS_AMOUNT_DUE),
    ],
)
def test_resolve_case_errors(s, args, code):
    err(s, call("resolve_case", note="n", **args), code)


def test_resolve_case_invoice_missing(s):
    s = replace(s, cases=(replace(s.cases[0], invoice_id="INV-99999"), *s.cases[1:]))
    err(s, call("resolve_case", case_id="CS-0001", adjustment=0, note="n"), E.NOT_FOUND)


def test_resolve_case_zero_on_overpaid_invoice_exceeds(s):
    # Literal SPEC reading: 0 > amount_due (-4999) -> exceeds_amount_due.
    case = replace(s.cases[2], invoice_id="INV-00002", customer_id="C0001")
    s = replace(s, cases=(*s.cases[:2], case))
    err(s, call("resolve_case", case_id="CS-0003", adjustment=0, note="n"), E.EXCEEDS_AMOUNT_DUE)


# --- void_invoice ---------------------------------------------------------------------------


def test_void_invoice(s):
    new = ok(s, call("void_invoice", invoice_id="INV-00001", reason="Issued in error"))
    inv = new.invoices[0]
    assert (inv.amount_billed, inv.amount_due, inv.status, inv.note) == (
        5999,
        0,
        InvoiceStatus.VOID,
        "Issued in error",
    )
    assert changed(s, new) == {"invoices": [0]}
    assert new.ledger[-1] == LedgerEntry("L0016", "C0001", "INV-00001", None, LedgerKind.VOID, -5999, "Issued in error")


def test_void_invoice_with_only_failed_payment(s):
    new = ok(s, call("void_invoice", invoice_id="INV-00007", reason="r"))
    assert invoice(new, "INV-00007").status == InvoiceStatus.VOID


@pytest.mark.parametrize(
    ("invoice_id", "code"),
    [
        ("INV-99999", E.NOT_FOUND),
        ("INV-00002", E.INVALID_STATUS),  # paid
        ("INV-00006", E.HAS_ACTIVITY),  # succeeded payment
    ],
)
def test_void_errors(s, invoice_id, code):
    err(s, call("void_invoice", invoice_id=invoice_id, reason="r"), code)


def test_void_after_credit_has_activity(s):
    s = ok(s, call("apply_credit", invoice_id="INV-00001", amount=1, reason="r"))
    err(s, call("void_invoice", invoice_id="INV-00001", reason="r"), E.HAS_ACTIVITY)


def test_void_already_void(s):
    s = ok(s, call("void_invoice", invoice_id="INV-00001", reason="r"))
    err(s, call("void_invoice", invoice_id="INV-00001", reason="r"), E.INVALID_STATUS)


def test_void_refunded_payment_counts_as_activity(s):
    # Open invoice whose only payment was refunded: amount_due == amount_billed, still activity.
    s = ok(s, call("refund_payment", payment_id="PAY-00004", reason="r"))
    assert invoice(s, "INV-00004").status == InvoiceStatus.OPEN
    err(s, call("void_invoice", invoice_id="INV-00004", reason="r"), E.HAS_ACTIVITY)


# --- argument checking, unknown tool, never raises ------------------------------------------

VALID = {
    "apply_credit": {"invoice_id": "INV-00001", "amount": 1, "reason": "r"},
    "refund_payment": {"payment_id": "PAY-00002", "reason": "r"},
    "adjust_invoice": {"invoice_id": "INV-00001", "new_amount": 1, "reason": "r"},
    "update_customer": {"customer_id": "C0001", "field": "name", "value": "Al"},
    "resolve_case": {"case_id": "CS-0001", "adjustment": 1, "note": "n"},
    "void_invoice": {"invoice_id": "INV-00001", "reason": "r"},
}


def test_valid_table_covers_every_tool(s):
    assert set(VALID) == set(TOOL_NAMES)
    for name, args in VALID.items():
        ok(s, ToolCall(name, dict(args)))


@pytest.mark.parametrize("name", ["", "Apply_credit", "delete_customer", "apply_credit "])
def test_unknown_tool(s, name):
    err(s, ToolCall(name, {}), E.UNKNOWN_TOOL)


@pytest.mark.parametrize("bad", [None, "apply_credit", 42, object()])
def test_non_toolcall_input(s, bad):
    new, res = apply_tool(s, bad)
    assert new is s and res.ok is False and res.error == E.UNKNOWN_TOOL


def _bad_variants(name: str):
    args = VALID[name]
    first = next(iter(args))
    yield "missing", {k: v for k, v in args.items() if k != first}
    yield "extra", {**args, "extra": "x"}
    yield "empty", {}
    for k, typ in PARAM_TYPES[name].items():
        if typ is int:
            yield f"{k}=bool", {**args, k: True}
            yield f"{k}=str", {**args, k: "1"}
            yield f"{k}=float", {**args, k: 1.0}
        else:
            yield f"{k}=int", {**args, k: 1}
            yield f"{k}=None", {**args, k: None}


@pytest.mark.parametrize(
    ("name", "label", "args"),
    [(n, label, a) for n in VALID for label, a in _bad_variants(n)],
)
def test_bad_arguments(s, name, label, args):
    err(s, ToolCall(name, args), E.BAD_ARGUMENTS)


def test_arguments_not_a_dict(s):
    not_a_dict = [("invoice_id", "INV-00001")]
    err(s, ToolCall("apply_credit", not_a_dict), E.BAD_ARGUMENTS)  # pyright: ignore[reportArgumentType]


def test_bad_arguments_precede_lookup(s):
    err(s, call("apply_credit", invoice_id="INV-99999", amount="1", reason="r"), E.BAD_ARGUMENTS)


def test_ids_are_stripped_before_lookup(s):
    new = ok(s, call("apply_credit", invoice_id="  INV-00001\n", amount=1, reason="r"))
    assert new.invoices[0].amount_due == 5998


def test_corrupt_state_never_raises():
    c = call("apply_credit", invoice_id="x", amount=1, reason="r")
    new, res = apply_tool(None, c)  # pyright: ignore[reportArgumentType]
    assert new is None and res.ok is False


def test_duplicate_ids_act_on_first(s):
    dup = replace(s.invoices[0], note="second")
    s2 = replace(s, invoices=(*s.invoices, dup))
    new, res = apply_tool(s2, call("adjust_invoice", invoice_id="INV-00001", new_amount=1, reason="r"))
    assert res.ok
    assert new.invoices[0].amount_billed == 1 and new.invoices[-1] == dup


def test_ledger_id_continues_after_existing_entries(s):
    s = ok(s, call("apply_credit", invoice_id="INV-00001", amount=1, reason="a"))
    s = ok(s, call("resolve_case", case_id="CS-0001", adjustment=1, note="b"))
    assert [e.id for e in s.ledger[-2:]] == ["L0016", "L0017"]
