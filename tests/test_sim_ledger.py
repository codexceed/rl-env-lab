"""Ledger construction, L-rules and S-rules (SPEC §1, §2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from envlab.sim.entities import (
    CaseStatus,
    Invoice,
    InvoiceStatus,
    LedgerEntry,
    LedgerKind,
    LineItem,
    PaymentStatus,
    State,
)
from envlab.sim.ledger import (
    build_ledger,
    check_invariants,
    check_ledger,
    derive_invoice_status,
    next_ledger_id,
)
from sim_factory import customers, invoices, make_state, payments

K = LedgerKind


def _inv(status: InvoiceStatus, due: int) -> Invoice:
    return Invoice("INV-1", "C1", "2026-08", (LineItem("Plan: x", 1, 100),), 100, due, status)


def _with(state: State, group: str, i: int, **changes) -> State:
    records = getattr(state, group)
    return replace(state, **{group: (*records[:i], replace(records[i], **changes), *records[i + 1 :])})


def _has(violations: list[str], prefix: str) -> bool:
    return any(v.startswith(prefix) for v in violations)


# --- derive_invoice_status / next_ledger_id -------------------------------------------------


@pytest.mark.parametrize(
    ("status", "due", "want"),
    [
        (InvoiceStatus.OPEN, 1, InvoiceStatus.OPEN),
        (InvoiceStatus.OPEN, 0, InvoiceStatus.PAID),
        (InvoiceStatus.PAID, -5, InvoiceStatus.PAID),
        (InvoiceStatus.PAID, 5, InvoiceStatus.OPEN),
        (InvoiceStatus.VOID, 0, InvoiceStatus.VOID),
        (InvoiceStatus.VOID, 7, InvoiceStatus.VOID),
    ],
)
def test_derive_invoice_status(status, due, want):
    assert derive_invoice_status(_inv(status, due)) == want


def test_next_ledger_id():
    assert next_ledger_id(State()) == "L0001"
    assert next_ledger_id(make_state()) == "L0016"


# --- build_ledger ---------------------------------------------------------------------------


def test_build_ledger_order_and_contents():
    got = [(e.id, e.customer_id, e.invoice_id, e.payment_id, e.kind, e.amount, e.memo) for e in make_state().ledger]
    assert got == [
        ("L0001", "C0001", "INV-00001", None, K.CHARGE, 5999, "Invoice issued"),
        ("L0002", "C0001", "INV-00002", None, K.CHARGE, 4999, "Invoice issued"),
        # PAY-00001 precedes PAY-00002 by created_at, despite tuple order.
        ("L0003", "C0001", "INV-00002", "PAY-00001", K.PAYMENT, -4999, "Payment received"),
        ("L0004", "C0001", "INV-00002", "PAY-00002", K.PAYMENT, -4999, "Payment received"),
        ("L0005", "C0001", "INV-00006", None, K.CHARGE, 10000, "Invoice issued"),
        ("L0006", "C0001", "INV-00006", "PAY-00007", K.PAYMENT, -4000, "Payment received"),
        # account_credit comes first for its customer.
        ("L0007", "C0002", None, None, K.ACCOUNT_CREDIT, -500, "Account credit"),
        ("L0008", "C0002", "INV-00003", None, K.CHARGE, 6500, "Invoice issued"),
        ("L0009", "C0002", "INV-00004", None, K.CHARGE, 3000, "Invoice issued"),
        # PAY-00003 failed: no entry.
        ("L0010", "C0002", "INV-00004", "PAY-00004", K.PAYMENT, -3000, "Payment received"),
        ("L0011", "C0002", "INV-00005", None, K.CHARGE, 2000, "Invoice issued"),
        ("L0012", "C0002", "INV-00005", "PAY-00005", K.PAYMENT, -2000, "Payment received"),
        ("L0013", "C0002", "INV-00005", "PAY-00006", K.PAYMENT, -2000, "Payment received"),
        ("L0014", "C0002", "INV-00005", "PAY-00006", K.REFUND, 2000, "Refund duplicate payment"),
        # PAY-00008 failed: only the charge.
        ("L0015", "C0002", "INV-00007", None, K.CHARGE, 1000, "Invoice issued"),
    ]


def test_build_ledger_is_deterministic_and_balances():
    assert build_ledger(customers(), invoices(), payments()) == make_state().ledger
    assert check_ledger(make_state()) == []
    assert check_invariants(make_state()) == []


def test_build_ledger_empty():
    assert build_ledger((), (), ()) == ()
    assert check_invariants(State()) == []


def test_build_ledger_follows_customer_order():
    rev = build_ledger(customers()[::-1], invoices(), payments())
    assert rev[0].kind == K.ACCOUNT_CREDIT and rev[0].customer_id == "C0002"
    assert [e.id for e in rev] == [f"L{i:04d}" for i in range(1, len(rev) + 1)]


# --- L-rules --------------------------------------------------------------------------------


def test_l1_invoice_amount_due_mismatch():
    s = _with(make_state(), "invoices", 0, amount_due=5998)
    v = check_ledger(s)
    assert _has(v, "L1: invoice INV-00001")
    assert len(v) == 1


def test_l1_missing_charge():
    s = make_state()
    s = replace(s, ledger=build_ledger(s.customers, s.invoices[1:], s.payments))
    assert _has(check_ledger(s), "L1: invoice INV-00001")


def test_l2_credit_balance_mismatch():
    s = _with(make_state(), "customers", 1, credit_balance=400)
    v = check_ledger(s)
    assert v == [v[0]] and _has(v, "L2: customer C0002")


def test_l2_zero_credit_needs_no_entry():
    s = _with(make_state(), "customers", 0, credit_balance=1)
    assert _has(check_ledger(s), "L2: customer C0001")


def test_l3_entry_customer_differs_from_invoice_and_payment():
    s = make_state()
    s = _with(s, "ledger", 2, customer_id="C0002")  # L0003: INV-00002 / PAY-00001 of C0001
    v = check_ledger(s)
    assert _has(v, "L3: entry L0003 customer C0002 != invoice INV-00002")
    assert _has(v, "L3: entry L0003 customer C0002 != payment PAY-00001")


def test_l3_dangling_references():
    s = make_state()
    s = _with(s, "ledger", 0, invoice_id="INV-99999")
    s = _with(s, "ledger", 2, payment_id="PAY-99999")
    v = check_ledger(s)
    assert _has(v, "L3: entry L0001 references unknown invoice INV-99999")
    assert _has(v, "L3: entry L0003 references unknown payment PAY-99999")


def test_l4_succeeded_payment_without_entry():
    s = make_state()
    s = replace(s, ledger=tuple(e for e in s.ledger if e.payment_id != "PAY-00007"))
    s = replace(s, ledger=tuple(replace(e, id=f"L{i:04d}") for i, e in enumerate(s.ledger, 1)))
    v = check_ledger(s)
    assert _has(v, "L4: payment PAY-00007")


def test_l4_payment_entry_wrong_amount():
    s = _with(make_state(), "ledger", 5, amount=-3999)  # L0006 for PAY-00007
    assert _has(check_ledger(s), "L4: payment PAY-00007")


def test_l4_refunded_payment_without_refund_entry():
    s = _with(make_state(), "ledger", 13, kind=K.CREDIT)  # L0014 refund -> credit
    assert _has(check_ledger(s), "L4: refunded payment PAY-00006")


def test_l4_status_flipped_to_failed_keeps_entries():
    # SPEC §8 env_bug F2: duplicate status = failed, ledger unchanged.
    s = _with(make_state(), "payments", 0, status=PaymentStatus.FAILED)
    assert _has(check_ledger(s), "L4: failed payment PAY-00002")


def test_l4_succeeded_payment_with_refund_entry():
    s = make_state()
    extra = LedgerEntry("L0016", "C0001", "INV-00006", "PAY-00007", K.REFUND, 4000, "x")
    s = replace(s, ledger=(*s.ledger, extra))
    assert _has(check_ledger(s), "L4: succeeded payment PAY-00007")


def test_l5_ids_out_of_order():
    s = make_state()
    led = list(s.ledger)
    led[0], led[1] = led[1], led[0]
    v = check_ledger(replace(s, ledger=tuple(led)))
    assert _has(v, "L5: entry at position 1 has id L0002")
    assert _has(v, "L5: entry at position 2 has id L0001")


def test_l5_gap_in_ids():
    s = _with(make_state(), "ledger", 14, id="L0099")
    assert check_ledger(s) == ["L5: entry at position 15 has id L0099, expected L0015"]


# --- S-rules --------------------------------------------------------------------------------


def test_s1_void_with_amount_due():
    # SPEC §8 env_bug F1/F4: status = void, amount_due unchanged.
    s = _with(make_state(), "invoices", 0, status=InvoiceStatus.VOID)
    v = check_invariants(s)
    assert _has(v, "S1: void invoice INV-00001")
    assert check_ledger(s) == []  # balance alone does not catch it


def test_s1_status_disagrees_with_amount_due():
    assert _has(
        check_invariants(_with(make_state(), "invoices", 0, status=InvoiceStatus.PAID)),
        "S1: invoice INV-00001",
    )
    assert _has(
        check_invariants(_with(make_state(), "invoices", 3, status=InvoiceStatus.OPEN)),
        "S1: invoice INV-00004",
    )


def test_s2_resolved_without_resolution():
    s = _with(make_state(), "cases", 0, status=CaseStatus.RESOLVED)
    assert _has(check_invariants(s), "S2: case CS-0001")


def test_s2_open_with_resolution():
    s = _with(make_state(), "cases", 0, resolution=5)
    assert _has(check_invariants(s), "S2: case CS-0001")


def test_s2_negative_resolution():
    s = _with(make_state(), "cases", 1, resolution=-1)
    assert _has(check_invariants(s), "S2: case CS-0002 resolution -1 < 0")


def test_s3_duplicate_customer_id():
    # SPEC §8 env_bug F5: append a second customer with C's id.
    s = make_state()
    s = replace(s, customers=(*s.customers, replace(s.customers[0], name="Other")))
    assert _has(check_invariants(s), "S3: duplicate customer id C0001")


@pytest.mark.parametrize("group", ["invoices", "payments", "cases"])
def test_s3_duplicate_ids(group):
    s = make_state()
    records = getattr(s, group)
    s = replace(s, **{group: (*records, records[-1])})
    assert _has(check_invariants(s), "S3: duplicate")


def test_s3_dangling_references():
    s = make_state()
    s = _with(s, "cases", 0, invoice_id="INV-99999")
    s = _with(s, "payments", 6, customer_id="C9999")
    v = check_invariants(s)
    assert _has(v, "S3: case CS-0001 references unknown invoice INV-99999")
    assert _has(v, "S3: payment PAY-00007 references unknown customer C9999")


def test_s3_dangling_invoice_customer_and_ledger_customer():
    s = make_state()
    s = replace(s, customers=s.customers[:1])
    v = check_invariants(s)
    assert _has(v, "S3: invoice INV-00003 references unknown customer C0002")
    assert _has(v, "S3: ledger entry L0007 references unknown customer C0002")


def test_s3_payment_and_case_customer_mismatch():
    s = make_state()
    s = _with(s, "payments", 6, customer_id="C0002")
    s = _with(s, "cases", 0, customer_id="C0001")
    v = check_invariants(s)
    assert _has(v, "S3: payment PAY-00007 customer C0002 != invoice INV-00006")
    assert _has(v, "S3: case CS-0001 customer C0001 != invoice INV-00003")


def test_s4_ranges():
    s = make_state()
    s = _with(s, "customers", 0, credit_balance=-1)
    s = _with(s, "invoices", 0, amount_billed=-1)
    s = _with(s, "invoices", 1, line_items=(LineItem("Plan: Pro", 0, 4999),))
    s = _with(s, "payments", 6, amount=0)
    s = _with(s, "cases", 0, disputed_amount=0)
    v = check_invariants(s)
    for prefix in (
        "S4: customer C0001",
        "S4: invoice INV-00001 amount_billed",
        "S4: invoice INV-00002 line",
        "S4: payment PAY-00007",
        "S4: case CS-0001",
    ):
        assert _has(v, prefix), prefix
