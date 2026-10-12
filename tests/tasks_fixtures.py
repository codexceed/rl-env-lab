"""A small hand-built TaskInstance for tests that must run without the simulator."""

from __future__ import annotations

from envlab.calls import ToolCall
from envlab.sim.entities import (
    Case,
    CaseStatus,
    ContractBasis,
    Customer,
    Invoice,
    InvoiceStatus,
    LedgerEntry,
    LedgerKind,
    LineItem,
    State,
)
from envlab.tasks.instance import TaskInstance


def hand_built_instance() -> TaskInstance:
    """F3T1-like instance: one customer, one invoice, one open case, no payments."""
    cust = Customer("C1234", "Ada Chen", "ada.chen@example.com", "+1-415-555-0101", "12 Oak St, Salem OR",
                    ContractBasis.FLAT, 4999, 0)  # fmt: skip
    inv = Invoice(
        "INV-10001",
        "C1234",
        "2026-08",
        (LineItem("Plan: Pro", 1, 4999), LineItem("Add-on: SSO", 2, 1500)),
        7999,
        7999,
        InvoiceStatus.OPEN,
    )
    case = Case("CS-2001", "C1234", "INV-10001", "Customer disputes the 'Add-on: SSO' charge.", 3000, CaseStatus.OPEN)
    ledger = (LedgerEntry("L0001", "C1234", "INV-10001", None, LedgerKind.CHARGE, 7999, "Invoice issued"),)
    initial = State(customers=(cust,), invoices=(inv,), cases=(case,), ledger=ledger)
    resolved = Case(case.id, case.customer_id, case.invoice_id, case.summary, 3000, CaseStatus.RESOLVED, 3000, "ok")
    paid_down = Invoice(inv.id, inv.customer_id, inv.period, inv.line_items, 7999, 4999, InvoiceStatus.OPEN)
    final = State(
        customers=(cust,),
        invoices=(paid_down,),
        cases=(resolved,),
        ledger=(
            *ledger,
            LedgerEntry("L0002", "C1234", "INV-10001", None, LedgerKind.DISPUTE_CREDIT, -3000, "ok"),
        ),
    )
    return TaskInstance(
        id="F3T1-train-00000",
        family="F3",
        template="F3T1",
        variant="base",
        split="train",
        seed=1,
        request="Resolve case CS-2001. Approve the dispute in full.",
        initial_state=initial,
        oracle_call=ToolCall("resolve_case", {"case_id": "CS-2001", "adjustment": 3000, "note": "ok"}),
        oracle_final_state=final,
        params={"case_id": "CS-2001", "rule": "percent", "percent": 100},
        meta={"n_distractors": 0, "target_ids": ["C1234", "INV-10001", "CS-2001"], "spec_version": "1.0"},
    )
