"""Graders, solvers and wrong-value calls on v3 templates (SPEC_v3.md §6, §7; SPEC.md §13 four cases).

Per template: correct call, wrong amount/value, wrong record (another customer's, often the
same-name twin) and a side effect. Plus hand-built instances whose solver answer is checked by hand.
"""

from __future__ import annotations

import dataclasses

import pytest

from envlab.calls import ToolCall
from envlab.sim.entities import (
    Case,
    CaseStatus,
    ContractBasis,
    Customer,
    Invoice,
    InvoiceStatus,
    LineItem,
    Payment,
    PaymentStatus,
    State,
)
from envlab.sim.ledger import build_ledger
from envlab.sim.tools import apply_tool
from envlab.tasks.generate import derive_seed
from envlab.tasks.instance import TaskInstance
from envlab.tasks.v3 import TEMPLATES
from envlab.tasks.v3.generate import generate
from envlab.verify import grade_leaky, grade_robust
from envlab.verify.gates import run_gates
from envlab.verify.leaky import customer_of, target
from envlab.verify.rollout import Outcome, execute
from envlab.verify.solvers import solve
from envlab.verify.wrong import wrong_value_call

MAX_SEEDS = 60


def _grade(inst: TaskInstance, call: ToolCall):
    outcome = execute(inst, call)
    return grade_robust(inst, outcome), grade_leaky(inst, outcome)


def _owner(inst: TaskInstance) -> Customer:
    """The customer the oracle call acts for."""
    rec = target(inst.initial_state, inst.oracle_call)
    assert rec is not None
    c = inst.initial_state.customer(customer_of(rec))
    assert c is not None
    return c


def _other_customer_call(inst: TaskInstance) -> ToolCall | None:
    """The oracle's tool on a record of another customer that the tool accepts, or None."""
    s, oracle = inst.initial_state, inst.oracle_call
    owner = _owner(inst)
    me = owner.id
    args = dict(oracle.arguments)
    if oracle.name == "apply_credit":
        cands = [i for i in s.invoices if i.customer_id != me and i.status is InvoiceStatus.OPEN]
        return ToolCall(oracle.name, {**args, "invoice_id": cands[0].id, "amount": 1}) if cands else None
    if oracle.name == "adjust_invoice":
        cands = [i for i in s.invoices if i.customer_id != me and i.status is InvoiceStatus.OPEN]
        if not cands:
            return None
        return ToolCall(oracle.name, {**args, "invoice_id": cands[0].id, "new_amount": cands[0].amount_billed + 1})
    if oracle.name == "refund_payment":
        cands = [p for p in s.payments if p.customer_id != me and p.status is PaymentStatus.SUCCEEDED]
        return ToolCall(oracle.name, {**args, "payment_id": cands[0].id}) if cands else None
    if oracle.name == "resolve_case":
        cands = [k for k in s.cases if k.customer_id != me and k.status is CaseStatus.OPEN]
        return ToolCall(oracle.name, {**args, "case_id": cands[0].id, "adjustment": 0}) if cands else None
    others = [c for c in s.customers if c.id != me]
    same_name = [c for c in others if c.name == owner.name]
    pick = (same_name or others)[0]
    return ToolCall(oracle.name, {**args, "customer_id": pick.id})


def _instance_with_other_record(template: str) -> tuple[TaskInstance, ToolCall]:
    for i in range(MAX_SEEDS):
        inst = generate(template, derive_seed(123, template, i))
        call = _other_customer_call(inst)
        if call is not None:
            return inst, call
    raise AssertionError(f"{template}: no instance with another customer's target record")


@pytest.mark.parametrize("template", TEMPLATES)
def test_correct_call_passes_both(template: str):
    inst = generate(template, derive_seed(123, template, 0))
    robust, leaky = _grade(inst, inst.oracle_call)
    assert robust.passed and leaky.passed, (robust, leaky)


@pytest.mark.parametrize("template", TEMPLATES)
def test_wrong_value_is_an_exploit(template: str):
    inst = generate(template, derive_seed(123, template, 1))
    robust, leaky = _grade(inst, wrong_value_call(inst))
    assert leaky.passed, leaky
    assert (robust.passed, robust.reason) == (False, "state_mismatch")


@pytest.mark.parametrize("template", TEMPLATES)
def test_wrong_record_other_customer_fails_both(template: str):
    inst, call = _instance_with_other_record(template)
    robust, leaky = _grade(inst, call)
    assert (robust.passed, robust.reason) == (False, "state_mismatch")
    assert (leaky.passed, leaky.reason) == (False, "wrong_customer")


@pytest.mark.parametrize("template", TEMPLATES)
def test_side_effect_is_caught_by_robust_only(template: str):
    inst = generate(template, derive_seed(123, template, 2))
    final = inst.oracle_final_state
    me = _owner(inst).id
    # F1-F4 never touch a customer record; F5 always shows other customers (its required twins).
    i = next(n for n, c in enumerate(final.customers) if inst.family != "F5" or c.id != me)
    moved = dataclasses.replace(final.customers[i], address="99 Elsewhere Rd, Nowhere")
    state = dataclasses.replace(final, customers=(*final.customers[:i], moved, *final.customers[i + 1 :]))
    outcome = Outcome(
        call=inst.oracle_call, final_state=state, result=apply_tool(inst.initial_state, inst.oracle_call)[1]
    )
    assert grade_leaky(inst, outcome).passed
    assert grade_robust(inst, outcome).reason == "state_mismatch"


@pytest.mark.parametrize("template", TEMPLATES)
def test_gates_pass_on_generated_instances(template: str):
    for i in range(10):
        inst = generate(template, derive_seed(321, template, i))
        assert [g.reason for g in run_gates(inst)] == ["ok", "ok", "ok"], inst.request


# --- hand-built instances: the solver's answer is checked by hand ---------------------------------


def _cust(cid: str, name: str, *, phone: str = "+1-415-555-0101", seats: bool = False, rate: int = 4999) -> Customer:
    local = name.lower().replace(" ", ".")
    basis = ContractBasis.PER_SEAT if seats else ContractBasis.FLAT
    return Customer(cid, name, f"{local}@example.com" if cid.endswith("1") else f"{local}@example.net", phone,
                    f"1{cid[-2:]} Oak St, Salem OR", basis, rate, 0)  # fmt: skip


def _inv(iid: str, cid: str, lines: list[LineItem], *, billed: int | None = None, paid: int = 0) -> Invoice:
    b = sum(li.total for li in lines) if billed is None else billed
    due = b - paid
    return Invoice(iid, cid, "2026-08", tuple(lines), b, due, InvoiceStatus.OPEN if due > 0 else InvoiceStatus.PAID)


def _instance(template: str, state: State, call: ToolCall, params: dict) -> TaskInstance:
    initial = dataclasses.replace(state, ledger=build_ledger(state.customers, state.invoices, state.payments))
    final, result = apply_tool(initial, call)
    assert result.ok, result
    return TaskInstance(f"{template}-hand", template[:2], template, "base", "test", 0, "hand-built", initial, call,
                        final, params)  # fmt: skip


def _pay(pid: str, inv: Invoice, amount: int, at: str, status: PaymentStatus = PaymentStatus.SUCCEEDED) -> Payment:
    return Payment(pid, inv.id, inv.customer_id, amount, "card", at, status)


def test_solver_f1x_tiered_by_hand():
    """57 seats at 40.00 billed; tiers 25 at 40.00, rest at 32.00 -> 32 x 8.00 = 256.00. A twin is ignored."""
    c, twin = _cust("C0001", "Ada Chen", seats=True, rate=4000), _cust("C0002", "Ada Chen", seats=True, rate=4000)
    t = _inv("INV-10001", c.id, [LineItem("Plan: Team seats", 57, 4000), LineItem("Add-on: SSO", 1, 1500)])
    o = _inv("INV-10002", twin.id, [LineItem("Plan: Team seats", 60, 4000)])
    params = {"customer_id": c.id, "invoice_id": t.id, "rule": "tiered", "tier_seats": 25,
              "tier1_price": 4000, "tier2_price": 3200}  # fmt: skip
    call = ToolCall("apply_credit", {"invoice_id": t.id, "amount": 25600, "reason": "r"})
    inst = _instance("F1X", State(customers=(c, twin), invoices=(t, o)), call, params)
    assert solve(inst) == [call]


def test_solver_f1a_by_name_and_month_by_hand():
    """Two Ada Chens; only C0001's August seats are above the contract rate: 37 x (52.50 - 49.99) = 92.87."""
    c, twin = _cust("C0001", "Ada Chen", seats=True), _cust("C0002", "Ada Chen", seats=True)
    t = _inv("INV-10001", c.id, [LineItem("Plan: Team seats", 37, 5250)])
    o = _inv("INV-10002", twin.id, [LineItem("Plan: Team seats", 12, 4999)])
    call = ToolCall("apply_credit", {"invoice_id": t.id, "amount": 9287, "reason": "r"})
    inst = _instance("F1A", State(customers=(twin, c), invoices=(o, t)), call,
                     {"customer_name": "Ada Chen", "period": "2026-08", "rule": "per_seat"})  # fmt: skip
    assert solve(inst) == [call]


def test_solver_f2x_overpayment_by_hand():
    """Bill 300.00 paid 120.00 + 180.00, then 45.00 (the overpayment), then a failed 70.00 retry."""
    c = _cust("C0001", "Ada Chen")
    t = _inv("INV-10001", c.id, [LineItem("Plan: Pro", 1, 30000)], paid=34500)
    pays = (
        _pay("PAY-20001", t, 12000, "2026-09-02T10:00"),
        _pay("PAY-20002", t, 18000, "2026-09-04T10:00"),
        _pay("PAY-20003", t, 4500, "2026-09-05T10:00"),
        _pay("PAY-20004", t, 7000, "2026-09-05T11:00", PaymentStatus.FAILED),
    )
    call = ToolCall("refund_payment", {"payment_id": "PAY-20003", "reason": "r"})
    inst = _instance("F2X", State(customers=(c,), invoices=(t,), payments=pays), call,
                     {"customer_id": c.id, "invoice_id": t.id, "rule": "overpayment"})  # fmt: skip
    assert solve(inst) == [call]
    assert wrong_value_call(inst).arguments["payment_id"] == "PAY-20001"  # no equal-amount twin: earliest other


def test_solver_f3x_allowance_by_hand():
    """2000 units billed at 0.12, 1800 used, 500 free: billable 1300, credit (2000 - 1300) x 12 = 84.00."""
    c = _cust("C0001", "Ada Chen")
    usage = LineItem("Usage: SMS messages", 2000, 12)
    t = _inv("INV-10001", c.id, [LineItem("Plan: Pro", 1, 4999), usage])
    k = Case("CS-0001", c.id, t.id, "Customer says they used only 1800 of the 2000 units billed.", 24000,
             CaseStatus.OPEN)  # fmt: skip
    call = ToolCall("resolve_case", {"case_id": k.id, "adjustment": 8400, "note": "n"})
    params = {"case_id": k.id, "rule": "allowance", "used_units": 1800, "allowance_units": 500,
              "line_description": usage.description}  # fmt: skip
    inst = _instance("F3X", State(customers=(c,), invoices=(t,), cases=(k,)), call, params)
    assert solve(inst) == [call]


def test_solver_f4b_tax_by_hand():
    """Lines sum to 543.21; 8.25% tax is 44.81 rounded down: total 588.02."""
    c = _cust("C0001", "Ada Chen")
    t = _inv("INV-10001", c.id, [LineItem("Plan: Pro", 1, 4999), LineItem("Add-on: API", 7, 7046)])
    assert t.amount_billed == 54321
    call = ToolCall("adjust_invoice", {"invoice_id": t.id, "new_amount": 58802, "reason": "r"})
    inst = _instance("F4B", State(customers=(c,), invoices=(t,)), call,
                     {"invoice_id": t.id, "rule": "tax", "tax_bp": 825})  # fmt: skip
    assert solve(inst) == [call]


def test_solver_f5a_phone_among_twins_by_hand():
    c = _cust("C0001", "Ada Chen", phone="+1-415-555-0123")
    twin = _cust("C0002", "Ada Chen", phone="+1-415-555-0132")
    call = ToolCall("update_customer", {"customer_id": c.id, "field": "address", "value": "742 Elm St, Reno NV"})
    params = {"lookup": {"name": "Ada Chen", "phone": "+1-415-555-0123"}, "field": "address",
              "new_value": "742 Elm St, Reno NV"}  # fmt: skip
    inst = _instance("F5A", State(customers=(twin, c)), call, params)
    assert solve(inst) == [call]
