"""Hand-built task instances (one or more per family) and broken variants for verify tests.

Built with the simulator's `build_ledger` and `apply_tool`, so every fixture is a state the real tools produce.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

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
from envlab.tasks.instance import TaskInstance

F1_REASON = "Credit overcharge versus contract rate"
F2_REASON = "Refund duplicate payment"
F3_NOTE = "Dispute resolved per policy"
F4_REASON = "Correct total to sum of line items"

# --- records -----------------------------------------------------------------------------------


def customer(
    cid: str,
    name: str,
    *,
    email: str | None = None,
    phone: str = "+1-415-555-0100",
    address: str = "1 Main St, Springfield",
    basis: ContractBasis = ContractBasis.FLAT,
    rate: int = 4999,
    credit: int = 0,
) -> Customer:
    if email is None:
        email = name.lower().replace(" ", ".") + "@example.com"
    return Customer(cid, name, email, phone, address, basis, rate, credit)


def line(description: str, quantity: int, unit_price: int) -> LineItem:
    return LineItem(description, quantity, unit_price)


def invoice(iid: str, cid: str, period: str, lines: list[LineItem], billed: int | None = None) -> Invoice:
    """Invoice billed at `billed` (default: sum of lines); amount_due/status are set by `state`."""
    total = sum(li.total for li in lines) if billed is None else billed
    return Invoice(iid, cid, period, tuple(lines), total, total, InvoiceStatus.OPEN)


def payment(
    pid: str, iid: str, cid: str, amount: int, at: str, status: PaymentStatus = PaymentStatus.SUCCEEDED
) -> Payment:
    note = "Refund duplicate payment" if status == PaymentStatus.REFUNDED else ""
    return Payment(pid, iid, cid, amount, "card", at, status, note)


def case(kid: str, cid: str, iid: str, summary: str, disputed: int, *, resolved: bool = False, note: str = "") -> Case:
    if resolved:
        return Case(kid, cid, iid, summary, disputed, CaseStatus.RESOLVED, 0, "Dispute rejected: charge is valid")
    return Case(kid, cid, iid, summary, disputed, CaseStatus.OPEN, None, note)


def state(
    customers: Sequence[Customer],
    invoices: Sequence[Invoice],
    payments: Sequence[Payment] = (),
    cases: Sequence[Case] = (),
) -> State:
    """State with amount_due = billed - succeeded payments, derived status, and a fresh ledger."""
    fixed = []
    for inv in invoices:
        if inv.status == InvoiceStatus.VOID:
            fixed.append(inv)
            continue
        paid = sum(p.amount for p in payments if p.invoice_id == inv.id and p.status == PaymentStatus.SUCCEEDED)
        due = inv.amount_billed - paid
        fixed.append(
            dataclasses.replace(inv, amount_due=due, status=InvoiceStatus.OPEN if due > 0 else InvoiceStatus.PAID)
        )
    cs, invs, ps = tuple(customers), tuple(fixed), tuple(payments)
    return State(customers=cs, invoices=invs, payments=ps, cases=tuple(cases), ledger=build_ledger(cs, invs, ps))


def reledger(st: State) -> State:
    return dataclasses.replace(st, ledger=build_ledger(st.customers, st.invoices, st.payments))


def instance(
    template: str, request: str, st: State, oracle: ToolCall, params: dict[str, Any], *, variant: str = "base"
) -> TaskInstance:
    final, result = apply_tool(st, oracle)
    assert result.ok, (template, result)
    return TaskInstance(
        id=f"{template}-test-00000",
        family=template[:2],
        template=template,
        variant=variant,
        split="train" if template[3] != "3" else "heldout",
        seed=0,
        request=request,
        initial_state=st,
        oracle_call=oracle,
        oracle_final_state=final,
        params=params,
        broken_type=None,
        meta={"n_distractors": 3, "target_ids": [], "spec_version": "1.0"},
    )


def call(name: str, **arguments: str | int) -> ToolCall:
    return ToolCall(name, dict(arguments))


# --- F1 ----------------------------------------------------------------------------------------


def f1t1() -> TaskInstance:
    st = state(
        [
            customer("C0001", "Ana Lima"),
            customer("C0002", "Ben Ode", rate=2999),
            customer("C0003", "Ana Park"),
        ],
        [
            invoice("INV-10001", "C0001", "2026-08", [line("Plan: Pro", 1, 5499)]),
            invoice("INV-10002", "C0001", "2026-07", [line("Plan: Pro", 1, 4999)]),
            invoice("INV-10003", "C0001", "2026-06", [line("Plan: Pro", 1, 4999)]),
            invoice("INV-10004", "C0002", "2026-08", [line("Plan: Starter", 1, 2999)]),
        ],
        [payment("PAY-20001", "INV-10003", "C0001", 4999, "2026-07-03T09:15")],
    )
    return instance(
        "F1T1",
        "Customer Ana Lima (C0001) says invoice INV-10001 is higher than their contract rate. Credit the overcharge.",
        st,
        call("apply_credit", invoice_id="INV-10001", amount=500, reason=F1_REASON),
        {"customer_id": "C0001", "invoice_id": "INV-10001"},
    )


def f1t2() -> TaskInstance:
    st = state(
        [customer("C0001", "Maya Chen", rate=2499), customer("C0002", "Omar Haddad")],
        [
            invoice("INV-10011", "C0001", "2026-08", [line("Plan: Basic", 1, 2899), line("Add-on: Storage", 1, 1500)]),
            invoice("INV-10012", "C0001", "2026-07", [line("Plan: Basic", 1, 2499)]),
        ],
    )
    return instance(
        "F1T2",
        "Maya Chen was promised $24.99 per month, but their August 2026 invoice is higher. Credit the difference.",
        st,
        call("apply_credit", invoice_id="INV-10011", amount=400, reason=F1_REASON),
        {"customer_name": "Maya Chen", "period": "2026-08"},
    )


def f1t3() -> TaskInstance:
    st = state(
        [
            customer("C0001", "Lena Varga", basis=ContractBasis.PER_SEAT, rate=1200),
            customer("C0002", "Paul Okoro", rate=3999),
        ],
        [
            invoice(
                "INV-10021",
                "C0001",
                "2026-05",
                [line("Plan: Team seats", 12, 1350), line("Add-on: Priority support", 1, 2500)],
            ),
            invoice("INV-10022", "C0002", "2026-05", [line("Plan: Pro", 1, 3999)]),
        ],
    )
    return instance(
        "F1T3",
        "Lena Varga (C0001) is on a per-seat contract. Invoice INV-10021 charged more per seat than agreed. "
        "Credit the overcharge.",
        st,
        call("apply_credit", invoice_id="INV-10021", amount=1800, reason=F1_REASON),
        {"customer_id": "C0001", "invoice_id": "INV-10021"},
    )


# --- F2 ----------------------------------------------------------------------------------------


def _f2_state() -> State:
    return state(
        [customer("C0001", "Ravi Patel", rate=6999), customer("C0002", "Sofia Rossi", rate=3999)],
        [
            invoice("INV-10101", "C0001", "2026-08", [line("Plan: Growth", 1, 6999)]),
            invoice("INV-10102", "C0001", "2026-07", [line("Plan: Growth", 1, 6999)]),
            invoice("INV-10103", "C0002", "2026-08", [line("Plan: Pro", 1, 3999)]),
        ],
        [
            payment("PAY-20103", "INV-10101", "C0001", 6999, "2026-09-01T08:00", PaymentStatus.FAILED),
            payment("PAY-20101", "INV-10101", "C0001", 6999, "2026-09-02T10:00"),
            payment("PAY-20102", "INV-10101", "C0001", 6999, "2026-09-02T10:07"),
            payment("PAY-20104", "INV-10102", "C0001", 6999, "2026-08-04T12:00"),
            payment("PAY-20105", "INV-10103", "C0002", 3999, "2026-09-05T09:00"),
        ],
    )


def f2t1() -> TaskInstance:
    return instance(
        "F2T1",
        "Ravi Patel (C0001) was charged twice for invoice INV-10101. Refund the duplicate charge.",
        _f2_state(),
        call("refund_payment", payment_id="PAY-20102", reason=F2_REASON),
        {"customer_id": "C0001", "invoice_id": "INV-10101"},
    )


def f2t2() -> TaskInstance:
    return instance(
        "F2T2",
        "Ravi Patel says they were double-charged on one of their invoices. Refund the extra payment.",
        _f2_state(),
        call("refund_payment", payment_id="PAY-20102", reason=F2_REASON),
        {"customer_name": "Ravi Patel"},
    )


def f2t3() -> TaskInstance:
    st = state(
        [customer("C0001", "Jonas Berg", rate=9999), customer("C0002", "Zoe Adams")],
        [invoice("INV-10111", "C0001", "2026-08", [line("Plan: Scale", 1, 9999), line("Add-on: SSO", 1, 2001)])],
        [
            payment("PAY-20114", "INV-10111", "C0001", 7000, "2026-09-02T09:00", PaymentStatus.FAILED),
            payment("PAY-20111", "INV-10111", "C0001", 5000, "2026-09-03T09:00"),
            payment("PAY-20112", "INV-10111", "C0001", 7000, "2026-09-10T09:00"),
            payment("PAY-20113", "INV-10111", "C0001", 7000, "2026-09-10T15:30"),
        ],
    )
    return instance(
        "F2T3",
        "Jonas Berg (C0001) paid invoice INV-10111 in two installments, but one installment went through twice. "
        "Refund the duplicate.",
        st,
        call("refund_payment", payment_id="PAY-20113", reason=F2_REASON),
        {"customer_id": "C0001", "invoice_id": "INV-10111"},
    )


# --- F3 ----------------------------------------------------------------------------------------


def _f3_state() -> State:
    return state(
        [customer("C0001", "Nora Quinn"), customer("C0002", "Tom Reyes", rate=2999)],
        [
            invoice("INV-10201", "C0001", "2026-08", [line("Plan: Pro", 1, 4999), line("Add-on: Backup", 2, 1500)]),
            invoice("INV-10202", "C0001", "2026-07", [line("Plan: Pro", 1, 4999)]),
            invoice("INV-10203", "C0002", "2026-08", [line("Plan: Starter", 1, 2999), line("Add-on: Seats", 1, 1000)]),
        ],
        cases=[
            case("CS-0001", "C0001", "INV-10201", "Customer disputes the 'Add-on: Backup' charge.", 3000),
            case("CS-0002", "C0001", "INV-10202", "Customer disputes the 'Plan: Pro' charge.", 4999, resolved=True),
            case("CS-0003", "C0002", "INV-10203", "Customer disputes the 'Add-on: Seats' charge.", 1000),
            case("CS-0004", "C0001", "INV-10202", "Customer disputes the 'Plan: Pro' charge.", 1000),
        ],
    )


def f3t1() -> TaskInstance:
    return instance(
        "F3T1",
        "Resolve case CS-0001. Approve the dispute in full.",
        _f3_state(),
        call("resolve_case", case_id="CS-0001", adjustment=3000, note=F3_NOTE),
        {"case_id": "CS-0001", "rule": "percent", "percent": 100},
    )


def f3t2() -> TaskInstance:
    return instance(
        "F3T2",
        "Resolve case CS-0001. Policy: approve 25% of the disputed amount, rounded down to the cent.",
        _f3_state(),
        call("resolve_case", case_id="CS-0001", adjustment=750, note=F3_NOTE),
        {"case_id": "CS-0001", "rule": "percent", "percent": 25},
    )


def f3t3() -> TaskInstance:
    usage = "Usage: API calls (per 1k)"
    st = state(
        [customer("C0001", "Iris Novak"), customer("C0002", "Leo Grant")],
        [
            invoice("INV-10211", "C0001", "2026-08", [line("Plan: Pro", 1, 4999), line(usage, 120, 40)]),
            invoice("INV-10212", "C0002", "2026-08", [line("Plan: Pro", 1, 4999), line(usage, 50, 40)]),
        ],
        cases=[
            case("CS-0011", "C0001", "INV-10211", "Customer says they used only 45k API calls, not 120k.", 4800),
            case("CS-0012", "C0002", "INV-10212", "Customer says they used only 20k API calls, not 50k.", 2000),
        ],
    )
    return instance(
        "F3T3",
        "Resolve case CS-0011. Credit only the unused units, at the invoice's unit price.",
        st,
        call("resolve_case", case_id="CS-0011", adjustment=3000, note=F3_NOTE),
        {"case_id": "CS-0011", "rule": "unused_units", "used_units": 45, "line_description": usage},
    )


# --- F4 ----------------------------------------------------------------------------------------


def f4t1() -> TaskInstance:
    st = state(
        [customer("C0001", "Eli Brooks"), customer("C0002", "Gita Rao", rate=2999)],
        [
            invoice("INV-10301", "C0001", "2026-08", [line("Plan: Pro", 1, 4999), line("Add-on: SSO", 1, 2500)], 7999),
            invoice("INV-10302", "C0001", "2026-07", [line("Plan: Pro", 1, 4999)]),
            invoice("INV-10303", "C0002", "2026-08", [line("Plan: Starter", 1, 2999)]),
        ],
    )
    return instance(
        "F4T1",
        "Invoice INV-10301 for Eli Brooks doesn't match its line items. Correct the invoice total.",
        st,
        call("adjust_invoice", invoice_id="INV-10301", new_amount=7499, reason=F4_REASON),
        {"invoice_id": "INV-10301"},
    )


def f4t2() -> TaskInstance:
    lines = [line("Plan: Team", 1, 9999), line("Add-on: Seats", 5, 1200), line("Add-on: Storage", 3, 700)]
    st = state(
        [customer("C0001", "Hana Ito", rate=9999), customer("C0002", "Ivan Petrov")],
        [
            invoice("INV-10311", "C0001", "2026-08", lines, 18909),
            invoice("INV-10312", "C0001", "2026-07", [line("Plan: Team", 1, 9999)]),
        ],
    )
    return instance(
        "F4T2",
        "One of Hana Ito's open invoices has a total that doesn't match its line items. Correct it.",
        st,
        call("adjust_invoice", invoice_id="INV-10311", new_amount=18099, reason=F4_REASON),
        {"customer_name": "Hana Ito"},
    )


def f4t3() -> TaskInstance:
    lines = [line("Plan: Pro", 1, 4999), line("Add-on: Seats", 4, 1000), line("Discount: Loyalty", 1, -900)]
    st = state(
        [customer("C0001", "Mia Clark"), customer("C0002", "Noah Diaz")],
        [invoice("INV-10321", "C0001", "2026-08", lines, 8999)],
    )
    return instance(
        "F4T3",
        "Invoice INV-10321 (Mia Clark) has the wrong total. Recompute it from the line items, including discounts.",
        st,
        call("adjust_invoice", invoice_id="INV-10321", new_amount=8099, reason=F4_REASON),
        {"invoice_id": "INV-10321"},
    )


# --- F5 ----------------------------------------------------------------------------------------


def f5t1() -> TaskInstance:
    st = state(
        [
            customer("C0001", "Ana Lima", phone="+1-415-555-0142"),
            customer("C0002", "Alex Moore", email="ana.lima@example.net"),
            customer("C0003", "Ana Park"),
        ],
        [invoice("INV-10401", "C0001", "2026-08", [line("Plan: Pro", 1, 4999)])],
    )
    return instance(
        "F5T1",
        "Customer C0001 asked to change their email to ana.l@example.org.",
        st,
        call("update_customer", customer_id="C0001", field="email", value="ana.l@example.org"),
        {"customer_id": "C0001", "field": "email", "new_value": "ana.l@example.org"},
    )


def f5t2() -> TaskInstance:
    st = state(
        [
            customer("C0001", "Dan Wu", phone="+1-646-555-0111"),
            customer("C0002", "Dan Wu", email="dwu@example.org", phone="+1-646-555-0122"),
            customer("C0003", "Rosa Lind"),
        ],
        [],
    )
    new = "88 Pine Ave, Riverton"
    return instance(
        "F5T2",
        f"Support inbox message: \"Hi, this is Dan Wu (dan.wu@example.com). I've moved, please update my address "
        f'to {new}."',
        st,
        call("update_customer", customer_id="C0001", field="address", value=new),
        {"lookup": {"name": "Dan Wu", "email": "dan.wu@example.com"}, "field": "address", "new_value": new},
    )


def f5t3() -> TaskInstance:
    st = state(
        [
            customer("C0001", "Kim Sato", phone="+1-212-555-0175"),
            customer("C0002", "Kim Lee", phone="+1-212-555-0188"),
        ],
        [],
    )
    return instance(
        "F5T3",
        'Support inbox message: "Please change the phone number on my account from +1-212-555-0175 to '
        "+1-212-555-0199. If it's urgent today, call my office at +1-212-555-0133. Thanks, Kim\"",
        st,
        call("update_customer", customer_id="C0001", field="phone", value="+1-212-555-0199"),
        {"lookup": {"phone": "+1-212-555-0175"}, "field": "phone", "new_value": "+1-212-555-0199"},
    )


BUILDERS: dict[str, Callable[[], TaskInstance]] = {
    "F1T1": f1t1,
    "F1T2": f1t2,
    "F1T3": f1t3,
    "F2T1": f2t1,
    "F2T2": f2t2,
    "F2T3": f2t3,
    "F3T1": f3t1,
    "F3T2": f3t2,
    "F3T3": f3t3,
    "F4T1": f4t1,
    "F4T2": f4t2,
    "F4T3": f4t3,
    "F5T1": f5t1,
    "F5T2": f5t2,
    "F5T3": f5t3,
}

# One T1 instance per family: the grader 4-case matrix runs on these.
FAMILY_T1: dict[str, Callable[[], TaskInstance]] = {f"F{n}": BUILDERS[f"F{n}T1"] for n in range(1, 6)}

# --- broken variants (SPEC §8), applied to the hand-built instances ------------------------------


def _as_broken(inst: TaskInstance, kind: str, **changes: Any) -> TaskInstance:
    return dataclasses.replace(
        inst, id=f"{inst.template}-broken-{kind}-000", variant="broken", split="broken", broken_type=kind, **changes
    )


def _target_invoice_id(inst: TaskInstance) -> str:
    args, st = inst.oracle_call.arguments, inst.initial_state
    if "invoice_id" in args:
        return str(args["invoice_id"])
    if "payment_id" in args:
        p = st.payment(str(args["payment_id"]))
        assert p is not None
        return p.invoice_id
    k = st.case(str(args["case_id"]))
    assert k is not None
    return k.invoice_id


def impossible(inst: TaskInstance) -> TaskInstance:
    """Remove the referenced record and what references it; rebuild the ledger; keep the stale oracle state."""
    st, args = inst.initial_state, inst.oracle_call.arguments
    if inst.family in ("F1", "F2", "F4"):
        t = _target_invoice_id(inst)
        st = dataclasses.replace(
            st,
            invoices=tuple(i for i in st.invoices if i.id != t),
            payments=tuple(p for p in st.payments if p.invoice_id != t),
            cases=tuple(k for k in st.cases if k.invoice_id != t) if inst.family != "F2" else st.cases,
        )
    elif inst.family == "F3":
        st = dataclasses.replace(st, cases=tuple(k for k in st.cases if k.id != args["case_id"]))
    else:
        c = args["customer_id"]
        st = dataclasses.replace(
            st,
            customers=tuple(r for r in st.customers if r.id != c),
            invoices=tuple(r for r in st.invoices if r.customer_id != c),
            payments=tuple(r for r in st.payments if r.customer_id != c),
            cases=tuple(r for r in st.cases if r.customer_id != c),
        )
    return _as_broken(inst, "impossible", initial_state=reledger(st))


def ambiguous(inst: TaskInstance) -> TaskInstance:
    """Add a second valid target under the params (SPEC §8); recompute the oracle final state."""
    st = inst.initial_state
    cid = st.customers[0].id  # builders put the target customer first
    rate = st.customers[0].contract_rate
    customers, invoices, payments, cases = list(st.customers), list(st.invoices), list(st.payments), list(st.cases)
    request, params = inst.request, inst.params
    if inst.family == "F1":
        invoices.append(invoice("INV-19001", cid, inst.params["period"], [line("Plan: Basic", 1, rate + 200)]))
    elif inst.family == "F2":
        invoices.append(invoice("INV-19002", cid, "2026-05", [line("Plan: Growth", 1, 1234)]))
        payments += [
            payment("PAY-29001", "INV-19002", cid, 1234, "2026-06-02T10:00"),
            payment("PAY-29002", "INV-19002", cid, 1234, "2026-06-02T11:00"),
        ]
    elif inst.family == "F3":
        t = _target_invoice_id(inst)
        request = f"Resolve the open dispute on invoice {t}. Approve it in full."
        params = {"invoice_id": t, "rule": "percent", "percent": 100}
        cases.append(case("CS-9002", cid, t, "Customer disputes the 'Plan: Pro' charge.", 500))
    elif inst.family == "F4":
        invoices.append(invoice("INV-19004", cid, "2026-05", [line("Plan: Team", 1, 5000)], 5500))
    else:
        c = st.customers[0]
        customers.append(dataclasses.replace(c, id="C9999", phone="+1-646-555-0999"))
    st2 = state(customers, invoices, payments, cases)
    final, result = apply_tool(st2, inst.oracle_call)
    assert result.ok
    return _as_broken(inst, "ambiguous", request=request, params=params, initial_state=st2, oracle_final_state=final)


def perturbed_call(inst: TaskInstance) -> ToolCall:
    """SPEC §8 grader_bug perturbation, written independently of `envlab.verify.wrong`."""
    o, st = inst.oracle_call, inst.initial_state
    a = dict(o.arguments)
    if o.name == "apply_credit":
        a["amount"] = int(a["amount"]) + 1
    elif o.name == "refund_payment":
        dup = st.payment(str(a["payment_id"]))
        assert dup is not None
        a["payment_id"] = next(
            p.id
            for p in sorted(st.payments, key=lambda p: p.created_at)
            if p.invoice_id == dup.invoice_id
            and p.status == PaymentStatus.SUCCEEDED
            and p.amount == dup.amount
            and p.id != dup.id
        )
    elif o.name == "resolve_case":
        a["adjustment"] = int(a["adjustment"]) + 1
    elif o.name == "adjust_invoice":
        inv = st.invoice(str(a["invoice_id"]))
        assert inv is not None
        bumped = int(a["new_amount"]) + 1
        a["new_amount"] = bumped + 1 if bumped == inv.amount_billed else bumped
    else:
        v = str(a["value"])
        if a["field"] == "email":
            a["value"] = "x" + v
        elif a["field"] == "address":
            a["value"] = v + " Apt 2"
        else:
            a["value"] = v[:-1] + str((int(v[-1]) + 1) % 10)
    return ToolCall(o.name, a)


def grader_bug(inst: TaskInstance) -> TaskInstance:
    final, result = apply_tool(inst.initial_state, perturbed_call(inst))
    assert result.ok
    return _as_broken(inst, "grader_bug", oracle_final_state=final)


def env_bug(inst: TaskInstance) -> TaskInstance:
    """Corrupt the target so the valid call breaks (SPEC §8); the ledger is left as it was."""
    st, args = inst.initial_state, inst.oracle_call.arguments
    if inst.family in ("F1", "F4"):
        st = dataclasses.replace(
            st,
            invoices=tuple(
                dataclasses.replace(i, status=InvoiceStatus.VOID) if i.id == args["invoice_id"] else i
                for i in st.invoices
            ),
        )
    elif inst.family == "F2":
        st = dataclasses.replace(
            st,
            payments=tuple(
                dataclasses.replace(p, status=PaymentStatus.FAILED) if p.id == args["payment_id"] else p
                for p in st.payments
            ),
        )
    elif inst.family == "F3":
        st = dataclasses.replace(
            st,
            cases=tuple(
                dataclasses.replace(k, status=CaseStatus.RESOLVED, resolution=None) if k.id == args["case_id"] else k
                for k in st.cases
            ),
        )
    else:
        c = st.customer(str(args["customer_id"]))
        assert c is not None
        twin = dataclasses.replace(c, name="Someone Else", email="someone.else@example.org")
        st = dataclasses.replace(st, customers=(*st.customers, twin))
    return _as_broken(inst, "env_bug", initial_state=st)


BROKEN: dict[str, Callable[[TaskInstance], TaskInstance]] = {
    "impossible": impossible,
    "ambiguous": ambiguous,
    "grader_bug": grader_bug,
    "env_bug": env_bug,
}

# SPEC §8 source templates per broken type.
BROKEN_SOURCE: dict[str, dict[str, Callable[[], TaskInstance]]] = {
    "impossible": {f"F{n}": BUILDERS[f"F{n}T1"] for n in range(1, 6)},
    "ambiguous": {"F1": f1t2, "F2": f2t2, "F3": f3t1, "F4": f4t2, "F5": f5t2},
    "grader_bug": {f"F{n}": BUILDERS[f"F{n}T2"] for n in range(1, 6)},
    "env_bug": {f"F{n}": BUILDERS[f"F{n}T1"] for n in range(1, 6)},
}

# --- completions -------------------------------------------------------------------------------


def native(c: ToolCall) -> str:
    """Qwen3.5 native XML-parameter tool call (SPEC §10)."""
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in c.arguments.items())
    return f"<tool_call>\n<function={c.name}>\n{params}</function>\n</tool_call>"
