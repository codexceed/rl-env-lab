"""Independent solvers for gate G2 (SPEC §4 "Solver" lines, §7).

Each solver reads only `params` and `initial_state` and returns every call the request admits.
The free-text argument is copied from the oracle call (SPEC §11.7: "text args = oracle's"); it is
the only thing read from `oracle_call`. Never reads `variant`, `broken_type`, `split` or `meta`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from envlab.calls import ToolCall
from envlab.sim.entities import (
    CaseStatus,
    Customer,
    Invoice,
    InvoiceStatus,
    LineItem,
    PaymentStatus,
    State,
)
from envlab.sim.schemas import TEXT_ARG
from envlab.tasks.instance import TaskInstance

FAMILY_TOOL: dict[str, str] = {
    "F1": "apply_credit",
    "F2": "refund_payment",
    "F3": "resolve_case",
    "F4": "adjust_invoice",
    "F5": "update_customer",
}


def _text_args(instance: TaskInstance, tool: str) -> dict[str, str | int]:
    arg = TEXT_ARG[tool]
    if arg is None:
        return {}
    value = instance.oracle_call.arguments.get(arg, "")
    return {arg: value}


def _matching_customers(state: State, params: dict[str, Any]) -> list[Customer]:
    """Customers matching `customer_id` and/or `customer_name` (whichever are given)."""
    out = []
    for c in state.customers:
        if "customer_id" in params and c.id != params["customer_id"]:
            continue
        if "customer_name" in params and c.name != params["customer_name"]:
            continue
        out.append(c)
    return out


def _invoices_of(state: State, customers: Iterable[Customer]) -> list[Invoice]:
    ids = {c.id for c in customers}
    return [i for i in state.invoices if i.customer_id in ids]


def _plan_line(invoice: Invoice) -> LineItem | None:
    # Generator convention (§1): exactly one "Plan: ..." line, first. Take the first one found.
    return next((li for li in invoice.line_items if li.description.startswith("Plan: ")), None)


def solve_f1(instance: TaskInstance) -> list[ToolCall]:
    """Open invoices of matching customers (filtered by invoice_id/period) with P.unit_price > rate.

    amount = P.quantity * (P.unit_price - rate).
    """
    state, params = instance.initial_state, instance.params
    calls = []
    for c in _matching_customers(state, params):
        for inv in _invoices_of(state, [c]):
            if inv.status != InvoiceStatus.OPEN:
                continue
            if "invoice_id" in params and inv.id != params["invoice_id"]:
                continue
            if "period" in params and inv.period != params["period"]:
                continue
            plan = _plan_line(inv)
            if plan is None or plan.unit_price <= c.contract_rate:
                continue
            amount = plan.quantity * (plan.unit_price - c.contract_rate)
            args = {"invoice_id": inv.id, "amount": amount}
            calls.append(ToolCall("apply_credit", {**args, **_text_args(instance, "apply_credit")}))
    return calls


def solve_f2(instance: TaskInstance) -> list[ToolCall]:
    """Succeeded payments p on a matching invoice with an earlier succeeded q of equal amount."""
    state, params = instance.initial_state, instance.params
    calls = []
    for inv in _invoices_of(state, _matching_customers(state, params)):
        if "invoice_id" in params and inv.id != params["invoice_id"]:
            continue
        ok = [p for p in state.payments if p.invoice_id == inv.id and p.status == PaymentStatus.SUCCEEDED]
        for p in ok:
            if any(q.amount == p.amount and q.created_at < p.created_at for q in ok):
                args = {"payment_id": p.id, **_text_args(instance, "refund_payment")}
                calls.append(ToolCall("refund_payment", args))
    return calls


def _f3_adjustment(state: State, params: dict[str, Any], case_invoice_id: str, disputed: int) -> int | None:
    rule = params.get("rule")
    if rule == "percent":
        return (disputed * params["percent"]) // 100
    if rule == "unused_units":
        inv = state.invoice(case_invoice_id)
        if inv is None:
            return None
        line = next((li for li in inv.line_items if li.description == params["line_description"]), None)
        if line is None:
            return None
        return (line.quantity - params["used_units"]) * line.unit_price
    return None


def solve_f3(instance: TaskInstance) -> list[ToolCall]:
    """Open cases filtered by case_id/invoice_id; adjustment per params.rule."""
    state, params = instance.initial_state, instance.params
    calls = []
    for k in state.cases:
        if k.status != CaseStatus.OPEN:
            continue
        if "case_id" in params and k.id != params["case_id"]:
            continue
        if "invoice_id" in params and k.invoice_id != params["invoice_id"]:
            continue
        adjustment = _f3_adjustment(state, params, k.invoice_id, k.disputed_amount)
        if adjustment is None:
            continue
        args = {"case_id": k.id, "adjustment": adjustment, **_text_args(instance, "resolve_case")}
        calls.append(ToolCall("resolve_case", args))
    return calls


def solve_f4(instance: TaskInstance) -> list[ToolCall]:
    """Open invoices (by invoice_id, or of the customer(s) named customer_name) with billed != S."""
    state, params = instance.initial_state, instance.params
    names = {c.id: c.name for c in state.customers}
    calls = []
    for inv in state.invoices:
        if inv.status != InvoiceStatus.OPEN:
            continue
        if "invoice_id" in params and inv.id != params["invoice_id"]:
            continue
        if "customer_name" in params and names.get(inv.customer_id) != params["customer_name"]:
            continue
        total = sum(li.total for li in inv.line_items)
        if inv.amount_billed == total:
            continue
        args = {"invoice_id": inv.id, "new_amount": total}
        calls.append(ToolCall("adjust_invoice", {**args, **_text_args(instance, "adjust_invoice")}))
    return calls


def solve_f5(instance: TaskInstance) -> list[ToolCall]:
    """Customers matching customer_id (if given) and every `lookup` pair exactly.

    SPEC writes `lookup={"name","email"}`; read literally as a dict {field: expected value}.
    """
    state, params = instance.initial_state, instance.params
    lookup: dict[str, str] = params.get("lookup") or {}
    calls = []
    for c in state.customers:
        if "customer_id" in params and c.id != params["customer_id"]:
            continue
        if any(getattr(c, key, None) != value for key, value in lookup.items()):
            continue
        args = {"customer_id": c.id, "field": params["field"], "value": params["new_value"]}
        calls.append(ToolCall("update_customer", args))
    return calls


SOLVERS: dict[str, Callable[[TaskInstance], list[ToolCall]]] = {
    "F1": solve_f1,
    "F2": solve_f2,
    "F3": solve_f3,
    "F4": solve_f4,
    "F5": solve_f5,
}


def solve(instance: TaskInstance) -> list[ToolCall]:
    """Every call the request admits under `params` + `initial_state` (SPEC §4 Solver lines)."""
    if instance.family not in SOLVERS:
        raise ValueError(f"unknown family {instance.family!r}")
    return SOLVERS[instance.family](instance)
