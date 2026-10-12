"""Independent solvers for the v3 templates (SPEC_v3.md §4 "Solver" lines, gate G2).

Written from the SPEC, not from the generators: each solver reads only ``params`` and
``initial_state`` (plus the oracle's free-text argument, as in v1) and returns every call the
request admits. Shape per family: filter candidates by the identification keys present in
``params`` (``customer_id``, ``customer_name``, ``lookup``, ``invoice_id``, ``period``,
``case_id``), then apply ``params["rule"]``.
"""

from __future__ import annotations

import calendar
from collections.abc import Callable, Iterator
from typing import Any

from envlab.calls import ToolCall
from envlab.sim.entities import CaseStatus, Customer, Invoice, InvoiceStatus, LineItem, PaymentStatus, State
from envlab.tasks.instance import TaskInstance
from envlab.verify.solvers import _text_args, solve_f5

PLAN_PREFIX = "Plan: "


def _customers(state: State, params: dict[str, Any]) -> list[Customer]:
    """Customers matching every identification key given: ``customer_id``, ``customer_name``, ``lookup``."""
    lookup: dict[str, str] = params.get("lookup") or {}
    out = []
    for c in state.customers:
        if "customer_id" in params and c.id != params["customer_id"]:
            continue
        if "customer_name" in params and c.name != params["customer_name"]:
            continue
        if any(getattr(c, key, None) != value for key, value in lookup.items()):
            continue
        out.append(c)
    return out


def _invoices(state: State, params: dict[str, Any], *, open_only: bool) -> Iterator[tuple[Customer, Invoice]]:
    """(customer, invoice) for invoices of matching customers, filtered by ``invoice_id`` / ``period``."""
    by_id = {c.id: c for c in _customers(state, params)}
    for inv in state.invoices:
        c = by_id.get(inv.customer_id)
        if c is None or (open_only and inv.status != InvoiceStatus.OPEN):
            continue
        if "invoice_id" in params and inv.id != params["invoice_id"]:
            continue
        if "period" in params and inv.period != params["period"]:
            continue
        yield c, inv


def _plan(inv: Invoice) -> LineItem | None:
    return next((li for li in inv.line_items if li.description.startswith(PLAN_PREFIX)), None)


def _month_days(period: str) -> int:
    year, month = (int(x) for x in period.split("-"))
    return calendar.monthrange(year, month)[1]


def _line(inv: Invoice, description: str) -> LineItem | None:
    return next((li for li in inv.line_items if li.description == description), None)


# --- F1 apply_credit ----------------------------------------------------------------------------


def _f1_amount(params: dict[str, Any], c: Customer, inv: Invoice) -> int | None:
    plan = _plan(inv)
    if plan is None:
        return None
    qty, unit, rate, rule = plan.quantity, plan.unit_price, c.contract_rate, params["rule"]
    if rule == "per_seat":
        return qty * (unit - rate)
    if rule == "percent_off":
        return (qty * unit * params["percent"]) // 100
    if rule == "prorate_days":
        days = _month_days(inv.period)
        downgrade_day = int(params["downgrade_date"].rsplit("-", 1)[1])
        return ((unit - rate) * (days - downgrade_day + 1)) // days
    if rule == "per_seat_percent":
        at_rate = qty * rate
        return qty * unit - (at_rate - (at_rate * params["percent"]) // 100)
    if rule == "tiered":
        n, p1, p2 = params["tier_seats"], params["tier1_price"], params["tier2_price"]
        return qty * unit - (min(qty, n) * p1 + max(0, qty - n) * p2)
    return None


def solve_f1(instance: TaskInstance) -> list[ToolCall]:
    """Open invoices of matching customers (by invoice_id / period) whose credit under the rule is > 0."""
    state, params = instance.initial_state, instance.params
    calls = []
    for c, inv in _invoices(state, params, open_only=True):
        amount = _f1_amount(params, c, inv)
        if amount is None or amount <= 0:
            continue
        args = {"invoice_id": inv.id, "amount": amount, **_text_args(instance, "apply_credit")}
        calls.append(ToolCall("apply_credit", args))
    return calls


# --- F2 refund_payment ----------------------------------------------------------------------------


def solve_f2(instance: TaskInstance) -> list[ToolCall]:
    """``duplicate``: succeeded p with an earlier succeeded q of equal amount on the same invoice.

    ``overpayment``: on an invoice whose succeeded payments total more than ``amount_billed``, the
    first succeeded payment (by ``created_at``) at which the running total exceeds it.
    """
    state, params = instance.initial_state, instance.params
    calls = []
    for _, inv in _invoices(state, params, open_only=False):
        ok = [p for p in state.payments if p.invoice_id == inv.id and p.status == PaymentStatus.SUCCEEDED]
        hits = []
        if params["rule"] == "duplicate":
            hits = [p for p in ok if any(q.amount == p.amount and q.created_at < p.created_at for q in ok)]
        elif params["rule"] == "overpayment":
            total = 0
            for p in sorted(ok, key=lambda p: p.created_at):
                total += p.amount
                if total > inv.amount_billed:
                    hits = [p]
                    break
        for p in hits:
            calls.append(ToolCall("refund_payment", {"payment_id": p.id, **_text_args(instance, "refund_payment")}))
    return calls


# --- F3 resolve_case ------------------------------------------------------------------------------


def _f3_adjustment(params: dict[str, Any], disputed: int, inv: Invoice) -> int | None:
    rule = params["rule"]
    if rule == "percent_cap":
        return min((disputed * params["percent"]) // 100, params["cap"])
    if rule == "outage_hours":
        plan = _plan(inv)
        if plan is None:
            return None
        return (plan.total * params["hours"]) // (_month_days(inv.period) * 24)
    line = _line(inv, params.get("line_description", ""))
    if line is None:
        return None
    unused = line.quantity - params["used_units"]
    if rule == "unused_units":
        return unused * line.unit_price
    if rule == "unused_units_percent":
        return (unused * line.unit_price * params["percent"]) // 100
    if rule == "allowance":
        billable = max(0, params["used_units"] - params["allowance_units"])
        return (line.quantity - billable) * line.unit_price
    return None


def solve_f3(instance: TaskInstance) -> list[ToolCall]:
    """Open cases (by case_id, or on invoices passing the customer/invoice/period filters); A per rule."""
    state, params = instance.initial_state, instance.params
    invoices = {inv.id: inv for _, inv in _invoices(state, params, open_only=False)}
    calls = []
    for k in state.cases:
        if k.status != CaseStatus.OPEN or k.invoice_id not in invoices:
            continue
        if "case_id" in params and k.id != params["case_id"]:
            continue
        adjustment = _f3_adjustment(params, k.disputed_amount, invoices[k.invoice_id])
        if adjustment is None:
            continue
        args = {"case_id": k.id, "adjustment": adjustment, **_text_args(instance, "resolve_case")}
        calls.append(ToolCall("resolve_case", args))
    return calls


# --- F4 adjust_invoice ----------------------------------------------------------------------------


def _f4_total(params: dict[str, Any], inv: Invoice) -> int | None:
    s = sum(li.total for li in inv.line_items)
    rule = params["rule"]
    if rule == "sum":
        return s
    if rule == "tax":
        return s + (s * params["tax_bp"]) // 10000
    if rule == "minimum":
        return max(s, params["minimum"])
    return None


def solve_f4(instance: TaskInstance) -> list[ToolCall]:
    """Open invoices passing the filters whose correct total under the rule differs from amount_billed."""
    state, params = instance.initial_state, instance.params
    calls = []
    for _, inv in _invoices(state, params, open_only=True):
        total = _f4_total(params, inv)
        if total is None or total == inv.amount_billed:
            continue
        args = {"invoice_id": inv.id, "new_amount": total, **_text_args(instance, "adjust_invoice")}
        calls.append(ToolCall("adjust_invoice", args))
    return calls


SOLVERS_V3: dict[str, Callable[[TaskInstance], list[ToolCall]]] = {
    "F1": solve_f1,
    "F2": solve_f2,
    "F3": solve_f3,
    "F4": solve_f4,
    "F5": solve_f5,  # v1's: customer_id and every lookup pair; the value is params.new_value
}
