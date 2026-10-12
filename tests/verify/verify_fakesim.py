"""Test-only stand-in for the sim workstream's modules, written from SPEC §2, §3, §6, §10, §11.

Installed by `conftest.py` into `sys.modules` **only** for modules that do not exist yet
(`envlab.sim.ledger`, `envlab.sim.tools`, `envlab.sim.state_hash`, `envlab.parse`). Once the
real sim is merged it is never used, and the verify tests run against the real simulator.
"""

import dataclasses
import hashlib
import json
import re
from typing import Any

from envlab.calls import ErrorCode, ParseError, ParseFailure, ToolCall, ToolResult
from envlab.sim.entities import (
    CaseStatus,
    Invoice,
    InvoiceStatus,
    LedgerEntry,
    LedgerKind,
    PaymentStatus,
    State,
)
from envlab.sim.schemas import PARAM_TYPES, TOOL_NAMES

# --- envlab.sim.ledger -------------------------------------------------------------------------


def derive_invoice_status(invoice: Invoice) -> InvoiceStatus:
    if invoice.status == InvoiceStatus.VOID:
        return InvoiceStatus.VOID
    return InvoiceStatus.OPEN if invoice.amount_due > 0 else InvoiceStatus.PAID


def next_ledger_id(state: State) -> str:
    return f"L{len(state.ledger) + 1:04d}"


def build_ledger(customers, invoices, payments) -> tuple[LedgerEntry, ...]:
    rows: list[tuple] = []
    for c in customers:
        if c.credit_balance > 0:
            rows.append((c.id, None, None, LedgerKind.ACCOUNT_CREDIT, -c.credit_balance, "Account credit"))
        for inv in (i for i in invoices if i.customer_id == c.id):
            rows.append((c.id, inv.id, None, LedgerKind.CHARGE, inv.amount_billed, "Invoice issued"))
            pays = sorted((p for p in payments if p.invoice_id == inv.id), key=lambda p: p.created_at)
            for p in pays:
                if p.status in (PaymentStatus.SUCCEEDED, PaymentStatus.REFUNDED):
                    rows.append((p.customer_id, inv.id, p.id, LedgerKind.PAYMENT, -p.amount, "Payment received"))
                if p.status == PaymentStatus.REFUNDED:
                    rows.append((p.customer_id, inv.id, p.id, LedgerKind.REFUND, p.amount, p.note))
    return tuple(LedgerEntry(f"L{n:04d}", *row) for n, row in enumerate(rows, start=1))


def check_ledger(state: State) -> list[str]:
    v: list[str] = []
    for inv in state.invoices:  # L1
        total = sum(e.amount for e in state.ledger if e.invoice_id == inv.id)
        if total != inv.amount_due:
            v.append(f"L1 {inv.id}: amount_due {inv.amount_due} != ledger {total}")
    for c in state.customers:  # L2
        total = -sum(e.amount for e in state.ledger if e.customer_id == c.id and e.invoice_id is None)
        if total != c.credit_balance:
            v.append(f"L2 {c.id}: credit_balance {c.credit_balance} != ledger {total}")
    for e in state.ledger:  # L3
        if e.invoice_id is not None:
            inv = state.invoice(e.invoice_id)
            if inv is None or inv.customer_id != e.customer_id:
                v.append(f"L3 {e.id}: invoice customer")
        if e.payment_id is not None:
            p = state.payment(e.payment_id)
            if p is None or p.customer_id != e.customer_id:
                v.append(f"L3 {e.id}: payment customer")
    for p in state.payments:  # L4
        pay = [e for e in state.ledger if e.payment_id == p.id and e.kind == LedgerKind.PAYMENT]
        ref = [e for e in state.ledger if e.payment_id == p.id and e.kind == LedgerKind.REFUND]
        if p.status == PaymentStatus.FAILED:
            if pay or ref:
                v.append(f"L4 {p.id}: failed payment has entries")
            continue
        if len(pay) != 1 or pay[0].amount != -p.amount:
            v.append(f"L4 {p.id}: payment entry")
        if p.status == PaymentStatus.REFUNDED and (len(ref) != 1 or ref[0].amount != p.amount):
            v.append(f"L4 {p.id}: refund entry")
    for n, e in enumerate(state.ledger, start=1):  # L5
        if e.id != f"L{n:04d}":
            v.append(f"L5 {e.id} at position {n}")
    return v


def check_invariants(state: State) -> list[str]:
    v = check_ledger(state)
    for inv in state.invoices:  # S1
        if inv.status == InvoiceStatus.VOID:
            if inv.amount_due != 0:
                v.append(f"S1 {inv.id}: void with amount_due")
        elif inv.status != derive_invoice_status(inv):
            v.append(f"S1 {inv.id}: status")
    for k in state.cases:  # S2
        if (k.status == CaseStatus.RESOLVED) != (k.resolution is not None):
            v.append(f"S2 {k.id}: status/resolution")
        if k.resolution is not None and k.resolution < 0:
            v.append(f"S2 {k.id}: negative resolution")
    for name in ("customers", "invoices", "payments", "cases"):  # S3 unique ids
        ids = [r.id for r in getattr(state, name)]
        if len(ids) != len(set(ids)):
            v.append(f"S3 duplicate {name} id")
    for inv in state.invoices:
        if state.customer(inv.customer_id) is None:
            v.append(f"S3 {inv.id}: customer")
    for r in (*state.payments, *state.cases):
        inv = state.invoice(r.invoice_id)
        if state.customer(r.customer_id) is None or inv is None or inv.customer_id != r.customer_id:
            v.append(f"S3 {r.id}: references")
    for c in state.customers:  # S4
        if c.credit_balance < 0:
            v.append(f"S4 {c.id}")
    for inv in state.invoices:
        if inv.amount_billed < 0 or any(li.quantity < 1 for li in inv.line_items):
            v.append(f"S4 {inv.id}")
    v += [f"S4 {p.id}" for p in state.payments if p.amount <= 0]
    v += [f"S4 {k.id}" for k in state.cases if k.disputed_amount <= 0]
    return v


# --- envlab.sim.state_hash ---------------------------------------------------------------------

_MASKED = {"invoices": "note", "payments": "note", "cases": "note", "ledger": "memo"}


def canonical_dict(state: State, *, mask_text: bool = False) -> dict:
    d = state.to_dict()
    if mask_text:
        for section, key in _MASKED.items():
            for rec in d[section]:
                rec[key] = "*" if rec[key].strip() else ""
    return d


def state_hash(state: State, *, mask_text: bool = False) -> str:
    blob = json.dumps(
        canonical_dict(state, mask_text=mask_text),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


# --- envlab.sim.tools --------------------------------------------------------------------------

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE = re.compile(r"\+?[0-9][0-9 \-]{5,18}[0-9]")


def _err(state: State, code: ErrorCode) -> tuple[State, ToolResult]:
    return state, ToolResult(ok=False, error=code)


def _swap(records: tuple, rid: str, new: Any) -> tuple:
    i = next(n for n, r in enumerate(records) if r.id == rid)
    return (*records[:i], new, *records[i + 1 :])


def _post(state: State, cust: str, inv: str | None, pay: str | None, kind, amount, memo) -> tuple:
    return (*state.ledger, LedgerEntry(next_ledger_id(state), cust, inv, pay, kind, amount, memo))


def _with_due(inv: Invoice, due: int, note: str, **kw) -> Invoice:
    new = dataclasses.replace(inv, amount_due=due, note=note, **kw)
    return dataclasses.replace(new, status=derive_invoice_status(new))


def apply_tool(state: State, call: ToolCall) -> tuple[State, ToolResult]:
    try:
        return _apply(state, call)
    except Exception:  # never raises (§3)
        return _err(state, ErrorCode.BAD_ARGUMENTS)


def _apply(state: State, call: ToolCall) -> tuple[State, ToolResult]:
    if call.name not in PARAM_TYPES:
        return _err(state, ErrorCode.UNKNOWN_TOOL)
    schema = PARAM_TYPES[call.name]
    if set(call.arguments) != set(schema):
        return _err(state, ErrorCode.BAD_ARGUMENTS)
    for k, t in schema.items():
        val = call.arguments[k]
        if (t is int and (type(val) is not int)) or (t is str and not isinstance(val, str)):
            return _err(state, ErrorCode.BAD_ARGUMENTS)
    a: dict[str, Any] = {k: v.strip() if isinstance(v, str) else v for k, v in call.arguments.items()}
    ok = ToolResult(ok=True)

    if call.name == "apply_credit":
        inv = state.invoice(a["invoice_id"])
        if inv is None:
            return _err(state, ErrorCode.NOT_FOUND)
        if inv.status != InvoiceStatus.OPEN:
            return _err(state, ErrorCode.INVALID_STATUS)
        if a["amount"] <= 0:
            return _err(state, ErrorCode.INVALID_AMOUNT)
        if a["amount"] > inv.amount_due:
            return _err(state, ErrorCode.EXCEEDS_AMOUNT_DUE)
        new = _with_due(inv, inv.amount_due - a["amount"], a["reason"])
        ledger = _post(state, inv.customer_id, inv.id, None, LedgerKind.CREDIT, -a["amount"], a["reason"])
        return dataclasses.replace(state, invoices=_swap(state.invoices, inv.id, new), ledger=ledger), ok

    if call.name == "refund_payment":
        p = state.payment(a["payment_id"])
        inv = None if p is None else state.invoice(p.invoice_id)
        if p is None or inv is None:
            return _err(state, ErrorCode.NOT_FOUND)
        if p.status != PaymentStatus.SUCCEEDED or inv.status == InvoiceStatus.VOID:
            return _err(state, ErrorCode.INVALID_STATUS)
        new_p = dataclasses.replace(p, status=PaymentStatus.REFUNDED, note=a["reason"])
        new_inv = _with_due(inv, inv.amount_due + p.amount, inv.note)
        ledger = _post(state, p.customer_id, inv.id, p.id, LedgerKind.REFUND, p.amount, a["reason"])
        return dataclasses.replace(
            state,
            payments=_swap(state.payments, p.id, new_p),
            invoices=_swap(state.invoices, inv.id, new_inv),
            ledger=ledger,
        ), ok

    if call.name == "adjust_invoice":
        inv = state.invoice(a["invoice_id"])
        if inv is None:
            return _err(state, ErrorCode.NOT_FOUND)
        if inv.status == InvoiceStatus.VOID:
            return _err(state, ErrorCode.INVALID_STATUS)
        if a["new_amount"] < 0:
            return _err(state, ErrorCode.INVALID_AMOUNT)
        if a["new_amount"] == inv.amount_billed:
            return _err(state, ErrorCode.NO_CHANGE)
        delta = a["new_amount"] - inv.amount_billed
        new = _with_due(inv, inv.amount_due + delta, a["reason"], amount_billed=a["new_amount"])
        ledger = _post(state, inv.customer_id, inv.id, None, LedgerKind.ADJUSTMENT, delta, a["reason"])
        return dataclasses.replace(state, invoices=_swap(state.invoices, inv.id, new), ledger=ledger), ok

    if call.name == "update_customer":
        c = state.customer(a["customer_id"])
        if c is None:
            return _err(state, ErrorCode.NOT_FOUND)
        field, value = a["field"], a["value"]
        if field not in ("name", "email", "phone", "address"):
            return _err(state, ErrorCode.FIELD_NOT_EDITABLE)
        if (
            not value
            or (field == "email" and not _EMAIL.fullmatch(value))
            or (field == "phone" and not _PHONE.fullmatch(value))
        ):
            return _err(state, ErrorCode.INVALID_VALUE)
        if getattr(c, field) == value:
            return _err(state, ErrorCode.NO_CHANGE)
        new = dataclasses.replace(c, **{field: value})
        return dataclasses.replace(state, customers=_swap(state.customers, c.id, new)), ok

    if call.name == "resolve_case":
        k = state.case(a["case_id"])
        inv = None if k is None else state.invoice(k.invoice_id)
        if k is None or inv is None:
            return _err(state, ErrorCode.NOT_FOUND)
        adj = a["adjustment"]
        if k.status != CaseStatus.OPEN or (adj > 0 and inv.status != InvoiceStatus.OPEN):
            return _err(state, ErrorCode.INVALID_STATUS)
        if adj < 0:
            return _err(state, ErrorCode.INVALID_AMOUNT)
        if adj > inv.amount_due:
            return _err(state, ErrorCode.EXCEEDS_AMOUNT_DUE)
        new_k = dataclasses.replace(k, status=CaseStatus.RESOLVED, resolution=adj, note=a["note"])
        out = dataclasses.replace(state, cases=_swap(state.cases, k.id, new_k))
        if adj > 0:
            new_inv = _with_due(inv, inv.amount_due - adj, inv.note)
            ledger = _post(state, k.customer_id, inv.id, None, LedgerKind.DISPUTE_CREDIT, -adj, a["note"])
            out = dataclasses.replace(out, invoices=_swap(state.invoices, inv.id, new_inv), ledger=ledger)
        return out, ok

    # void_invoice
    inv = state.invoice(a["invoice_id"])
    if inv is None:
        return _err(state, ErrorCode.NOT_FOUND)
    if inv.status != InvoiceStatus.OPEN:
        return _err(state, ErrorCode.INVALID_STATUS)
    active = any(p.invoice_id == inv.id and p.status != PaymentStatus.FAILED for p in state.payments)
    if active or inv.amount_due != inv.amount_billed:
        return _err(state, ErrorCode.HAS_ACTIVITY)
    new = dataclasses.replace(inv, amount_due=0, status=InvoiceStatus.VOID, note=a["reason"])
    ledger = _post(state, inv.customer_id, inv.id, None, LedgerKind.VOID, -inv.amount_due, a["reason"])
    return dataclasses.replace(state, invoices=_swap(state.invoices, inv.id, new), ledger=ledger), ok


# --- envlab.parse ------------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S)
_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNC = re.compile(r"<function=([A-Za-z_]\w*)>(.*)</function>", re.S)
_PARAM = re.compile(r"\s*<parameter=([A-Za-z_]\w*)>(.*?)</parameter>", re.S)
_INT = re.compile(r"-?\d+")


def _fail(reason: ParseError, detail: str = "") -> ParseFailure:
    return ParseFailure(reason=reason, detail=detail)


def parse_tool_call(text: str) -> ToolCall | ParseFailure:
    text = _THINK.sub("", text)
    blocks = _BLOCK.findall(text)
    if not blocks:
        return _fail(ParseError.MALFORMED if "<tool_call>" in text else ParseError.NO_TOOL_CALL)
    if len(blocks) > 1:
        return _fail(ParseError.MULTIPLE_TOOL_CALLS)
    inner = blocks[0].strip()
    raw: dict[str, Any] = {}
    if inner.startswith("{"):
        try:
            obj = json.loads(inner)
        except ValueError:
            return _fail(ParseError.MALFORMED)
        if not (
            isinstance(obj, dict)
            and set(obj) == {"name", "arguments"}
            and isinstance(obj["name"], str)
            and isinstance(obj["arguments"], dict)
        ):
            return _fail(ParseError.MALFORMED)
        name, raw = obj["name"], obj["arguments"]
    else:
        m = _FUNC.fullmatch(inner)
        if m is None:
            return _fail(ParseError.MALFORMED)
        name, body, pos = m.group(1), m.group(2), 0
        while (pm := _PARAM.match(body, pos)) is not None:
            if pm.group(1) in raw:
                return _fail(ParseError.MALFORMED)
            value = pm.group(2)
            value = value[1:] if value.startswith("\n") else value
            value = value[:-1] if value.endswith("\n") else value
            raw[pm.group(1)] = value
            pos = pm.end()
        if body[pos:].strip():
            return _fail(ParseError.MALFORMED)
    if name not in TOOL_NAMES:
        return _fail(ParseError.UNKNOWN_TOOL)
    schema = PARAM_TYPES[name]
    if set(schema) - set(raw):
        return _fail(ParseError.MISSING_ARGUMENT)
    if set(raw) - set(schema):
        return _fail(ParseError.UNEXPECTED_ARGUMENT)
    args: dict[str, str | int] = {}
    for k, t in schema.items():
        v = raw[k]
        if t is int:
            if type(v) is int:
                args[k] = v
            elif isinstance(v, str) and _INT.fullmatch(v.strip()):
                args[k] = int(v.strip())
            else:
                return _fail(ParseError.BAD_ARGUMENT_TYPE)
        elif isinstance(v, str):
            args[k] = v.strip()
        else:
            return _fail(ParseError.BAD_ARGUMENT_TYPE)
    return ToolCall(name=name, arguments=args)


# Which fake functions stand in for which module.
MODULES: dict[str, tuple[str, ...]] = {
    "envlab.sim.ledger": (
        "derive_invoice_status",
        "next_ledger_id",
        "build_ledger",
        "check_ledger",
        "check_invariants",
    ),
    "envlab.sim.tools": ("apply_tool",),
    "envlab.sim.state_hash": ("canonical_dict", "state_hash"),
    "envlab.parse": ("parse_tool_call",),
}
