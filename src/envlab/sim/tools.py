"""The six billing tools (SPEC §3, §11.3).

`apply_tool` is pure: records are replaced copy-on-write and the input state is never mutated.
On any error it returns the *same* state object with `ToolResult(ok=False, error=...)`.
It never raises.
"""

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from envlab.calls import ErrorCode, ToolCall, ToolResult
from envlab.sim.entities import (
    CaseStatus,
    InvoiceStatus,
    LedgerEntry,
    LedgerKind,
    PaymentStatus,
    State,
)
from envlab.sim.ledger import derive_invoice_status, next_ledger_id
from envlab.sim.schemas import PARAM_TYPES

EDITABLE_FIELDS: tuple[str, ...] = ("name", "email", "phone", "address")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_RE = re.compile(r"\+?[0-9][0-9 \-]{5,18}[0-9]")

_OK = ToolResult(ok=True)


class _ToolError(Exception):
    def __init__(self, code: ErrorCode, message: str = "") -> None:
        super().__init__(code, message)
        self.code = code
        self.message = message


def _index(records: tuple[Any, ...], id: str) -> int | None:
    """Position of the first record with this id (corrupt states: first match wins)."""
    return next((i for i, r in enumerate(records) if r.id == id), None)


def _put(records: tuple[Any, ...], i: int, record: Any) -> tuple[Any, ...]:
    """Replace the record at position i, keeping tuple order."""
    return records[:i] + (record,) + records[i + 1 :]


def _post(
    state: State,
    *,
    customer_id: str,
    invoice_id: str | None,
    payment_id: str | None,
    kind: LedgerKind,
    amount: int,
    memo: str,
) -> tuple[LedgerEntry, ...]:
    entry = LedgerEntry(
        id=next_ledger_id(state),
        customer_id=customer_id,
        invoice_id=invoice_id,
        payment_id=payment_id,
        kind=kind,
        amount=amount,
        memo=memo,
    )
    return state.ledger + (entry,)


def _apply_credit(state: State, invoice_id: str, amount: int, reason: str) -> State:
    i = _index(state.invoices, invoice_id)
    if i is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"invoice {invoice_id}")
    inv = state.invoices[i]
    if inv.status != InvoiceStatus.OPEN:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"invoice {inv.id} is {inv.status}")
    if amount <= 0:
        raise _ToolError(ErrorCode.INVALID_AMOUNT, "amount must be > 0")
    if amount > inv.amount_due:
        raise _ToolError(ErrorCode.EXCEEDS_AMOUNT_DUE, f"amount_due is {inv.amount_due}")
    new_inv = replace(inv, amount_due=inv.amount_due - amount, note=reason)
    new_inv = replace(new_inv, status=derive_invoice_status(new_inv))
    return replace(
        state,
        invoices=_put(state.invoices, i, new_inv),
        ledger=_post(
            state,
            customer_id=inv.customer_id,
            invoice_id=inv.id,
            payment_id=None,
            kind=LedgerKind.CREDIT,
            amount=-amount,
            memo=reason,
        ),
    )


def _refund_payment(state: State, payment_id: str, reason: str) -> State:
    pi = _index(state.payments, payment_id)
    if pi is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"payment {payment_id}")
    pay = state.payments[pi]
    ii = _index(state.invoices, pay.invoice_id)
    if ii is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"invoice {pay.invoice_id}")
    inv = state.invoices[ii]
    if pay.status != PaymentStatus.SUCCEEDED:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"payment {pay.id} is {pay.status}")
    if inv.status == InvoiceStatus.VOID:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"invoice {inv.id} is void")
    new_pay = replace(pay, status=PaymentStatus.REFUNDED, note=reason)
    new_inv = replace(inv, amount_due=inv.amount_due + pay.amount)
    new_inv = replace(new_inv, status=derive_invoice_status(new_inv))
    return replace(
        state,
        invoices=_put(state.invoices, ii, new_inv),
        payments=_put(state.payments, pi, new_pay),
        ledger=_post(
            state,
            customer_id=inv.customer_id,
            invoice_id=inv.id,
            payment_id=pay.id,
            kind=LedgerKind.REFUND,
            amount=pay.amount,
            memo=reason,
        ),
    )


def _adjust_invoice(state: State, invoice_id: str, new_amount: int, reason: str) -> State:
    i = _index(state.invoices, invoice_id)
    if i is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"invoice {invoice_id}")
    inv = state.invoices[i]
    if inv.status == InvoiceStatus.VOID:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"invoice {inv.id} is void")
    if new_amount < 0:
        raise _ToolError(ErrorCode.INVALID_AMOUNT, "new_amount must be >= 0")
    if new_amount == inv.amount_billed:
        raise _ToolError(ErrorCode.NO_CHANGE, f"amount_billed is already {new_amount}")
    delta = new_amount - inv.amount_billed
    new_inv = replace(inv, amount_billed=new_amount, amount_due=inv.amount_due + delta, note=reason)
    new_inv = replace(new_inv, status=derive_invoice_status(new_inv))
    return replace(
        state,
        invoices=_put(state.invoices, i, new_inv),
        ledger=_post(
            state,
            customer_id=inv.customer_id,
            invoice_id=inv.id,
            payment_id=None,
            kind=LedgerKind.ADJUSTMENT,
            amount=delta,
            memo=reason,
        ),
    )


def _update_customer(state: State, customer_id: str, field: str, value: str) -> State:
    i = _index(state.customers, customer_id)
    if i is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"customer {customer_id}")
    cust = state.customers[i]
    if field not in EDITABLE_FIELDS:
        raise _ToolError(ErrorCode.FIELD_NOT_EDITABLE, f"field {field!r}")
    if not value:
        raise _ToolError(ErrorCode.INVALID_VALUE, "empty value")
    if field == "email" and not EMAIL_RE.fullmatch(value):
        raise _ToolError(ErrorCode.INVALID_VALUE, "not an email address")
    if field == "phone" and not PHONE_RE.fullmatch(value):
        raise _ToolError(ErrorCode.INVALID_VALUE, "not a phone number")
    if getattr(cust, field) == value:
        raise _ToolError(ErrorCode.NO_CHANGE, f"{field} is already {value!r}")
    return replace(state, customers=_put(state.customers, i, replace(cust, **{field: value})))


def _resolve_case(state: State, case_id: str, adjustment: int, note: str) -> State:
    ci = _index(state.cases, case_id)
    if ci is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"case {case_id}")
    case = state.cases[ci]
    ii = _index(state.invoices, case.invoice_id)
    if ii is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"invoice {case.invoice_id}")
    inv = state.invoices[ii]
    if case.status != CaseStatus.OPEN:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"case {case.id} is {case.status}")
    if adjustment > 0 and inv.status != InvoiceStatus.OPEN:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"invoice {inv.id} is {inv.status}")
    if adjustment < 0:
        raise _ToolError(ErrorCode.INVALID_AMOUNT, "adjustment must be >= 0")
    # Literal SPEC reading: this check also applies when adjustment == 0, so resolving with
    # 0 on an overpaid invoice (amount_due < 0) returns exceeds_amount_due.
    if adjustment > inv.amount_due:
        raise _ToolError(ErrorCode.EXCEEDS_AMOUNT_DUE, f"amount_due is {inv.amount_due}")
    new_case = replace(case, status=CaseStatus.RESOLVED, resolution=adjustment, note=note)
    new = replace(state, cases=_put(state.cases, ci, new_case))
    if adjustment > 0:
        new_inv = replace(inv, amount_due=inv.amount_due - adjustment)
        new_inv = replace(new_inv, status=derive_invoice_status(new_inv))
        new = replace(
            new,
            invoices=_put(state.invoices, ii, new_inv),
            ledger=_post(
                state,
                customer_id=inv.customer_id,
                invoice_id=inv.id,
                payment_id=None,
                kind=LedgerKind.DISPUTE_CREDIT,
                amount=-adjustment,
                memo=note,
            ),
        )
    return new


def _void_invoice(state: State, invoice_id: str, reason: str) -> State:
    i = _index(state.invoices, invoice_id)
    if i is None:
        raise _ToolError(ErrorCode.NOT_FOUND, f"invoice {invoice_id}")
    inv = state.invoices[i]
    if inv.status != InvoiceStatus.OPEN:
        raise _ToolError(ErrorCode.INVALID_STATUS, f"invoice {inv.id} is {inv.status}")
    has_payments = any(
        p.invoice_id == inv.id and p.status in (PaymentStatus.SUCCEEDED, PaymentStatus.REFUNDED)
        for p in state.payments
    )
    if has_payments or inv.amount_due != inv.amount_billed:
        raise _ToolError(ErrorCode.HAS_ACTIVITY, f"invoice {inv.id} has payments or credits")
    new_inv = replace(inv, amount_due=0, status=InvoiceStatus.VOID, note=reason)
    return replace(
        state,
        invoices=_put(state.invoices, i, new_inv),
        ledger=_post(
            state,
            customer_id=inv.customer_id,
            invoice_id=inv.id,
            payment_id=None,
            kind=LedgerKind.VOID,
            amount=-inv.amount_due,
            memo=reason,
        ),
    )


_TOOLS: dict[str, Callable[..., State]] = {
    "apply_credit": _apply_credit,
    "refund_payment": _refund_payment,
    "adjust_invoice": _adjust_invoice,
    "update_customer": _update_customer,
    "resolve_case": _resolve_case,
    "void_invoice": _void_invoice,
}


def _check_arguments(name: str, arguments: Any) -> dict[str, str | int]:
    """bad_arguments check: exact key set and exact types; returns stripped arguments."""
    schema = PARAM_TYPES[name]
    if not isinstance(arguments, dict) or set(arguments) != set(schema):
        raise _ToolError(ErrorCode.BAD_ARGUMENTS, f"expected arguments {list(schema)}")
    out: dict[str, str | int] = {}
    for key, typ in schema.items():
        value = arguments[key]
        if typ is int:
            if type(value) is not int:  # rejects bool and int subclasses
                raise _ToolError(ErrorCode.BAD_ARGUMENTS, f"{key} must be an integer")
            out[key] = value
        else:
            if not isinstance(value, str):
                raise _ToolError(ErrorCode.BAD_ARGUMENTS, f"{key} must be a string")
            # SPEC §3: all string arguments are stored stripped. Reading: ids are stripped
            # before lookup too, so lookups and stored values agree.
            out[key] = value.strip()
    return out


def apply_tool(state: State, call: ToolCall) -> tuple[State, ToolResult]:
    """Run one tool call. Pure and never raises (SPEC §3)."""
    try:
        name = getattr(call, "name", None)
        if not isinstance(name, str) or name not in _TOOLS:
            raise _ToolError(ErrorCode.UNKNOWN_TOOL, f"unknown tool {name!r}")
        args = _check_arguments(name, getattr(call, "arguments", None))
        return _TOOLS[name](state, **args), _OK
    except _ToolError as err:
        return state, ToolResult(ok=False, error=err.code, message=err.message)
    except Exception as exc:  # contract: never raise, whatever the input
        # No error code fits an internal failure (e.g. a state of the wrong shape).
        # Reading: report it as bad_arguments, with the exception type in the message.
        return state, ToolResult(
            ok=False, error=ErrorCode.BAD_ARGUMENTS, message=f"internal:{type(exc).__name__}"
        )
