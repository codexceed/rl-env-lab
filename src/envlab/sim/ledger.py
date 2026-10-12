"""Ledger construction and invariant checks (SPEC §1 S-rules, §2 L-rules, §11.2).

Sign convention: positive = customer owes more, negative = customer owes less.
"""

from __future__ import annotations

from collections import Counter

from envlab.sim.entities import (
    CaseStatus,
    Customer,
    Invoice,
    InvoiceStatus,
    LedgerEntry,
    LedgerKind,
    Payment,
    PaymentStatus,
    State,
)

_LEDGER_PAYMENT_STATUSES = (PaymentStatus.SUCCEEDED, PaymentStatus.REFUNDED)


def derive_invoice_status(invoice: Invoice) -> InvoiceStatus:
    """S1: void stays void; otherwise open iff amount_due > 0, else paid."""
    if invoice.status == InvoiceStatus.VOID:
        return InvoiceStatus.VOID
    return InvoiceStatus.OPEN if invoice.amount_due > 0 else InvoiceStatus.PAID


def ledger_id(n: int) -> str:
    """Ledger id for the n-th entry (1-based)."""
    return f"L{n:04d}"


def next_ledger_id(state: State) -> str:
    return ledger_id(len(state.ledger) + 1)


def build_ledger(
    customers: tuple[Customer, ...],
    invoices: tuple[Invoice, ...],
    payments: tuple[Payment, ...],
) -> tuple[LedgerEntry, ...]:
    """Initial ledger in §2 order, numbered L0001, L0002, ...

    For each customer in order: `account_credit` (if credit_balance > 0); then for each of the
    customer's invoices in order: `charge`, then that invoice's payments sorted by `created_at`
    (stable, so ties keep tuple order): `payment` entry, then `refund` entry if refunded.
    Failed payments post nothing. Invoices whose customer is not in `customers` post nothing.
    """
    rows: list[tuple[str, str | None, str | None, LedgerKind, int, str]] = []
    for cust in customers:
        if cust.credit_balance > 0:
            rows.append(
                (
                    cust.id,
                    None,
                    None,
                    LedgerKind.ACCOUNT_CREDIT,
                    -cust.credit_balance,
                    "Account credit",
                )
            )
        for inv in invoices:
            if inv.customer_id != cust.id:
                continue
            rows.append((cust.id, inv.id, None, LedgerKind.CHARGE, inv.amount_billed, "Invoice issued"))
            inv_payments = sorted((p for p in payments if p.invoice_id == inv.id), key=lambda p: p.created_at)
            for p in inv_payments:
                if p.status not in _LEDGER_PAYMENT_STATUSES:
                    continue
                rows.append((cust.id, inv.id, p.id, LedgerKind.PAYMENT, -p.amount, "Payment received"))
                if p.status == PaymentStatus.REFUNDED:
                    rows.append((cust.id, inv.id, p.id, LedgerKind.REFUND, p.amount, p.note))
    return tuple(
        LedgerEntry(
            id=ledger_id(i),
            customer_id=cid,
            invoice_id=iid,
            payment_id=pid,
            kind=kind,
            amount=amount,
            memo=memo,
        )
        for i, (cid, iid, pid, kind, amount, memo) in enumerate(rows, start=1)
    )


def check_ledger(state: State) -> list[str]:
    """L1-L5 violations; an empty list means the ledger balances."""
    out: list[str] = []
    ledger = state.ledger

    # L1: every invoice's amount_due equals the sum of its entries.
    by_invoice: Counter[str] = Counter()
    for e in ledger:
        if e.invoice_id is not None:
            by_invoice[e.invoice_id] += e.amount
    for inv in state.invoices:
        total = by_invoice.get(inv.id, 0)
        if inv.amount_due != total:
            out.append(f"L1: invoice {inv.id} amount_due {inv.amount_due} != ledger sum {total}")

    # L2: every customer's credit_balance equals minus the sum of its invoice-less entries.
    by_customer: Counter[str] = Counter()
    for e in ledger:
        if e.invoice_id is None:
            by_customer[e.customer_id] += e.amount
    for cust in state.customers:
        total = -by_customer.get(cust.id, 0)
        if cust.credit_balance != total:
            out.append(f"L2: customer {cust.id} credit_balance {cust.credit_balance} != ledger {total}")

    # L3: an entry's customer matches its invoice's and its payment's customer.
    # Reading: a reference that does not resolve cannot match, so it is a violation too.
    for e in ledger:
        if e.invoice_id is not None:
            inv = state.invoice(e.invoice_id)
            if inv is None:
                out.append(f"L3: entry {e.id} references unknown invoice {e.invoice_id}")
            elif inv.customer_id != e.customer_id:
                out.append(f"L3: entry {e.id} customer {e.customer_id} != invoice {inv.id} customer {inv.customer_id}")
        if e.payment_id is not None:
            pay = state.payment(e.payment_id)
            if pay is None:
                out.append(f"L3: entry {e.id} references unknown payment {e.payment_id}")
            elif pay.customer_id != e.customer_id:
                out.append(f"L3: entry {e.id} customer {e.customer_id} != payment {pay.id} customer {pay.customer_id}")

    # L4: payment/refund entries agree with payment status.
    # Reading: a succeeded payment must also have no refund entry, and a failed payment has
    # no entry of any kind.
    for p in state.payments:
        mine = [e for e in ledger if e.payment_id == p.id]
        pay_entries = [e for e in mine if e.kind == LedgerKind.PAYMENT]
        ref_entries = [e for e in mine if e.kind == LedgerKind.REFUND]
        if p.status == PaymentStatus.FAILED:
            if mine:
                out.append(f"L4: failed payment {p.id} has {len(mine)} ledger entries")
            continue
        if len(pay_entries) != 1 or pay_entries[0].amount != -p.amount:
            out.append(f"L4: payment {p.id} needs exactly one payment entry of {-p.amount}")
        want_refunds = 1 if p.status == PaymentStatus.REFUNDED else 0
        if len(ref_entries) != want_refunds or any(e.amount != p.amount for e in ref_entries):
            out.append(f"L4: {p.status} payment {p.id} needs {want_refunds} refund entry(ies)")

    # L5: ids are exactly L0001...L{n} in tuple order.
    for i, e in enumerate(ledger, start=1):
        if e.id != ledger_id(i):
            out.append(f"L5: entry at position {i} has id {e.id}, expected {ledger_id(i)}")

    return out


def _dupes(ids: list[str]) -> list[str]:
    return sorted(k for k, n in Counter(ids).items() if n > 1)


def check_invariants(state: State) -> list[str]:
    """L1-L5 plus S1-S4 violations; an empty list means the state is valid."""
    out = check_ledger(state)

    # S1: invoice status agrees with amount_due.
    for inv in state.invoices:
        if inv.status == InvoiceStatus.VOID:
            if inv.amount_due != 0:
                out.append(f"S1: void invoice {inv.id} has amount_due {inv.amount_due}")
        elif inv.status != derive_invoice_status(inv):
            out.append(f"S1: invoice {inv.id} status {inv.status} with amount_due {inv.amount_due}")

    # S2: resolved <=> resolution set; resolution >= 0.
    for case in state.cases:
        resolved = case.status == CaseStatus.RESOLVED
        if resolved != (case.resolution is not None):
            out.append(f"S2: case {case.id} status {case.status} with resolution {case.resolution}")
        if case.resolution is not None and case.resolution < 0:
            out.append(f"S2: case {case.id} resolution {case.resolution} < 0")

    # S3: unique ids per record type, references resolve, customers agree.
    for label, records in (
        ("customer", state.customers),
        ("invoice", state.invoices),
        ("payment", state.payments),
        ("case", state.cases),
    ):
        for dup in _dupes([r.id for r in records]):
            out.append(f"S3: duplicate {label} id {dup}")
    customer_ids = {c.id for c in state.customers}
    for inv in state.invoices:
        if inv.customer_id not in customer_ids:
            out.append(f"S3: invoice {inv.id} references unknown customer {inv.customer_id}")
    for label, records in (("payment", state.payments), ("case", state.cases)):
        for r in records:
            if r.customer_id not in customer_ids:
                out.append(f"S3: {label} {r.id} references unknown customer {r.customer_id}")
            inv = state.invoice(r.invoice_id)
            if inv is None:
                out.append(f"S3: {label} {r.id} references unknown invoice {r.invoice_id}")
            elif inv.customer_id != r.customer_id:
                out.append(
                    f"S3: {label} {r.id} customer {r.customer_id} != invoice {inv.id} customer {inv.customer_id}"
                )
    for e in state.ledger:
        if e.customer_id not in customer_ids:
            out.append(f"S3: ledger entry {e.id} references unknown customer {e.customer_id}")

    # S4: value ranges.
    for c in state.customers:
        if c.credit_balance < 0:
            out.append(f"S4: customer {c.id} credit_balance {c.credit_balance} < 0")
    for inv in state.invoices:
        if inv.amount_billed < 0:
            out.append(f"S4: invoice {inv.id} amount_billed {inv.amount_billed} < 0")
        for li in inv.line_items:
            if li.quantity < 1:
                out.append(f"S4: invoice {inv.id} line {li.description!r} quantity {li.quantity}")
    for p in state.payments:
        if p.amount <= 0:
            out.append(f"S4: payment {p.id} amount {p.amount} <= 0")
    for case in state.cases:
        if case.disputed_amount <= 0:
            out.append(f"S4: case {case.id} disputed_amount {case.disputed_amount} <= 0")

    return out
