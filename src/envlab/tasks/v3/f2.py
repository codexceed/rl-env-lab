"""v3 F2 duplicate or extra payment -> ``refund_payment`` (SPEC_v3.md §4 F2).

Templates: F2A duplicate among failed and already-refunded attempts · F2B installment charged twice
(customer by name + month) · F2C duplicate, customer found by email or phone only · F2N installment
duplicate, customer found by email only (near) · F2X overpayment: refund the payment that pushed the
total paid above the bill (far).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from envlab.calls import ToolCall
from envlab.sim.entities import Customer, Invoice, LineItem, Payment, PaymentStatus
from envlab.tasks.common import PERIODS, Built, Draft, Record, Unit, UnitSpec, add_units, line_sum, month_name
from envlab.tasks.v3.common import (
    addon_lines,
    draw_k,
    first_name_unit,
    installments,
    later,
    local_twin,
    paid_installments,
    refunded,
    split_amount,
    twin,
    z_unit,
)

FAMILY = "F2"
REASON = "Refund duplicate payment"

# --- difficulty knobs (SPEC_v3.md §5) ---------------------------------------------------------
T_ADDONS = (0, 2)  # add-on lines on the target invoice
F2A_PAYMENTS = (4, 5)  # payments on T in F2A: original, duplicate, >= 1 failed, >= 1 refunded
INSTALLMENTS = (2, 4)  # installments in F2B / F2N
F2X_INSTALLMENTS = (2, 3)  # installments before the extra payment in F2X
F2X_EXTRA_SHARE = (5, 60)  # extra payment as % of the bill (F2X)
F2C_PHONE_SHARE = 0.5  # share of F2C requests that give a phone instead of an email


def _lines(d: Draft, c: Customer) -> list[LineItem]:
    return [d.plan_line(c), *addon_lines(d, d.rng.randint(*T_ADDONS))]


def _ts(p: Payment) -> datetime:
    return datetime.strptime(p.created_at, "%Y-%m-%dT%H:%M")


def duplicated(d: Draft, c: Customer, period: str, lines: list[LineItem]) -> tuple[Invoice, Payment, Payment]:
    """An invoice charged twice in full: original, then duplicate 1 minute to 1 day later (3 rec)."""
    billed = line_sum(lines)
    inv = d.invoice(c, period, lines, paid=2 * billed)
    t1 = d.pay_time(period)
    return inv, d.payment(inv, billed, t1), d.payment(inv, billed, later(d.rng, t1, 1, 1440))


def installment_duplicate(
    d: Draft, c: Customer, period: str, lines: list[LineItem]
) -> tuple[Invoice, list[Payment], Payment]:
    """An invoice paid in 2-4 unequal installments, one of which went through twice.

    Returns the invoice, every payment (time order) and the duplicate (the later copy).
    """
    rng = d.rng
    billed = line_sum(lines)
    amounts = split_amount(rng, billed, rng.randint(*INSTALLMENTS))
    i = rng.randrange(len(amounts))
    inv = d.invoice(c, period, lines, paid=billed + amounts[i])
    pays = installments(d, inv, amounts, d.pay_time(period))
    dup = d.payment(inv, amounts[i], later(rng, _ts(pays[i]), 1, 2 * 1440))
    return inv, sorted([*pays, dup], key=lambda p: p.created_at), dup


def _f2a_payments(d: Draft, inv: Invoice, billed: int, period: str) -> tuple[list[Payment], Payment, Payment]:
    """Original, duplicate, >= 1 failed and >= 1 already-refunded attempt, all of the bill's amount."""
    rng = d.rng
    n = rng.randint(*F2A_PAYMENTS)
    extras = ["failed", "refunded"] + [rng.choice(("failed", "refunded")) for _ in range(n - 4)]
    roles = ["original", "duplicate", *extras]
    # The original is the earliest charge that went through (succeeded or later refunded).
    while True:
        rng.shuffle(roles)
        through = [r for r in roles if r != "failed"]
        if through[0] == "original":
            break
    pays, ts = [], d.pay_time(period)
    original = duplicate = None
    for role in roles:
        status = PaymentStatus.FAILED if role == "failed" else PaymentStatus.SUCCEEDED
        p = d.payment(inv, billed, ts, status)
        if role == "refunded":
            p = refunded(p)
        pays.append(p)
        original = p if role == "original" else original
        duplicate = p if role == "duplicate" else duplicate
        ts = later(rng, ts, 1, 1440)
    assert original is not None and duplicate is not None
    return pays, original, duplicate


def build(d: Draft, template: str, variant: str) -> Built:
    rng = d.rng
    c = d.customer()
    period = rng.choice(PERIODS)
    lines = _lines(d, c)
    billed = line_sum(lines)
    contact_kind = "email"

    if template == "F2A":
        t = d.invoice(c, period, lines, paid=2 * billed)
        pays, _, dup = _f2a_payments(d, t, billed, period)
        request = (
            f"{c.name} ({c.id}) says one of their invoices was charged more than once. "
            "Refund the duplicate charge that has not been refunded yet."
        )
        params = {"customer_id": c.id, "rule": "duplicate"}
    elif template in ("F2B", "F2N"):
        t, pays, dup = installment_duplicate(d, c, period, lines)
        if template == "F2B":
            request = (
                f"{c.name} paid their {month_name(period)} invoice in installments, but one installment "
                "went through twice. Refund the duplicate."
            )
            params = {"customer_name": c.name, "period": period, "rule": "duplicate"}
        else:
            request = (
                f'Support inbox message from {c.email}: "Hi, I paid one of my invoices in installments and one '
                'installment went through twice. Please refund the duplicate."'
            )
            params = {"lookup": {"email": c.email}, "rule": "duplicate"}
    elif template == "F2C":
        t, p1, dup = duplicated(d, c, period, lines)
        pays = [p1, dup]
        message = '"Hi, I was charged twice for one of my invoices. Please refund the extra payment."'
        if rng.random() < F2C_PHONE_SHARE:
            contact_kind = "phone"
            request = f"Voicemail from {c.phone}: {message}"
            params = {"lookup": {"phone": c.phone}, "rule": "duplicate"}
        else:
            request = f"Support inbox message from {c.email}: {message}"
            params = {"lookup": {"email": c.email}, "rule": "duplicate"}
    elif template == "F2X":
        amounts = split_amount(rng, billed, rng.randint(*F2X_INSTALLMENTS))
        extra = billed * rng.randint(*F2X_EXTRA_SHARE) // 100
        while extra in amounts:
            extra += 1
        t = d.invoice(c, period, lines, paid=billed + extra)
        pays = installments(d, t, amounts, d.pay_time(period))
        dup = d.payment(t, extra, later(rng, _ts(pays[-1]), 60, 3 * 1440))
        pays.append(dup)
        if rng.random() < 0.5:  # a declined retry after the extra payment, so "latest" is not the answer
            pays.append(d.payment(t, rng.choice(amounts), later(rng, _ts(dup), 1, 1440), PaymentStatus.FAILED))
        request = (
            f"{c.name} ({c.id}) paid more than invoice {t.id} was billed. Refund the payment that pushed "
            "their total paid above the billed amount."
        )
        params = {"customer_id": c.id, "invoice_id": t.id, "rule": "overpayment"}
    else:
        raise ValueError(f"unknown F2 template {template!r}")

    call = ToolCall("refund_payment", {"payment_id": dup.id, "reason": REASON})
    units = _units(d, template, c, t, dup, pays, period, contact_kind)
    return Built(FAMILY, template, variant, request, params, call, [c, t, *pays], units)


# --- distractor units -------------------------------------------------------------------------


def _units(
    d: Draft,
    template: str,
    c: Customer,
    t: Invoice,
    dup: Payment,
    pays: list[Payment],
    period: str,
    contact_kind: str,
) -> list[Unit]:
    rng = d.rng
    first_ok = min((p for p in pays if p.status is not PaymentStatus.FAILED), key=lambda p: p.created_at)

    def make_a(_max: int) -> list[Record]:
        """``a``: a failed payment on T of the duplicate's amount, before the first charge (1)."""
        when = _ts(first_ok) - timedelta(minutes=rng.randint(1, 420))
        return [d.payment(t, dup.amount, when, PaymentStatus.FAILED)]

    def own_period() -> str | None:
        return d.free_period(c.id)

    def make_b(_max: int) -> list[Record] | None:
        """``b``: another invoice of C paid once, by a payment of T's billed amount (2)."""
        p = own_period()
        return None if p is None else d.paid_once(c, p, list(t.line_items))

    def make_d(_max: int) -> list[Record] | None:
        """``d``: another invoice of C whose duplicate was already refunded (3)."""
        p = own_period()
        if p is None:
            return None
        ls = d.background_lines(c)
        amt = line_sum(ls)
        inv = d.invoice(c, p, ls, paid=amt)
        t1 = d.pay_time(p)
        return [inv, d.payment(inv, amt, t1), refunded(d.payment(inv, amt, later(rng, t1, 1, 1440)))]

    def make_e(_max: int) -> list[Record] | None:
        """``e``: another invoice of C paid in 2-3 unequal installments (3-4)."""
        p = own_period()
        return None if p is None else paid_installments(d, c, p, d.background_lines(c))

    def make_c(_max: int) -> list[Record]:
        """``c``: another customer with an invoice paid once (3)."""
        o = d.customer()
        return [o, *d.paid_once(o, rng.choice(PERIODS), d.background_lines(o))]

    def make_s(_max: int) -> list[Record]:
        """``s``: a customer with C's exact name and an invoice in C's month paid in installments (4-5)."""
        o = twin(d, c)
        return [o, *paid_installments(d, o, period, d.background_lines(o))]

    def make_n(_max: int) -> list[Record] | None:
        """``n``: a near-duplicate contact (same name; same email local part or look-alike phone)
        with its own genuine duplicate: charged twice (F2C) or an installment twice (F2N) (4-6)."""
        o = twin(d, c, phone_like=contact_kind == "phone")
        if contact_kind == "email" and o.email.split("@")[0] != c.email.split("@")[0]:
            o2 = local_twin(d, c)
            if o2 is None:
                return None
            o = o2
        p = rng.choice(PERIODS)
        if template == "F2N":
            inv, ps, _ = installment_duplicate(d, o, p, d.background_lines(o))
            return [o, inv, *ps]
        inv, p1, p2 = duplicated(d, o, p, d.background_lines(o))
        return [o, inv, p1, p2]

    unit_a = UnitSpec("a", 1, make_a)
    unit_b = UnitSpec("b", 2, make_b)
    unit_c = UnitSpec("c", 3, make_c)
    unit_d = UnitSpec("d", 3, make_d)
    unit_e = UnitSpec("e", 3, make_e)
    unit_s = UnitSpec("s", 4, make_s)
    unit_n = UnitSpec("n", 4, make_n)
    unit_f = first_name_unit(d, c)

    required = {
        "F2A": [unit_d],
        "F2B": [unit_s],
        "F2C": [unit_n],
        "F2N": [unit_n],
        "F2X": [unit_e],
    }[template]
    allowed = [unit_b, unit_c, unit_d, unit_e, unit_f]
    if template != "F2X":
        allowed.append(unit_a)
    return add_units(d, draw_k(rng), required, [*allowed, z_unit(d)], hard=False)
