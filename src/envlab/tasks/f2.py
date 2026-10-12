"""F2 Duplicate payment -> ``refund_payment`` (SPEC §4 F2, §5)."""

from dataclasses import replace
from datetime import datetime, timedelta

from envlab.calls import ToolCall
from envlab.sim.entities import Customer, LineItem, Payment, PaymentStatus
from envlab.tasks.common import (
    PERIODS,
    REFUND_NOTE,
    Built,
    Draft,
    Record,
    UnitSpec,
    add_units,
    draw_k,
    line_sum,
    other_customer_unit,
    z_unit,
)

FAMILY = "F2"
REASON = "Refund duplicate payment"


def split_installment(d: Draft, total: int) -> int:
    """First installment ``a`` ∈ [30%, 70%] of ``total`` with ``a ≠ total − a``."""
    lo, hi = -(-3 * total // 10), 7 * total // 10
    while True:
        a = d.rng.randint(lo, hi)
        if a != total - a:
            return a


def duplicate_invoice(d: Draft, c: Customer, period: str, lines: list[LineItem], *, hard=False):
    """An invoice charged twice: P1, then P2 = P1 + m minutes, both succeeded (3 records).

    Base: m ∈ [1, 1440]. Hard: m ∈ [1, 15] on the same day (``pay_time`` keeps hour ≤ 20).
    """
    billed = line_sum(lines)
    inv = d.invoice(c, period, lines, paid=2 * billed)
    t1 = d.pay_time(period)
    m = d.rng.randint(1, 15) if hard else d.rng.randint(1, 1440)
    p1 = d.payment(inv, billed, t1)
    p2 = d.payment(inv, billed, t1 + timedelta(minutes=m))
    return inv, p1, p2


def build(d: Draft, template: str, variant: str) -> Built:
    hard = variant == "hard"
    rng = d.rng
    c = d.customer()
    period = rng.choice(PERIODS)
    lines = [d.plan_line(c), *d.addons(rng.randint(0, 1))]  # T billed B, 1–2 lines
    billed = line_sum(lines)

    if template == "F2T3":
        a = split_installment(d, billed)
        t = d.invoice(c, period, lines, paid=a + 2 * (billed - a))
        t1 = d.pay_time(period)
        t2 = t1 + timedelta(minutes=rng.randint(60, 2 * 1440))
        t3 = t2 + timedelta(minutes=rng.randint(1, 1440))
        pays = [d.payment(t, a, t1), d.payment(t, billed - a, t2), d.payment(t, billed - a, t3)]
    else:
        t, p1, p2 = duplicate_invoice(d, c, period, lines, hard=hard)
        pays = [p1, p2]
    dup, first_success = pays[-1], pays[0]
    call = ToolCall("refund_payment", {"payment_id": dup.id, "reason": REASON})

    if template == "F2T1":
        request = (
            f"{c.name} ({c.id}) was charged twice for invoice {t.id}. "
            "Refund the duplicate charge."
        )
        params = {"customer_id": c.id, "invoice_id": t.id}
    elif template == "F2T2":
        request = (
            f"{c.name} says they were double-charged on one of their invoices. "
            "Refund the extra payment."
        )
        params = {"customer_name": c.name}
    else:
        request = (
            f"{c.name} ({c.id}) paid invoice {t.id} in two installments, but one installment "
            "went through twice. Refund the duplicate."
        )
        params = {"customer_id": c.id, "invoice_id": t.id}

    def make_a(_max: int) -> list[Record]:
        """Failed payment on T, the duplicate's amount, before the first success (1)."""
        ts = rng.randint(1, 420)
        when = _parse_ts(first_success) - timedelta(minutes=ts)
        return [d.payment(t, dup.amount, when, PaymentStatus.FAILED)]

    def own_period() -> str | None:
        return d.free_period(c.id)

    def make_b(_max: int, *, same_lines: bool = False) -> list[Record] | None:
        """Another invoice of C paid once (2). F2T2 requires its payment amount = B."""
        p = own_period()
        if p is None:
            return None
        return d.paid_once(c, p, list(lines) if same_lines else d.background_lines(c))

    def c_build(o: Customer) -> list[Record]:
        return d.paid_once(o, rng.choice(PERIODS), d.background_lines(o))

    def make_d(_max: int) -> list[Record] | None:
        """(hard) Another invoice of C whose earlier duplicate was already refunded (3)."""
        p = own_period()
        if p is None:
            return None
        ls = d.background_lines(c)
        amt = line_sum(ls)
        inv = d.invoice(c, p, ls, paid=amt)
        t1 = d.pay_time(p)
        first = d.payment(inv, amt, t1)
        again = d.payment(
            inv, amt, t1 + timedelta(minutes=rng.randint(1, 1440)), PaymentStatus.REFUNDED
        )
        return [inv, first, _with_note(again, REFUND_NOTE)]

    def make_e(_max: int) -> list[Record] | None:
        """(hard) Another invoice of C paid in two unequal installments (3)."""
        p = own_period()
        if p is None:
            return None
        ls = d.background_lines(c)
        amt = line_sum(ls)
        a = split_installment(d, amt)
        inv = d.invoice(c, p, ls, paid=amt)
        t1 = d.pay_time(p)
        return [
            inv,
            d.payment(inv, a, t1),
            d.payment(inv, amt - a, t1 + timedelta(minutes=rng.randint(60, 2 * 1440))),
        ]

    unit_a = UnitSpec("a", 1, make_a)
    unit_b = UnitSpec("b", 2, make_b)
    unit_b_same = UnitSpec("b", 2, lambda m: make_b(m, same_lines=True))
    unit_c = other_customer_unit(d, "c", 3, c_build)
    unit_d = UnitSpec("d", 3, make_d, hard_only=True)
    unit_e = UnitSpec("e", 3, make_e, hard_only=True)

    required: list[UnitSpec] = []
    if template == "F2T2":
        required = [unit_e, unit_b_same] if hard else [unit_b_same]
    elif template == "F2T3":
        required = [unit_a]
    elif template == "F2T1" and hard:
        required = [unit_d, unit_a]

    units = add_units(
        d,
        draw_k(rng, variant),
        required,
        [unit_a, unit_b, unit_c, unit_d, unit_e, z_unit(d)],
        hard=hard,
    )
    return Built(FAMILY, template, variant, request, params, call, [c, t, *pays], units)


def _parse_ts(p: Payment) -> datetime:
    return datetime.strptime(p.created_at, "%Y-%m-%dT%H:%M")


def _with_note(p: Payment, note: str) -> Payment:
    return replace(p, note=note)
