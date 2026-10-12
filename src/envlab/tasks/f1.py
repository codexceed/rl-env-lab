"""F1 Overcharge -> ``apply_credit`` (SPEC §4 F1, §5)."""

from __future__ import annotations

from envlab.calls import ToolCall
from envlab.sim.entities import ContractBasis, Customer, LineItem
from envlab.tasks.common import (
    PERIODS,
    Built,
    Draft,
    Record,
    UnitSpec,
    add_units,
    dollars,
    draw_k,
    month_name,
    other_customer_unit,
    shift_period,
    z_unit,
)

FAMILY = "F1"
REASON = "Credit overcharge versus contract rate"


def overbilled_lines(d: Draft, c: Customer, n_addons: int) -> list[LineItem]:
    """Plan line at ``rate + D`` (D in [100, 5000]) plus ``n_addons`` add-ons."""
    return [d.plan_line(c, unit=c.contract_rate + d.rng.randint(100, 5000)), *d.addons(n_addons)]


def credit_amount(c: Customer, plan: LineItem) -> int:
    """Oracle rule: ``P.quantity x (P.unit_price - rate)``."""
    return plan.quantity * (plan.unit_price - c.contract_rate)


def build(d: Draft, template: str, variant: str) -> Built:
    hard = variant == "hard"
    rng = d.rng

    if template == "F1T3":
        rate = rng.randint(500, 5000)
        c = d.customer(basis=ContractBasis.PER_SEAT, rate=rate)
        seats, over = rng.randint(3, 40), rng.randint(25, 500)
        lines = [d.plan_line(c, seats=seats, unit=rate + over), *d.addons(rng.randint(1, 2))]
    else:
        c = d.customer()
        if template == "F1T1":
            n_addons = rng.randint(1, 2) if hard else 0
        else:
            n_addons = rng.randint(1, 2) if hard else rng.randint(0, 1)
        lines = overbilled_lines(d, c, n_addons)
    # F1T2-hard needs both adjacent months to exist inside PERIODS.
    period = rng.choice(PERIODS[1:-1] if (template == "F1T2" and hard) else PERIODS)
    t = d.invoice(c, period, lines)
    amount = credit_amount(c, lines[0])
    call = ToolCall("apply_credit", {"invoice_id": t.id, "amount": amount, "reason": REASON})

    first = c.name.split(" ")[0]
    if template == "F1T1":
        request = (
            f"Customer {c.name} ({c.id}) says invoice {t.id} is higher than their contract rate. Credit the overcharge."
        )
        params = {"customer_id": c.id, "invoice_id": t.id}
    elif template == "F1T2":
        request = (
            f"{c.name} was promised {dollars(c.contract_rate)} per month, but their "
            f"{month_name(period)} invoice is higher. Credit the difference."
        )
        params = {"customer_name": c.name, "period": period}
    else:
        request = (
            f"{c.name} ({c.id}) is on a per-seat contract. Invoice {t.id} charged more per "
            "seat than agreed. Credit the overcharge."
        )
        params = {"customer_id": c.id, "invoice_id": t.id}

    seats = lines[0].quantity

    def unit_a(periods: tuple[str, ...] = PERIODS) -> UnitSpec:
        """Another invoice of C, other period, correctly priced: open (1) or paid (2)."""

        def make(max_records: int) -> list[Record] | None:
            p = d.free_period(c.id, periods)
            if p is None:
                return None
            ls = [d.plan_line(c, seats=seats), *d.addons(rng.randint(0, 2))]
            if max_records >= 2 and rng.random() < 0.5:
                return d.paid_once(c, p, ls)
            return [d.invoice(c, p, ls)]

        return UnitSpec("a", 1, make)

    def b_build(o: Customer) -> list[Record]:
        return [d.invoice(o, rng.choice(PERIODS), d.background_lines(o))]

    def make_c(_max: int) -> list[Record]:
        """(hard) Another customer with C's contract rate and an overbilled open invoice."""
        o = d.customer(rate=c.contract_rate)
        return [o, d.invoice(o, rng.choice(PERIODS), overbilled_lines(d, o, 0))]

    unit_b = other_customer_unit(d, "b", 2, b_build)
    unit_c = UnitSpec("c", 2, make_c, hard_only=True)
    unit_d = UnitSpec("d", 1, lambda _m: [d.customer(first=first)])

    required: list[UnitSpec] = []
    if template == "F1T2" and not hard:
        required = [unit_a()]
    elif template == "F1T1" and hard:
        required = [unit_a(), unit_c]
    elif template == "F1T2" and hard:
        required = [
            unit_a((shift_period(period, -1),)),
            unit_a((shift_period(period, 1),)),
            unit_d,
        ]

    units = add_units(d, draw_k(rng, variant), required, [unit_a(), unit_b, unit_c, unit_d, z_unit(d)], hard=hard)
    return Built(FAMILY, template, variant, request, params, call, [c, t], units)
