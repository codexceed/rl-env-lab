"""v3 F1 overcharge -> ``apply_credit`` (SPEC_v3.md §4 F1).

Templates: F1A per-seat overcharge (customer by name + month) · F1B promised p% off the plan line ·
F1C mid-month downgrade prorated by days · F1N per-seat overcharge and p% off (near) · F1X tiered
seat pricing (far).
"""

from __future__ import annotations

from envlab.calls import ToolCall
from envlab.sim.entities import ContractBasis, Customer, LineItem
from envlab.tasks.common import PERIODS, Built, Draft, Record, Unit, UnitSpec, add_units, dollars, month_name
from envlab.tasks.v3.common import (
    addon_lines,
    days_in_month,
    draw_k,
    first_name_unit,
    long_date,
    other_flat_plan,
    seat_customer,
    twin,
    z_unit,
)

FAMILY = "F1"
REASON = "Credit overcharge versus contract rate"

# --- difficulty knobs (SPEC_v3.md §5) ---------------------------------------------------------
SEAT_RATE = (800, 9000)  # per-seat contract rate, cents
SEATS = (12, 180)  # seats on a per-seat plan line
SEAT_OVER = (25, 900)  # per-seat overcharge (F1A, F1N), cents
T_ADDONS = (1, 3)  # add-on lines on the target invoice
PERCENTS = (6, 8, 12, 15, 18, 22, 24, 27, 35)  # promised discount (F1B, F1N)
F1B_PER_SEAT_SHARE = 0.5  # share of F1B customers on a per-seat contract
DOWNGRADE_GAP = (500, 15000)  # old plan price - new plan price (F1C), cents
TIER_SEATS = (10, 20, 25, 40, 50)  # seats in the first price tier (F1X)
TIER_EXTRA = (3, 120)  # seats beyond the first tier (F1X)
TIER1_PRICE = (1500, 7000)  # first-tier seat price (F1X), cents
TIER_GAP = (150, 1500)  # first-tier minus second-tier seat price (F1X), cents


# --- oracle rules (integer cents) -------------------------------------------------------------


def per_seat_credit(qty: int, unit: int, rate: int) -> int:
    """F1A: ``qty x (unit - rate)``."""
    return qty * (unit - rate)


def percent_off_credit(qty: int, unit: int, percent: int) -> int:
    """F1B: ``percent`` % of the plan line total, rounded down."""
    return qty * unit * percent // 100


def prorate_credit(old: int, new: int, days_on_new: int, month_days: int) -> int:
    """F1C: ``(old - new) x days_on_new / month_days``, rounded down."""
    return (old - new) * days_on_new // month_days


def per_seat_percent_credit(qty: int, unit: int, rate: int, percent: int) -> int:
    """F1N: billed seat line minus the correct one, ``qty x rate`` less ``percent`` % rounded down."""
    correct = qty * rate - qty * rate * percent // 100
    return qty * unit - correct


def tiered_credit(qty: int, unit: int, tier_seats: int, tier1: int, tier2: int) -> int:
    """F1X: billed seat line minus ``min(qty, n) x tier1 + max(0, qty - n) x tier2``."""
    return qty * unit - (min(qty, tier_seats) * tier1 + max(0, qty - tier_seats) * tier2)


# --- generator ----------------------------------------------------------------------------------


def _seat_rate(d: Draft) -> int:
    return d.rng.randint(*SEAT_RATE)


def _seats(d: Draft) -> int:
    return d.rng.randint(*SEATS)


def build(d: Draft, template: str, variant: str) -> Built:
    rng = d.rng
    period = rng.choice(PERIODS)
    month = month_name(period)
    tier_n: int | None = None  # F1X only
    addons = addon_lines(d, rng.randint(*T_ADDONS))

    if template in ("F1A", "F1N"):
        c = seat_customer(d, _seat_rate(d))
        rate = c.contract_rate
        plan = d.plan_line(c, seats=_seats(d), unit=rate + rng.randint(*SEAT_OVER))
        t = d.invoice(c, period, [plan, *addons])
        if template == "F1A":
            amount = per_seat_credit(plan.quantity, plan.unit_price, rate)
            request = (
                f"{c.name} is on a per-seat contract. Their {month} invoice charged more per seat than "
                "their contract rate. Credit the overcharge."
            )
            params = {"customer_name": c.name, "period": period, "rule": "per_seat"}
        else:
            pct = rng.choice(PERCENTS)
            amount = per_seat_percent_credit(plan.quantity, plan.unit_price, rate, pct)
            request = (
                f"{c.name} ({c.id}) has a per-seat contract and was promised {pct}% off their seats. "
                f"Their {month} invoice billed the seats above the contract rate and without the discount. "
                f"The correct seat charge is seats x contract rate, less {pct}% of that amount rounded down "
                "to the cent. Credit the difference between the billed seat line and the correct seat charge."
            )
            params = {"customer_id": c.id, "period": period, "rule": "per_seat_percent", "percent": pct}
    elif template == "F1B":
        if rng.random() < F1B_PER_SEAT_SHARE:
            c = seat_customer(d, _seat_rate(d))
            plan = d.plan_line(c, seats=_seats(d))
        else:
            c = d.customer()
            plan = d.plan_line(c)
        t = d.invoice(c, period, [plan, *addons])
        pct = rng.choice(PERCENTS)
        amount = percent_off_credit(plan.quantity, plan.unit_price, pct)
        request = (
            f"{c.name} ({c.id}) was promised {pct}% off their plan, but their {month} invoice billed the plan "
            f"at the full price. Credit {pct}% of that invoice's plan line total, rounded down to the cent."
        )
        params = {"customer_id": c.id, "period": period, "rule": "percent_off", "percent": pct}
    elif template == "F1C":
        c = d.customer()
        old = c.contract_rate + rng.randint(*DOWNGRADE_GAP)
        old_name = other_flat_plan(c)
        plan = LineItem(old_name, 1, old)
        t = d.invoice(c, period, [plan, *addons])
        month_days = days_in_month(period)
        day = rng.randint(2, month_days - 1)
        amount = prorate_credit(old, c.contract_rate, month_days - day + 1, month_days)
        new_name = d.plan_line(c).description
        request = (
            f"{c.name} ({c.id}) switched from the {old_name.removeprefix('Plan: ')} plan to the cheaper "
            f"{new_name.removeprefix('Plan: ')} plan on {long_date(period, day)}; their contract rate is now the "
            f"new plan's monthly price. Invoice {t.id} still charged the old plan's price for the whole "
            "month. Credit (old price - new price) x days on the new plan / days in the month, rounded down to "
            "the cent. Days on the new plan run from the switch date through the last day of the month."
        )
        params = {
            "customer_id": c.id,
            "invoice_id": t.id,
            "rule": "prorate_days",
            "downgrade_date": f"{period}-{day:02d}",
        }
    elif template == "F1X":
        tier1 = rng.randint(*TIER1_PRICE)
        tier2 = tier1 - rng.randint(*TIER_GAP)
        n = tier_n = rng.choice(TIER_SEATS)
        c = seat_customer(d, tier1)
        plan = d.plan_line(c, seats=n + rng.randint(*TIER_EXTRA))
        t = d.invoice(c, period, [plan, *addons])
        amount = tiered_credit(plan.quantity, plan.unit_price, n, tier1, tier2)
        request = (
            f"{c.name} ({c.id}) has tiered seat pricing: the first {n} seats cost {dollars(tier1)} each and "
            f"every seat beyond {n} costs {dollars(tier2)}. Invoice {t.id} charged every seat at the "
            "first-tier price. Credit the overcharge."
        )
        params = {
            "customer_id": c.id,
            "invoice_id": t.id,
            "rule": "tiered",
            "tier_seats": n,
            "tier1_price": tier1,
            "tier2_price": tier2,
        }
    else:
        raise ValueError(f"unknown F1 template {template!r}")

    if not 2 <= amount < t.amount_due:
        raise AssertionError(f"{template}: credit {amount} outside [2, amount_due)")
    call = ToolCall("apply_credit", {"invoice_id": t.id, "amount": amount, "reason": REASON})
    units = _units(d, template, c, plan, period, tier_seats=tier_n)
    return Built(FAMILY, template, variant, request, params, call, [c, t], units)


# --- distractor units -------------------------------------------------------------------------


def _units(d: Draft, template: str, c: Customer, plan: LineItem, period: str, *, tier_seats: int | None) -> list[Unit]:
    rng = d.rng
    per_seat = c.contract_basis is ContractBasis.PER_SEAT

    def own_lines() -> list[LineItem]:
        """A correctly priced invoice of ``c`` (F1X: seats within the first tier)."""
        seats = rng.randint(3, tier_seats) if tier_seats else (_seats(d) if per_seat else None)
        return [d.plan_line(c, seats=seats), *addon_lines(d, rng.randint(0, 2))]

    def make_a(max_records: int) -> list[Record] | None:
        """``a``: another invoice of C, other month, at the contract rate: open (1) or paid (2)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        if max_records >= 2 and rng.random() < 0.5:
            return d.paid_once(c, p, own_lines())
        return [d.invoice(c, p, own_lines())]

    def make_p(_max: int) -> list[Record] | None:
        """``p``: another open invoice of C, other month, overcharged the same way (1)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        over = d.plan_line(c, seats=_seats(d), unit=plan.unit_price)
        return [d.invoice(c, p, [over, *addon_lines(d, rng.randint(0, 2))])]

    def make_s(_max: int) -> list[Record]:
        """``s``: a customer with C's exact name, same contract basis, a correct invoice in C's month (2)."""
        o = twin(d, c, rate=_seat_rate(d) if per_seat else d.flat_rate())
        seats = _seats(d) if per_seat else None
        return [o, d.invoice(o, period, [d.plan_line(o, seats=seats), *addon_lines(d, rng.randint(0, 2))])]

    def make_b(_max: int) -> list[Record]:
        """``b``: another customer with one correctly priced invoice (2)."""
        o = d.customer()
        return [o, d.invoice(o, rng.choice(PERIODS), d.background_lines(o))]

    unit_a = UnitSpec("a", 1, make_a)
    unit_p = UnitSpec("p", 1, make_p)
    unit_s = UnitSpec("s", 2, make_s)
    unit_b = UnitSpec("b", 2, make_b)
    unit_d = first_name_unit(d, c)

    required = {
        "F1A": [unit_s, unit_p],
        "F1B": [unit_a],
        "F1C": [unit_a],
        "F1N": [unit_p],
        "F1X": [unit_a],
    }[template]
    allowed = [unit_a, unit_b, unit_d, unit_s] + ([unit_p] if template in ("F1A", "F1N") else [])
    return add_units(d, draw_k(rng), required, [*allowed, z_unit(d)], hard=False)
