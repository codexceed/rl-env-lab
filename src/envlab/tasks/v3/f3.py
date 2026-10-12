"""v3 F3 dispute -> ``resolve_case`` (SPEC_v3.md §4 F3).

Templates: F3A p% of the disputed amount with a cap · F3B unused units x unit price (customer by
name + month) · F3C outage credit prorated by hours · F3N unused units, then p% (near) · F3X usage
with a free allowance (far).
"""

from __future__ import annotations

from envlab.calls import ToolCall
from envlab.sim.entities import Case, Customer, Invoice, LineItem
from envlab.tasks.common import PERIODS, Built, Draft, Record, Unit, UnitSpec, add_units, dollars, month_name
from envlab.tasks.v3.common import (
    USAGE_LINES,
    addon_lines,
    days_in_month,
    draw_k,
    first_name_unit,
    seat_customer,
    twin,
    z_unit,
)

FAMILY = "F3"
NOTE = "Dispute resolved per policy"

# --- difficulty knobs (SPEC_v3.md §5) ---------------------------------------------------------
PERCENTS = (15, 35, 45, 55, 65, 85)  # policy percent (F3A, F3N)
DISPUTED_QTY = (1, 12)  # quantity of the disputed add-on line (F3A)
T_ADDONS = (2, 4)  # add-on lines on F3A's target invoice
CAP_BINDING_SHARE = 0.5  # share of F3A instances where the cap is below p% of the disputed amount
USAGE_QTY = (120, 4000)  # billed units on a usage line
USAGE_UNIT = (3, 400)  # usage unit price, cents
USAGE_LINES_ON_T = (1, 2)  # usage lines on the target invoice
USED_SHARE = (25, 95)  # used units as % of billed units
ALLOWANCE_SHARE = (10, 60)  # free allowance as % of used units (F3X)
OUTAGE_HOURS = (3, 72)  # outage length (F3C)
SEAT_SHARE_C = 0.5  # share of F3C customers on a per-seat plan
SEATS = (12, 180)
SEAT_RATE = (800, 9000)


# --- oracle rules (integer cents) -------------------------------------------------------------


def percent_cap_adjustment(disputed: int, percent: int, cap: int) -> int:
    """F3A: ``min(disputed x percent / 100 rounded down, cap)``."""
    return min(disputed * percent // 100, cap)


def unused_adjustment(billed_units: int, used_units: int, unit_price: int) -> int:
    """F3B: ``(billed - used) x unit price``."""
    return (billed_units - used_units) * unit_price


def outage_adjustment(plan_total: int, hours: int, month_days: int) -> int:
    """F3C: ``plan total x hours / (days x 24)``, rounded down."""
    return plan_total * hours // (month_days * 24)


def unused_percent_adjustment(billed_units: int, used_units: int, unit_price: int, percent: int) -> int:
    """F3N: ``percent`` % of the unused units' value, rounded down."""
    return (billed_units - used_units) * unit_price * percent // 100


def allowance_adjustment(billed_units: int, used_units: int, allowance: int, unit_price: int) -> int:
    """F3X: ``(billed - max(0, used - allowance)) x unit price``."""
    return (billed_units - max(0, used_units - allowance)) * unit_price


def usage_summary(line: LineItem, used: int) -> str:
    return f"Customer says they used only {used} of the {line.quantity} units billed on the '{line.description}' line."


# --- generator ----------------------------------------------------------------------------------


def _usage_invoice_lines(d: Draft, c: Customer) -> tuple[list[LineItem], LineItem]:
    """The plan, 1-2 usage lines (one is disputed), 0-2 add-ons; returns lines and the disputed line."""
    rng = d.rng
    names = rng.sample(USAGE_LINES, rng.randint(*USAGE_LINES_ON_T))
    usage = [LineItem(n, rng.randint(*USAGE_QTY), rng.randint(*USAGE_UNIT)) for n in names]
    rest = [*usage, *addon_lines(d, rng.randint(0, 2))]
    rng.shuffle(rest)
    return [d.plan_line(c), *rest], usage[0]


def _used(d: Draft, billed_units: int) -> int:
    lo, hi = USED_SHARE
    return max(1, min(billed_units - 1, billed_units * d.rng.randint(lo, hi) // 100))


def _cap(d: Draft, raw: int) -> int:
    """A whole-dollar cap, below ``raw`` (binding) or above it, never equal to it."""
    if raw >= 300 and d.rng.random() < CAP_BINDING_SHARE:
        return 100 * d.rng.randint(1, (raw - 1) // 100)
    return 100 * d.rng.randint(raw // 100 + 1, raw // 100 + 1 + max(1, raw // 200))


def build(d: Draft, template: str, variant: str) -> Built:
    rng = d.rng
    period = rng.choice(PERIODS)
    month = month_name(period)

    if template == "F3A":
        c = d.customer()
        addons = addon_lines(d, rng.randint(*T_ADDONS), qty=DISPUTED_QTY)
        disputed = addons[0]
        rng.shuffle(addons)
        t = d.invoice(c, period, [d.plan_line(c), *addons])
        k = d.case(t, d.dispute_summary(disputed), disputed.total)
        pct = rng.choice(PERCENTS)
        cap = _cap(d, disputed.total * pct // 100)
        amount = percent_cap_adjustment(k.disputed_amount, pct, cap)
        request = (
            f"Resolve the open dispute on invoice {t.id}. Policy: approve {pct}% of the disputed amount, "
            f"rounded down to the cent, but no more than {dollars(cap)}."
        )
        params = {"invoice_id": t.id, "rule": "percent_cap", "percent": pct, "cap": cap}
    elif template == "F3C":
        if rng.random() < SEAT_SHARE_C:
            c = seat_customer(d, rng.randint(*SEAT_RATE))
            plan = d.plan_line(c, seats=rng.randint(*SEATS))
        else:
            c = d.customer()
            plan = d.plan_line(c)
        t = d.invoice(c, period, [plan, *addon_lines(d, rng.randint(0, 2))])
        hours = rng.randint(*OUTAGE_HOURS)
        day = rng.randint(1, days_in_month(period))
        summary = (
            f"Customer reports a {hours}-hour service outage on {month.split(' ')[0]} {day} and asks for a credit."
        )
        k = d.case(t, summary, plan.total)
        amount = outage_adjustment(plan.total, hours, days_in_month(period))
        request = (
            f"Resolve case {k.id}. Outage policy: credit the plan line's total x outage hours / hours in the "
            "invoice's billing month, rounded down to the cent."
        )
        params = {"case_id": k.id, "rule": "outage_hours", "hours": hours}
    elif template in ("F3B", "F3N", "F3X"):
        c = d.customer()
        lines, usage = _usage_invoice_lines(d, c)
        t = d.invoice(c, period, lines)
        used = _used(d, usage.quantity)
        k = d.case(t, usage_summary(usage, used), usage.total)
        if template == "F3B":
            amount = unused_adjustment(usage.quantity, used, usage.unit_price)
            request = (
                f"{c.name} disputes the usage billed on their {month} invoice. Resolve the open case: credit "
                "only the unused units, at the invoice's unit price."
            )
            params = {
                "customer_name": c.name,
                "period": period,
                "rule": "unused_units",
                "used_units": used,
                "line_description": usage.description,
            }
        elif template == "F3N":
            pct = rng.choice(PERCENTS)
            amount = unused_percent_adjustment(usage.quantity, used, usage.unit_price, pct)
            request = (
                f"Resolve case {k.id}. Policy: take the unused units at the invoice's unit price, then approve "
                f"{pct}% of that amount, rounded down to the cent."
            )
            params = {
                "case_id": k.id,
                "rule": "unused_units_percent",
                "used_units": used,
                "line_description": usage.description,
                "percent": pct,
            }
        else:
            allowance = max(1, used * rng.randint(*ALLOWANCE_SHARE) // 100)
            amount = allowance_adjustment(usage.quantity, used, allowance, usage.unit_price)
            request = (
                f"Resolve case {k.id}. The customer's plan includes {allowance} free units on the "
                f"'{usage.description}' line each month; only usage above that allowance is billable. Credit "
                "the billed units minus the billable units, at the invoice's unit price."
            )
            params = {
                "case_id": k.id,
                "rule": "allowance",
                "used_units": used,
                "allowance_units": allowance,
                "line_description": usage.description,
            }
    else:
        raise ValueError(f"unknown F3 template {template!r}")

    if not 2 <= amount < t.amount_due:
        raise AssertionError(f"{template}: adjustment {amount} outside [2, amount_due)")
    call = ToolCall("resolve_case", {"case_id": k.id, "adjustment": amount, "note": NOTE})
    units = _units(d, template, c, t, period)
    return Built(FAMILY, template, variant, request, params, call, [c, t, k], units)


# --- distractor units -------------------------------------------------------------------------


def _case_on(d: Draft, inv: Invoice, *, resolved: bool = False) -> Case:
    """A case on ``inv``: a usage-style dispute if it has a usage line, else a line dispute."""
    line = d.rng.choice(inv.line_items)
    if line.description.startswith("Usage: ") and line.quantity > 1:
        summary = usage_summary(line, _used(d, line.quantity))
    else:
        summary = d.dispute_summary(line)
    return d.case(inv, summary, line.total, resolved=resolved)


def _units(d: Draft, template: str, c: Customer, t: Invoice, period: str) -> list[Unit]:
    rng = d.rng

    def usage_lines(o: Customer) -> list[LineItem]:
        return _usage_invoice_lines(d, o)[0]

    def other_invoice(*, usage: bool = False) -> Invoice | None:
        p = d.free_period(c.id)
        if p is None:
            return None
        return d.invoice(c, p, usage_lines(c) if usage else d.background_lines(c))

    def make_a(_max: int) -> list[Record] | None:
        """``a``: another invoice of C with a resolved case (2)."""
        inv = other_invoice(usage=rng.random() < 0.5)
        return None if inv is None else [inv, _case_on(d, inv, resolved=True)]

    def make_b(_max: int) -> list[Record]:
        """``b``: another customer with an invoice and an open case (3)."""
        o = d.customer()
        inv = d.invoice(o, rng.choice(PERIODS), usage_lines(o) if rng.random() < 0.5 else d.background_lines(o))
        return [o, inv, _case_on(d, inv)]

    def make_c(_max: int) -> list[Record] | None:
        """``c``: another invoice of C (1)."""
        inv = other_invoice()
        return None if inv is None else [inv]

    def make_d(_max: int) -> list[Record] | None:
        """``d``: an open case of C on another invoice, other month (2)."""
        inv = other_invoice(usage=template in ("F3B", "F3N", "F3X"))
        return None if inv is None else [inv, _case_on(d, inv)]

    def make_e(_max: int) -> list[Record]:
        """``e``: a resolved (rejected) case on T (1)."""
        return [_case_on(d, t, resolved=True)]

    def make_s(_max: int) -> list[Record]:
        """``s``: a customer with C's exact name and an open usage case on an invoice of another month (3)."""
        o = twin(d, c)
        p = rng.choice([x for x in PERIODS if x != period])
        inv = d.invoice(o, p, usage_lines(o))
        return [o, inv, _case_on(d, inv)]

    unit_a = UnitSpec("a", 2, make_a)
    unit_b = UnitSpec("b", 3, make_b)
    unit_c = UnitSpec("c", 1, make_c)
    unit_d = UnitSpec("d", 2, make_d)
    unit_e = UnitSpec("e", 1, make_e)
    unit_s = UnitSpec("s", 3, make_s)
    unit_f = first_name_unit(d, c)

    required = {
        "F3A": [unit_e, unit_d],
        "F3B": [unit_s, unit_d],
        "F3C": [unit_e],
        "F3N": [unit_d],
        "F3X": [unit_a],
    }[template]
    allowed = [unit_a, unit_b, unit_c, unit_d, unit_e, unit_f]
    return add_units(d, draw_k(rng), required, [*allowed, z_unit(d)], hard=False)
