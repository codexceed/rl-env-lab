"""v3 F4 billing error -> ``adjust_invoice`` (SPEC_v3.md §4 F4).

Templates: F4A recompute with discount lines (customer by name + month) · F4B add sales tax at a
stated rate · F4C 4-6 lines with large quantities (customer by name) · F4N discounts and tax (near) ·
F4X minimum monthly commitment (far).
"""

from __future__ import annotations

import random

from envlab.calls import ToolCall
from envlab.sim.entities import ContractBasis, Customer, LineItem
from envlab.tasks.common import (
    DISCOUNTS,
    PERIODS,
    Built,
    Draft,
    Record,
    Unit,
    UnitSpec,
    add_units,
    dollars,
    line_sum,
    month_name,
)
from envlab.tasks.f4 import inject_error
from envlab.tasks.v3.common import addon_lines, draw_k, first_name_unit, percent_text, seat_customer, twin, z_unit

FAMILY = "F4"
REASON = "Correct total to sum of line items"

# --- difficulty knobs (SPEC_v3.md §5) ---------------------------------------------------------
DISCOUNT_ADDONS = (3, 5)  # add-on lines before the discounts (F4A, F4N)
DISCOUNT_QTY = (1, 24)
DISCOUNT_LINES = (1, 2)  # discount lines (F4A, F4N)
DISCOUNT_PCT = (5, 20)  # percentage discount line, % of the positive subtotal
FIXED_DISCOUNT = (500, 5000)  # fixed discount line, cents
TAX_ADDONS = (2, 4)  # add-on lines (F4B)
TAX_QTY = (1, 30)
TAX_BP = (425, 500, 600, 625, 725, 775, 825, 875, 950, 1025)  # tax rates, basis points
LARGE_ADDONS = (3, 5)  # add-on lines after the plan (F4C): 4-6 lines in all
LARGE_QTY = (10, 250)
LARGE_SEAT_SHARE = 0.5  # share of F4C customers on a per-seat plan
SEATS = (20, 300)
SEAT_RATE = (800, 9000)
MIN_ADDONS = (1, 3)  # add-on lines (F4X)
F4X_BELOW_SHARE = 0.7  # share of F4X instances whose line sum is below the commitment


# --- oracle rules (integer cents) -------------------------------------------------------------


def with_tax(subtotal: int, basis_points: int) -> int:
    """F4B/F4N: subtotal plus tax at ``basis_points`` / 100 %, the tax rounded down."""
    return subtotal + subtotal * basis_points // 10000


def with_minimum(subtotal: int, minimum: int) -> int:
    """F4X: the larger of the line sum and the committed minimum."""
    return max(subtotal, minimum)


# --- generator ----------------------------------------------------------------------------------


def discounted_lines(d: Draft, c: Customer) -> list[LineItem]:
    """The plan, 3-5 add-ons, then 1-2 discount lines: a % of the positive subtotal, then a fixed amount
    (at most 10% of the positive subtotal)."""
    rng = d.rng
    pos = [d.plan_line(c), *addon_lines(d, rng.randint(*DISCOUNT_ADDONS), qty=DISCOUNT_QTY)]
    names = rng.sample(DISCOUNTS, rng.randint(*DISCOUNT_LINES))
    discounts = [LineItem(names[0], 1, -(line_sum(pos) * rng.randint(*DISCOUNT_PCT) // 100))]
    if len(names) > 1:
        # capped at 10% of the positive subtotal, so the invoice total stays well above zero
        discounts.append(LineItem(names[1], 1, -min(rng.randint(*FIXED_DISCOUNT), line_sum(pos) // 10)))
    return [*pos, *discounts]


def discount_error(rng: random.Random, lines: list[LineItem]) -> int:
    """A wrong total for discounted lines: discounts omitted, one applied twice, one added as a charge,
    or the family's v1 error injection; uniformly over the kinds that give a value > 0 and != S."""
    s = line_sum(lines)
    discounts = [li.total for li in lines if li.unit_price < 0]
    kinds = {
        "omitted": [s - sum(discounts)],
        "twice": [s + x for x in discounts],
        "sign": [s - 2 * x for x in discounts],
        "injected": [inject_error(rng, lines)],
    }
    valid = {k: [v for v in vs if v > 0 and v != s] for k, vs in kinds.items()}
    names = [k for k in ("omitted", "twice", "sign", "injected") if valid[k]]
    return rng.choice(valid[rng.choice(names)])


def build(d: Draft, template: str, variant: str) -> Built:
    rng = d.rng
    period = rng.choice(PERIODS)
    month = month_name(period)
    bp = minimum = 0  # tax rate (F4B, F4N) and commitment (F4X)

    if template in ("F4A", "F4N"):
        c = d.customer()
        lines = discounted_lines(d, c)
        s = line_sum(lines)
        if template == "F4A":
            new, billed = s, discount_error(rng, lines)
        else:
            bp = rng.choice(TAX_BP)
            pos = s - sum(li.total for li in lines if li.unit_price < 0)
            new = with_tax(s, bp)
            billed = rng.choice((s, pos, with_tax(pos, bp)))
    elif template == "F4B":
        c = d.customer()
        lines = [d.plan_line(c), *addon_lines(d, rng.randint(*TAX_ADDONS), qty=TAX_QTY)]
        s = line_sum(lines)
        bp = rng.choice(TAX_BP)
        new, billed = with_tax(s, bp), s
    elif template == "F4C":
        if rng.random() < LARGE_SEAT_SHARE:
            c = seat_customer(d, rng.randint(*SEAT_RATE))
            plan = d.plan_line(c, seats=rng.randint(*SEATS))
        else:
            c = d.customer()
            plan = d.plan_line(c)
        lines = [plan, *addon_lines(d, rng.randint(*LARGE_ADDONS), qty=LARGE_QTY)]
        new = line_sum(lines)
        billed = inject_error(rng, lines)
    elif template == "F4X":
        c = d.customer()
        lines = [d.plan_line(c), *addon_lines(d, rng.randint(*MIN_ADDONS))]
        s = line_sum(lines)
        if rng.random() < F4X_BELOW_SHARE:
            minimum = 100 * (s * rng.randint(105, 160) // 10000 + 1)  # whole dollars, above s
            billed = s
        else:
            minimum = 100 * (s * rng.randint(40, 90) // 10000)  # whole dollars, below s
            billed = rng.choice((minimum, inject_error(rng, lines)))
        new = with_minimum(s, minimum)
        if billed == new:
            billed = minimum if new == s else s
    else:
        raise ValueError(f"unknown F4 template {template!r}")

    t = d.invoice(c, period, lines, billed=billed)
    if template == "F4A":
        request = (
            f"{c.name}'s {month} invoice has the wrong total. Recompute it from the line items, including discounts."
        )
        params = {"customer_name": c.name, "period": period, "rule": "sum"}
    elif template == "F4B":
        request = (
            f"Invoice {t.id} for {c.name} is missing sales tax. Set its total to the sum of its line items plus "
            f"{percent_text(bp)} tax on that sum, with the tax rounded down to the cent."
        )
        params = {"invoice_id": t.id, "rule": "tax", "tax_bp": bp}
    elif template == "F4C":
        request = f"One of {c.name}'s open invoices has a total that doesn't match its line items. Correct it."
        params = {"customer_name": c.name, "rule": "sum"}
    elif template == "F4N":
        request = (
            f"Invoice {t.id} ({c.name}) has the wrong total. Recompute it from the line items, including "
            f"discounts, then add {percent_text(bp)} sales tax on that amount, with the tax rounded down to the cent."
        )
        params = {"invoice_id": t.id, "rule": "tax", "tax_bp": bp}
    else:
        request = (
            f"{c.name}'s contract has a minimum monthly commitment of {dollars(minimum)}: each invoice's total is "
            f"the sum of its line items or the commitment, whichever is higher. Invoice {t.id} does not follow "
            "this rule. Correct its total."
        )
        params = {"invoice_id": t.id, "rule": "minimum", "minimum": minimum}

    if new == billed or new <= 0 or billed <= 0:
        raise AssertionError(f"{template}: new {new} vs billed {billed}")
    call = ToolCall("adjust_invoice", {"invoice_id": t.id, "new_amount": new, "reason": REASON})
    units = _units(d, template, c, lines, period, minimum)
    return Built(FAMILY, template, variant, request, params, call, [c, t], units)


# --- distractor units -------------------------------------------------------------------------


def _units(d: Draft, template: str, c: Customer, lines: list[LineItem], period: str, minimum: int) -> list[Unit]:
    rng = d.rng
    per_seat = c.contract_basis is ContractBasis.PER_SEAT

    def own_lines(o: Customer) -> list[LineItem]:
        if template in ("F4A", "F4N") and rng.random() < 0.5:
            return discounted_lines(d, o)
        seats = rng.randint(*SEATS) if o.contract_basis is ContractBasis.PER_SEAT else None
        return [d.plan_line(o, seats=seats), *addon_lines(d, rng.randint(0, 3), qty=(1, 12))]

    def make_a(max_records: int, *, open_only: bool = False) -> list[Record] | None:
        """``a``: another invoice of C summing correctly: open (1) or paid (2)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        if not open_only and max_records >= 2 and rng.random() < 0.5:
            return d.paid_once(c, p, own_lines(c))
        return [d.invoice(c, p, own_lines(c))]

    def make_b(_max: int) -> list[Record]:
        """``b``: another customer with a correct invoice (2)."""
        o = d.customer()
        return [o, d.invoice(o, rng.choice(PERIODS), own_lines(o))]

    def make_c(_max: int) -> list[Record] | None:
        """``c``: an open invoice of C with T's lines at other quantities, summing correctly (1)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        same = []
        for li in lines:
            fixed = li.unit_price < 0 or (li.description.startswith("Plan: ") and not per_seat)
            qty = li.quantity if fixed else max(1, li.quantity + rng.randint(-5, 5))
            same.append(LineItem(li.description, qty, li.unit_price))
        if line_sum(same) <= 0:
            return None
        return [d.invoice(c, p, same)]

    def make_s(_max: int) -> list[Record]:
        """``s``: a customer with C's exact name and a correct open invoice (F4A: in C's month) (2)."""
        o = twin(d, c, rate=rng.randint(*SEAT_RATE) if per_seat else d.flat_rate())
        p = period if template == "F4A" else rng.choice(PERIODS)
        return [o, d.invoice(o, p, own_lines(o))]

    def make_m(_max: int) -> list[Record] | None:
        """``m`` (F4X): another open invoice of C billed per the commitment rule (1)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        ls = [d.plan_line(c), *addon_lines(d, rng.randint(*MIN_ADDONS))]
        return [d.invoice(c, p, ls, billed=with_minimum(line_sum(ls), minimum))]

    unit_a = UnitSpec("a", 1, make_a)
    unit_a_open = UnitSpec("a", 1, lambda m: make_a(m, open_only=True))
    unit_b = UnitSpec("b", 2, make_b)
    unit_c = UnitSpec("c", 1, make_c)
    unit_s = UnitSpec("s", 2, make_s)
    unit_m = UnitSpec("m", 1, make_m)
    unit_f = first_name_unit(d, c)

    required = {
        "F4A": [unit_s, unit_a_open],
        "F4B": [unit_a],
        "F4C": [unit_s, unit_a_open],
        "F4N": [unit_a],
        "F4X": [unit_m],
    }[template]
    allowed = [unit_a, unit_b, unit_c, unit_f, unit_s]
    if template == "F4X":
        allowed.append(unit_m)
    return add_units(d, draw_k(rng), required, [*allowed, z_unit(d)], hard=False)
