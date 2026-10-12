"""F4 Billing error -> ``adjust_invoice`` (SPEC §4 F4, §5)."""

import random

from envlab.calls import ToolCall
from envlab.sim.entities import Customer, LineItem
from envlab.tasks.common import (
    DISCOUNTS,
    PERIODS,
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

FAMILY = "F4"
REASON = "Correct total to sum of line items"

# (number of lines, add-on quantity range) per template/variant.
_SHAPES: dict[tuple[str, str], tuple[tuple[int, int], tuple[int, int]]] = {
    ("F4T1", "base"): ((2, 3), (1, 1)),
    ("F4T2", "base"): ((3, 4), (1, 12)),
    ("F4T3", "base"): ((3, 4), (1, 12)),
    ("F4T1", "hard"): ((4, 5), (1, 9)),
    ("F4T2", "hard"): ((4, 5), (1, 24)),
}


def inject_error(rng: random.Random, lines: list[LineItem]) -> int:
    """A wrong billed total for ``lines`` (SPEC §4 F4 "Error injection").

    Reading: pick one of the four error *kinds* uniformly among those with at least one
    candidate giving billed > 0 and ≠ S, then one candidate of that kind uniformly. Digit
    swaps that would create a leading zero are not counted as candidates.
    """
    s = line_sum(lines)
    digits = str(s)
    swaps = []
    for i in range(len(digits) - 1):
        sw = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2 :]
        if sw[0] != "0":
            swaps.append(int(sw))
    kinds = {
        "swap": swaps,
        "double": [s + li.total for li in lines],
        "drop": [s - li.total for li in lines[1:]],  # lines[0] is the plan line
        "shift": [s + sign * 100 * k for k in range(1, 10) for sign in (1, -1)],
    }
    valid = {k: [v for v in vs if v > 0 and v != s] for k, vs in kinds.items()}
    names = [k for k in ("swap", "double", "drop", "shift") if valid[k]]
    return rng.choice(valid[rng.choice(names)])


def mis_summed_lines(d: Draft, c: Customer, template: str, variant: str) -> list[LineItem]:
    """T's line items: the plan (qty 1, flat C) then add-ons with the template's qty range.

    "qty 1–12" etc. is read as applying to the add-on lines; the plan line keeps the
    generator convention (flat customer, qty 1, priced at contract_rate).
    """
    (lo, hi), qty = _SHAPES[(template, variant)]
    return [d.plan_line(c), *d.addons(d.rng.randint(lo, hi) - 1, qty=qty)]


def discounted_lines(d: Draft, c: Customer) -> list[LineItem]:
    """F4T3: 3–4 positive lines (qty 1–12) then a discount of 5–20% of the positive subtotal."""
    pos = mis_summed_lines(d, c, "F4T3", "base")
    pct = d.rng.randint(5, 20)
    disc = max(1, line_sum(pos) * pct // 100)
    return [*pos, LineItem(d.rng.choice(DISCOUNTS), 1, -disc)]


def build(d: Draft, template: str, variant: str) -> Built:
    hard = variant == "hard"
    rng = d.rng
    c = d.customer()
    period = rng.choice(PERIODS)

    if template == "F4T3":
        lines = discounted_lines(d, c)
        s = line_sum(lines)
        discount = lines[-1].total
        # Discount omitted (billed = positive subtotal) or applied twice (S + discount).
        billed = rng.choice((s - discount, s + discount))
    else:
        lines = mis_summed_lines(d, c, template, variant)
        s = line_sum(lines)
        billed = inject_error(rng, lines)
    t = d.invoice(c, period, lines, billed=billed)
    call = ToolCall("adjust_invoice", {"invoice_id": t.id, "new_amount": s, "reason": REASON})

    if template == "F4T1":
        request = (
            f"Invoice {t.id} for {c.name} doesn't match its line items. "
            "Correct the invoice total."
        )
        params = {"invoice_id": t.id}
    elif template == "F4T2":
        request = (
            f"One of {c.name}'s open invoices has a total that doesn't match its line items. "
            "Correct it."
        )
        params = {"customer_name": c.name}
    else:
        request = (
            f"Invoice {t.id} ({c.name}) has the wrong total. Recompute it from the line items, "
            "including discounts."
        )
        params = {"invoice_id": t.id}

    def make_a(max_records: int, *, open_only: bool = False) -> list[Record] | None:
        """Another invoice of C summing correctly: open (1) or paid (2)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        return d.background_invoice(c, p, allow_paid=not open_only, max_records=max_records)

    def b_build(o: Customer) -> list[Record]:
        return [d.invoice(o, rng.choice(PERIODS), d.background_lines(o))]

    def make_c(_max: int) -> list[Record] | None:
        """(hard) An open invoice of C with T's line descriptions, summing correctly (1)."""
        p = d.free_period(c.id)
        if p is None:
            return None
        _, qty = _SHAPES[(template, variant)]
        same = [lines[0]] + [
            LineItem(li.description, rng.randint(*qty), li.unit_price) for li in lines[1:]
        ]
        return [d.invoice(c, p, same)]

    unit_a = UnitSpec("a", 1, make_a)
    unit_a_open = UnitSpec("a", 1, lambda m: make_a(m, open_only=True))
    unit_b = other_customer_unit(d, "b", 2, b_build)
    unit_c = UnitSpec("c", 1, make_c, hard_only=True)

    required: list[UnitSpec] = []
    if template == "F4T2":
        required = [unit_c, unit_a_open] if hard else [unit_a_open]
    elif template == "F4T1" and hard:
        required = [unit_c]

    units = add_units(
        d, draw_k(rng, variant), required, [unit_a, unit_b, unit_c, z_unit(d)], hard=hard
    )
    return Built(FAMILY, template, variant, request, params, call, [c, t], units)
