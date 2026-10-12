"""F3 Dispute -> ``resolve_case`` (SPEC §4 F3, §5)."""

from envlab.calls import ToolCall
from envlab.sim.entities import Case, Customer, Invoice, LineItem
from envlab.tasks.common import (
    PERIODS,
    USAGE_LINE,
    Built,
    Draft,
    Record,
    UnitSpec,
    add_units,
    draw_k,
    other_customer_unit,
    z_unit,
)

FAMILY = "F3"
NOTE = "Dispute resolved per policy"
PERCENTS = (10, 20, 25, 30, 40, 50, 60, 75)
HARD_PERCENTS = (15, 35, 45, 65, 85)


def adjustment(params: dict, case: Case, invoice: Invoice) -> int:
    """Oracle rule for A under ``params.rule``."""
    if params["rule"] == "percent":
        return (case.disputed_amount * params["percent"]) // 100
    line = next(li for li in invoice.line_items if li.description == params["line_description"])
    return (line.quantity - params["used_units"]) * line.unit_price


def disputed_invoice_lines(d: Draft, c: Customer) -> tuple[list[LineItem], LineItem]:
    """T with 2–4 lines: the plan, the disputed add-on L (qty 1–3), then 0–2 more add-ons."""
    disputed = d.addons(1, qty=(1, 3))[0]
    others = d.addons(d.rng.randint(0, 2), exclude=[disputed.description])
    rest = [disputed, *others]
    d.rng.shuffle(rest)
    return [d.plan_line(c), *rest], disputed


def open_case_on(d: Draft, inv: Invoice, *, resolved: bool = False) -> Case:
    """A case disputing one random line of ``inv`` in full."""
    line = d.rng.choice(inv.line_items)
    return d.case(inv, d.dispute_summary(line), line.total, resolved=resolved)


def build(d: Draft, template: str, variant: str) -> Built:
    hard = variant == "hard"
    rng = d.rng
    c = d.customer()
    period = rng.choice(PERIODS)

    if template == "F3T3":
        x = rng.randint(20, 400)
        u = rng.randint(5, 300)
        y = rng.randint(-(-x // 4), x - 1)
        usage = LineItem(USAGE_LINE, x, u)
        lines = [d.plan_line(c), usage, *d.addons(rng.randint(0, 1))]
        t = d.invoice(c, period, lines)
        k = d.case(t, f"Customer says they used only {y}k API calls, not {x}k.", usage.total)
        params = {
            "case_id": k.id,
            "rule": "unused_units",
            "used_units": y,
            "line_description": USAGE_LINE,
        }
        request = f"Resolve case {k.id}. Credit only the unused units, at the invoice's unit price."
    else:
        lines, disputed = disputed_invoice_lines(d, c)
        t = d.invoice(c, period, lines)
        k = d.case(t, d.dispute_summary(disputed), disputed.total)
        if template == "F3T1" and hard:
            params = {"invoice_id": t.id, "rule": "percent", "percent": 100}
            request = f"Resolve the open dispute on invoice {t.id}. Approve it in full."
        elif template == "F3T1":
            params = {"case_id": k.id, "rule": "percent", "percent": 100}
            request = f"Resolve case {k.id}. Approve the dispute in full."
        else:
            pct = rng.choice(HARD_PERCENTS if hard else PERCENTS)
            params = {"case_id": k.id, "rule": "percent", "percent": pct}
            request = (
                f"Resolve case {k.id}. Policy: approve {pct}% of the disputed amount, "
                "rounded down to the cent."
            )

    amount = adjustment(params, k, t)
    call = ToolCall("resolve_case", {"case_id": k.id, "adjustment": amount, "note": NOTE})

    def other_invoice() -> Invoice | None:
        p = d.free_period(c.id)
        return None if p is None else d.invoice(c, p, d.background_lines(c))

    def make_a(_max: int) -> list[Record] | None:
        """Another invoice of C with a resolved case (2)."""
        inv = other_invoice()
        return None if inv is None else [inv, open_case_on(d, inv, resolved=True)]

    def b_build(o: Customer) -> list[Record]:
        inv = d.invoice(o, rng.choice(PERIODS), d.background_lines(o))
        return [inv, open_case_on(d, inv)]

    def make_c(_max: int) -> list[Record] | None:
        """Another invoice of C (1)."""
        inv = other_invoice()
        return None if inv is None else [inv]

    def make_d(_max: int) -> list[Record] | None:
        """(hard) An open case of C on a different invoice (2)."""
        inv = other_invoice()
        return None if inv is None else [inv, open_case_on(d, inv)]

    def make_e(_max: int) -> list[Record]:
        """(hard) A resolved case on T (1)."""
        return [open_case_on(d, t, resolved=True)]

    unit_a = UnitSpec("a", 2, make_a)
    unit_b = other_customer_unit(d, "b", 3, b_build)
    unit_c = UnitSpec("c", 1, make_c)
    unit_d = UnitSpec("d", 2, make_d, hard_only=True)
    unit_e = UnitSpec("e", 1, make_e, hard_only=True)

    required: list[UnitSpec] = []
    if hard and template == "F3T1":
        required = [unit_e, unit_d]
    elif hard and template == "F3T2":
        required = [unit_d]

    units = add_units(
        d,
        draw_k(rng, variant),
        required,
        [unit_a, unit_b, unit_c, unit_d, unit_e, z_unit(d)],
        hard=hard,
    )
    return Built(FAMILY, template, variant, request, params, call, [c, t, k], units)

