"""v3 broken variants: invalid instances the gates must reject (SPEC_v3.md §8).

``impossible``, ``grader_bug`` and ``env_bug`` are v1's family-level mutations (SPEC.md §8,
``envlab.tasks.broken.make_broken``) applied to v3 base instances. ``ambiguous`` needs a second
valid target under v3 params, so it is v3's own.
"""

from __future__ import annotations

import dataclasses
import random

from envlab.sim.entities import Case, Customer, Invoice, Payment, State
from envlab.tasks.broken import BROKEN_TYPES, broken_id, make_broken
from envlab.tasks.common import PERIODS, Draft, oracle_final, with_ledger
from envlab.tasks.f4 import inject_error
from envlab.tasks.generate import derive_seed
from envlab.tasks.instance import TaskInstance
from envlab.tasks.v3.common import addon_lines
from envlab.tasks.v3.f2 import duplicated
from envlab.tasks.v3.generate import BROKEN_SEED, generate

PER_TEMPLATE = 5  # 5 per family per type -> 25 per type, 100 in all

# Source templates per type: train templates only, so held-out templates are never mutated.
SOURCES: dict[str, tuple[str, ...]] = {
    "impossible": ("F1A", "F2A", "F3A", "F4A", "F5A"),
    "ambiguous": ("F1A", "F2C", "F3A", "F4C", "F5C"),
    "grader_bug": ("F1B", "F2B", "F3B", "F4B", "F5B"),
    "env_bug": ("F1C", "F2C", "F3C", "F4C", "F5C"),
}


def _require[T](value: T | None, what: str) -> T:
    if value is None:
        raise ValueError(f"base instance has no {what}")
    return value


def ambiguous(base: TaskInstance, rng: random.Random) -> TaskInstance:
    """Add a second record that satisfies the base's params and rule; recompute the oracle final state.

    F1A: another per-seat overcharged open invoice of C in the same month · F2C: another invoice of C
    charged twice · F3A: a second open case on T · F4C: another open mis-summed invoice of C · F5C:
    another customer with C's name and email.
    """
    s, call = base.initial_state, base.oracle_call
    d = Draft(rng, existing=s)
    new_cust: list[Customer] = []
    new_inv: list[Invoice] = []
    new_pay: list[Payment] = []
    new_case: list[Case] = []
    if base.family == "F1":
        t = _require(s.invoice(str(call.arguments["invoice_id"])), "target invoice")
        c = _require(s.customer(t.customer_id), "customer")
        plan = t.line_items[0]
        over = d.plan_line(c, seats=plan.quantity + rng.randint(1, 9), unit=plan.unit_price)
        new_inv.append(d.invoice(c, t.period, [over, *addon_lines(d, rng.randint(0, 2))]))
    elif base.family == "F2":
        dup = _require(s.payment(str(call.arguments["payment_id"])), "target payment")
        c = _require(s.customer(dup.customer_id), "customer")
        inv, p1, p2 = duplicated(d, c, d.free_period(c.id) or rng.choice(PERIODS), d.background_lines(c))
        new_inv.append(inv)
        new_pay += [p1, p2]
    elif base.family == "F3":
        k = _require(s.case(str(call.arguments["case_id"])), "target case")
        t = _require(s.invoice(k.invoice_id), "case invoice")
        line = rng.choice([li for li in t.line_items if f"'{li.description}'" not in k.summary])
        new_case.append(d.case(t, d.dispute_summary(line), line.total))
    elif base.family == "F4":
        t = _require(s.invoice(str(call.arguments["invoice_id"])), "target invoice")
        c = _require(s.customer(t.customer_id), "customer")
        lines = [d.plan_line(c, seats=t.line_items[0].quantity), *addon_lines(d, rng.randint(2, 4), qty=(10, 99))]
        new_inv.append(d.invoice(c, d.free_period(c.id) or rng.choice(PERIODS), lines, billed=inject_error(rng, lines)))
    else:
        c = _require(s.customer(str(call.arguments["customer_id"])), "target customer")
        new_cust.append(dataclasses.replace(d.customer(), name=c.name, email=c.email))
    state = with_ledger(
        State(
            customers=(*s.customers, *new_cust),
            invoices=(*s.invoices, *new_inv),
            payments=(*s.payments, *new_pay),
            cases=(*s.cases, *new_case),
        )
    )
    return dataclasses.replace(base, initial_state=state, oracle_final_state=oracle_final(state, call))


def make_broken_v3(base: TaskInstance, kind: str, seed: int) -> TaskInstance:
    """A broken variant of a valid v3 ``base`` (SPEC_v3.md §8). The id is provisional."""
    if kind != "ambiguous":
        return make_broken(base, kind, seed)
    out = ambiguous(base, random.Random(derive_seed(seed, "mutate")))
    return dataclasses.replace(
        out, id=f"{base.template}-broken-{kind}", split="broken", variant="broken", broken_type=kind, seed=seed
    )


def generate_broken() -> list[TaskInstance]:
    """The 100 v3 broken instances: 4 types x 5 families x 5, ordered by type, template, index."""
    out = []
    for kind in BROKEN_TYPES:
        for template in SOURCES[kind]:
            for index in range(PER_TEMPLATE):
                seed = derive_seed(BROKEN_SEED, kind, template, index)
                inst = make_broken_v3(generate(template, seed), kind, seed)
                out.append(dataclasses.replace(inst, id=broken_id(template, kind, index)))
    return out
