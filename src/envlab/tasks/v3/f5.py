"""v3 F5 account change -> ``update_customer`` (SPEC_v3.md §4 F5).

Templates: F5A customer found by phone among same-name customers · F5B the message corrects itself ·
F5C customers share name and email local part · F5N correction and phone lookup (near) · F5X phone
given in another format, stored in the canonical format (far).
"""

from __future__ import annotations

import random

from envlab.calls import ToolCall
from envlab.sim.entities import Customer
from envlab.tasks.common import DOMAINS, PERIODS, Built, Draft, Record, Unit, UnitSpec, add_units
from envlab.tasks.f5 import new_email, wrong_value
from envlab.tasks.v3.common import draw_k, first_name_unit, local_twin, twin, z_unit

FAMILY = "F5"

# --- difficulty knobs (SPEC_v3.md §5) ---------------------------------------------------------
TWINS = (1, 2)  # same-name customers next to C (F5A, F5C, F5N)
# Field changed, drawn uniformly per template.
FIELDS: dict[str, tuple[str, ...]] = {
    "F5A": ("email", "address"),
    "F5B": ("email", "address", "phone"),
    "F5C": ("phone", "address"),
    "F5N": ("email", "address"),
    "F5X": ("phone",),
}
FIELD_LABELS: dict[str, str] = {"email": "email address", "address": "mailing address", "phone": "phone number"}
# F5B/F5N: the message first asks for a wrong value W, then corrects it. A full correction restates
# the final value V; a partial one fixes one part of W (house number, last four digits, domain), so V
# must be pieced together.
PARTIAL_CORRECTION_SHARE = 0.6
CORRECTIONS: tuple[str, ...] = (
    "Actually, scratch that - please use {v} instead.",
    "Sorry, I made a typo there: the correct one is {v}.",
    "Wait, ignore that: it should be {v}.",
)
PARTIAL_CORRECTIONS: dict[str, str] = {
    "address": "Sorry, typo: the house number is {v}, not {w}.",
    "phone": "Sorry, typo: the last four digits are {v}, not {w}.",
    "email": "Oops, the domain is {v}, not {w}.",
}
# How F5X writes a canonical "+1-AAA-555-NNNN" number in the message (area, last four digits).
PHONE_FORMATS: tuple[str, ...] = (
    "({a}) 555-{n}",
    "{a}.555.{n}",
    "{a} 555 {n}",
    "{a}555{n}",
    "1-{a}-555-{n}",
    "+1 {a} 555 {n}",
    "1 ({a}) 555-{n}",
)


def new_value(d: Draft, c: Customer, field: str) -> str:
    """A fresh valid value for ``field`` that differs from ``c``'s, also after the G3 perturbation."""
    while True:
        if field == "email":
            v = new_email(d, c)
        elif field == "address":
            v = d.address()
        else:
            v = d.phone()
        if v != getattr(c, field) and wrong_value(field, v) != getattr(c, field):
            return v


def format_phone(rng: random.Random, canonical: str) -> str:
    """``+1-AAA-555-NNNN`` written in one of ``PHONE_FORMATS`` (never the canonical form)."""
    area, last4 = canonical[3:6], canonical[-4:]
    return rng.choice(PHONE_FORMATS).format(a=area, n=last4)


def typo(rng: random.Random, digits: str) -> str:
    """``digits`` with two adjacent digits swapped (no new leading zero), else with one digit changed."""
    for _ in range(20):
        i = rng.randrange(len(digits) - 1)
        swapped = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2 :]
        if swapped != digits and swapped[0] != "0":
            return swapped
    i = rng.randrange(1, len(digits))
    return digits[:i] + str((int(digits[i]) + rng.randint(1, 9)) % 10) + digits[i + 1 :]


def partial_correction(d: Draft, field: str, value: str) -> tuple[str, str]:
    """A wrong first value W that differs from ``value`` in one part, and the sentence fixing that part.

    address: the house number · phone: the last four digits · email: the domain.
    """
    if field == "address":
        part, rest = value.split(" ", 1)
        wrong = typo(d.rng, part)
        earlier = f"{wrong} {rest}"
    elif field == "phone":
        head, part = value[:-4], value[-4:]
        wrong = typo(d.rng, part)
        earlier = head + wrong
    else:
        local, part = value.split("@")
        wrong = d.rng.choice([dom for dom in DOMAINS if dom != part])
        earlier = f"{local}@{wrong}"
    return earlier, PARTIAL_CORRECTIONS[field].format(v=part, w=wrong)


def _ask(field: str, v: str) -> str:
    return f"Please change my {FIELD_LABELS[field]} to {v}."


def build(d: Draft, template: str, variant: str) -> Built:
    rng = d.rng
    c = d.customer()
    field = rng.choice(FIELDS[template])
    value = new_value(d, c, field)

    if template in ("F5B", "F5N"):
        if rng.random() < PARTIAL_CORRECTION_SHARE:
            earlier, correction = partial_correction(d, field, value)
        else:
            earlier, correction = new_value(d, c, field), rng.choice(CORRECTIONS).format(v=value)
        asked = f"{_ask(field, earlier)} {correction}"
    else:
        asked = _ask(field, value)

    if template in ("F5A", "F5N"):
        request = f'Support inbox message: "Hi, this is {c.name}. My phone number on file is {c.phone}. {asked}"'
        lookup = {"name": c.name, "phone": c.phone}
    elif template == "F5B":
        request = f'Support inbox message from {c.email}: "Hi! {asked}"'
        lookup = {"email": c.email}
    elif template == "F5C":
        request = f'Support inbox message: "Hi, this is {c.name} ({c.email}). {asked}"'
        lookup = {"name": c.name, "email": c.email}
    elif template == "F5X":
        request = (
            f'Support inbox message from {c.email}: "Hi, please change my phone number to '
            f'{format_phone(rng, value)}." Save the number in the same format as the phone numbers already on file.'
        )
        lookup = {"email": c.email}
    else:
        raise ValueError(f"unknown F5 template {template!r}")

    params = {"lookup": lookup, "field": field, "new_value": value}
    call = ToolCall("update_customer", {"customer_id": c.id, "field": field, "value": value})
    return Built(FAMILY, template, variant, request, params, call, [c], _units(d, template, c))


# --- distractor units -------------------------------------------------------------------------


def _units(d: Draft, template: str, c: Customer) -> list[Unit]:
    rng = d.rng
    local, c_domain = c.email.split("@")

    def make_t(_max: int) -> list[Record]:
        """``t``: a customer with C's exact name, a look-alike phone and an email on another domain (1)."""
        return [twin(d, c, phone_like=True)]

    def make_l(_max: int) -> list[Record] | None:
        """``l``: a customer with C's exact name and C's email local part on another domain (1)."""
        free = [dom for dom in DOMAINS if dom != c_domain and f"{local}@{dom}" not in d.emails]
        if not free:
            return None
        first, last = c.name.split(" ")
        return [d.customer(first=first, last=last, email=d.email(local, rng.choice(free)))]

    def make_a(_max: int) -> list[Record] | None:
        """``a``: a customer with another name whose email shares C's local part (1)."""
        o = local_twin(d, c)
        return None if o is None else [o]

    def make_b(_max: int) -> list[Record] | None:
        """``b``: an invoice of C (1)."""
        p = d.free_period(c.id, PERIODS)
        return None if p is None else [d.invoice(c, p, d.background_lines(c))]

    unit_t = UnitSpec("t", 1, make_t)
    unit_l = UnitSpec("l", 1, make_l)
    unit_a = UnitSpec("a", 1, make_a)
    unit_b = UnitSpec("b", 1, make_b)
    unit_f = first_name_unit(d, c)

    n_twins = rng.randint(*TWINS)
    required = {
        "F5A": [unit_t] * n_twins,
        "F5B": [unit_l],
        "F5C": [unit_l] * n_twins,
        "F5N": [unit_t] * n_twins,
        "F5X": [unit_a],
    }[template]
    return add_units(d, draw_k(rng), required, [unit_t, unit_l, unit_a, unit_b, unit_f, z_unit(d)], hard=False)
