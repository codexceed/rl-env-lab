"""F5 Account change -> ``update_customer`` (SPEC §4 F5, §5)."""

from envlab.calls import ToolCall
from envlab.sim.entities import Customer
from envlab.tasks.common import (
    DOMAINS,
    PERIODS,
    Built,
    Draft,
    Record,
    UnitSpec,
    add_units,
    draw_k,
    z_unit,
)

FAMILY = "F5"


def wrong_value(field: str, value: str) -> str:
    """The G3 scripted wrong value (SPEC §7, F5 row)."""
    if field == "email":
        return "x" + value
    if field == "address":
        return value + " Apt 2"
    return value[:-1] + str((int(value[-1]) + 1) % 10)


def new_email(d: Draft, c: Customer) -> str:
    first, last = (s.lower() for s in c.name.split(" "))
    while True:
        local = d.rng.choice((f"{first}{last}", f"{first[0]}.{last}", f"{first}.{last}.home"))
        v = d.email(local)
        if wrong_value("email", v) != c.email:
            return v


def new_phone(d: Draft, c: Customer) -> str:
    while True:
        v = d.phone()
        if wrong_value("phone", v) != c.phone:
            return v


def build(d: Draft, template: str, variant: str) -> Built:
    hard = variant == "hard"
    rng = d.rng
    c = d.customer()
    first = c.name.split(" ")[0]

    if template == "F5T1":
        field, value = "email", new_email(d, c)
        if hard:
            request = f'Customer {c.id} wrote: "Mail still goes to {c.email}. Please switch my email to {value}."'
        else:
            request = f"Customer {c.id} asked to change their email to {value}."
        params = {"customer_id": c.id, "field": field, "new_value": value}
    elif template == "F5T2":
        field, value = "address", d.address()  # registered, so differs from every address
        if hard:
            request = (
                f'Support inbox message: "Hi, this is {c.name} ({c.email}). My phone {c.phone} '
                f"stays the same, but I've moved: please update my address to {value}.\""
            )
        else:
            request = (
                f"Support inbox message: \"Hi, this is {c.name} ({c.email}). I've moved, "
                f'please update my address to {value}."'
            )
        params = {"lookup": {"name": c.name, "email": c.email}, "field": field, "new_value": value}
    else:
        field, value = "phone", new_phone(d, c)
        noise = d.phone()  # reserved, so no record in the state gets it
        request = (
            f'Support inbox message: "Please change the phone number on my account from '
            f"{c.phone} to {value}. If it's urgent today, call my office at {noise}. "
            f'Thanks, {first}"'
        )
        params = {"lookup": {"phone": c.phone}, "field": field, "new_value": value}
    assert wrong_value(field, value) != getattr(c, field)

    call = ToolCall("update_customer", {"customer_id": c.id, "field": field, "value": value})

    local, c_domain = c.email.split("@")

    def make_a(_max: int) -> list[Record] | None:
        """Another customer whose email shares C's local part, other domain (1)."""
        free = [dom for dom in DOMAINS if dom != c_domain and f"{local}@{dom}" not in d.emails]
        if not free:
            return None
        return [d.customer(email=f"{local}@{rng.choice(free)}")]

    def make_b(_max: int) -> list[Record] | None:
        """An invoice of C (1)."""
        p = d.free_period(c.id, PERIODS)
        return None if p is None else [d.invoice(c, p, d.background_lines(c))]

    def make_c(_max: int) -> list[Record]:
        """A customer with C's exact name, different email and phone (1)."""
        f, last = c.name.split(" ")
        return [d.customer(first=f, last=last)]

    unit_a = UnitSpec("a", 1, make_a)
    unit_b = UnitSpec("b", 1, make_b)
    unit_c = UnitSpec("c", 1, make_c)
    unit_d = UnitSpec("d", 1, lambda _m: [d.customer(first=first)])

    required: list[UnitSpec] = []
    if template == "F5T1" and hard:
        required = [unit_a, unit_d]
    elif template == "F5T2":
        required = [unit_c, unit_a] if hard else [unit_c]
    elif template == "F5T3":
        required = [unit_d]

    units = add_units(d, draw_k(rng, variant), required, [unit_a, unit_b, unit_c, unit_d, z_unit(d)], hard=hard)
    return Built(FAMILY, template, variant, request, params, call, [c], units)
