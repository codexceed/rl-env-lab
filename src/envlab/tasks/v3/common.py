"""Shared building blocks for the v3 generators (SPEC_v3.md §4).

v3 reuses the v1 machinery in ``envlab.tasks.common`` (``Draft``, units, ``finalize``) and adds:
the distractor-count knob, near-duplicate customers (same full name, same email local part,
look-alike phone numbers), installment splits, and a few text helpers. Difficulty knobs that a
tuning pass may change (SPEC_v3.md §5) are module-level constants in UPPER_CASE, here and in
each family module.
"""

from __future__ import annotations

import calendar
import random
from dataclasses import replace
from datetime import datetime, timedelta

from envlab.sim.entities import ContractBasis, Customer, Invoice, LineItem, Payment, PaymentStatus
from envlab.tasks.common import (
    ADDONS,
    AREA_CODES,
    DOMAINS,
    FLAT_PLANS,
    MONTH_NAMES,
    REFUND_NOTE,
    Built,
    Draft,
    Record,
    UnitSpec,
    finalize,
    line_sum,
)
from envlab.tasks.instance import TaskInstance
from envlab.tasks.v3 import SPEC_VERSION

# ---------------------------------------------------------------------------------------------
# Difficulty knobs shared by every family (SPEC_v3.md §5)
# ---------------------------------------------------------------------------------------------

# Distractor record count k, drawn uniformly (v1 base: {2, 3, 4}).
K_CHOICES: tuple[int, ...] = (4, 5, 6, 7)
# Trimming to the prompt budget never drops below this many distractor records.
MIN_DISTRACTORS = 4
# Add-on unit price range (cents) and the add-on quantity range on multi-quantity lines.
ADDON_UNIT = (500, 5000)

# Usage lines (F3). Each is billed per unit; the case summary quotes billed and used units.
USAGE_LINES: tuple[str, ...] = (
    "Usage: API calls (per 1k)",
    "Usage: Storage (per GB)",
    "Usage: SMS messages",
    "Usage: Compute hours",
    "Usage: Data transfer (per GB)",
)


def draw_k(rng: random.Random) -> int:
    return rng.choice(K_CHOICES)


def finalize_v3(b: Built, rng: random.Random, seed: int, split: str) -> TaskInstance:
    """``finalize`` with the v3 prompt budget, distractor floor and spec version."""
    from envlab.prompts import V3_ESTIMATE_BUDGET

    return finalize(
        b,
        rng,
        seed,
        split,
        budget=V3_ESTIMATE_BUDGET,
        min_distractors=MIN_DISTRACTORS,
        spec_version=SPEC_VERSION,
    )


# ---------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------


def days_in_month(period: str) -> int:
    year, month = (int(x) for x in period.split("-"))
    return calendar.monthrange(year, month)[1]


def long_date(period: str, day: int) -> str:
    """``("2026-08", 19)`` -> ``"August 19, 2026"``."""
    year, month = period.split("-")
    return f"{MONTH_NAMES[int(month) - 1]} {day}, {year}"


def percent_text(basis_points: int) -> str:
    """``825`` -> ``"8.25%"``, ``650`` -> ``"6.5%"``, ``600`` -> ``"6%"``."""
    whole, frac = divmod(basis_points, 100)
    if frac == 0:
        return f"{whole}%"
    return f"{whole}.{frac:02d}".rstrip("0") + "%"


def first_name(c: Customer) -> str:
    return c.name.split(" ")[0]


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


def seat_customer(d: Draft, rate: int) -> Customer:
    return d.customer(basis=ContractBasis.PER_SEAT, rate=rate)


def addon_lines(d: Draft, n: int, qty: tuple[int, int] = (1, 1), exclude: tuple[str, ...] = ()) -> list[LineItem]:
    names = [a for a in ADDONS if a not in exclude]
    return [LineItem(name, d.rng.randint(*qty), d.rng.randint(*ADDON_UNIT)) for name in d.rng.sample(names, n)]


def other_flat_plan(c: Customer) -> str:
    """A flat plan name other than the one ``Draft.plan_line`` gives ``c`` (F1C's old plan)."""
    own = FLAT_PLANS[int(c.id[1:]) % len(FLAT_PLANS)]
    return FLAT_PLANS[(FLAT_PLANS.index(own) + 1 + int(c.id[1:]) % (len(FLAT_PLANS) - 1)) % len(FLAT_PLANS)]


def similar_phone(d: Draft, phone: str) -> str:
    """A fresh phone that looks like ``phone``: same area code, last four digits nearly the same."""
    head, last4 = phone[: -len("NNNN")], phone[-4:]
    for _ in range(1000):
        digits = list(last4)
        if d.rng.random() < 0.5:
            i = d.rng.randrange(3)
            digits[i], digits[i + 1] = digits[i + 1], digits[i]
        else:
            i = d.rng.randrange(4)
            digits[i] = str((int(digits[i]) + d.rng.randint(1, 9)) % 10)
        cand = head + "".join(digits)
        if cand != phone and cand not in d.phones:
            d.phones.add(cand)
            return cand
    raise RuntimeError("no look-alike phone left")  # pragma: no cover


def twin(d: Draft, c: Customer, *, phone_like: bool = False, rate: int | None = None) -> Customer:
    """A different customer with ``c``'s exact name.

    Its email shares ``c``'s local part on another domain when one is free (``Draft.email``'s
    default), else gets a numbered local part. ``phone_like`` gives it a look-alike phone.
    """
    first, last = c.name.split(" ")
    t = d.customer(first=first, last=last, basis=c.contract_basis, rate=c.contract_rate if rate is None else rate)
    if phone_like:
        d.phones.discard(t.phone)
        t = replace(t, phone=similar_phone(d, c.phone))
    return t


def local_twin(d: Draft, c: Customer) -> Customer | None:
    """A customer with a different name whose email shares ``c``'s local part on another domain."""
    local, domain = c.email.split("@")
    free = [dom for dom in DOMAINS if dom != domain and f"{local}@{dom}" not in d.emails]
    if not free:
        return None
    return d.customer(email=d.email(local, d.rng.choice(free)))


def canonical_phone(area: int, last4: int) -> str:
    """The format every generated phone uses (SPEC.md §1): ``+1-AAA-555-NNNN``."""
    return f"+1-{area}-555-{last4:04d}"


def phone_area(d: Draft) -> int:
    return d.rng.choice(AREA_CODES)


# ---------------------------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------------------------


def split_amount(rng: random.Random, total: int, parts: int) -> list[int]:
    """``parts`` distinct positive amounts summing to ``total``, each at least 10% of it."""
    lo = max(1, total // 10)
    for _ in range(1000):
        cuts = sorted(rng.sample(range(1, total), parts - 1))
        amounts = [b - a for a, b in zip((0, *cuts), (*cuts, total), strict=True)]
        if min(amounts) >= lo and len(set(amounts)) == parts:
            return amounts
    raise RuntimeError(f"cannot split {total} into {parts} distinct parts")  # pragma: no cover


def later(rng: random.Random, ts: datetime, lo_min: int, hi_min: int) -> datetime:
    return ts + timedelta(minutes=rng.randint(lo_min, hi_min))


def installments(d: Draft, inv: Invoice, amounts: list[int], start: datetime) -> list[Payment]:
    """One succeeded payment per amount, 1 hour to 3 days apart, in order."""
    out, ts = [], start
    for amt in amounts:
        out.append(d.payment(inv, amt, ts))
        ts = later(d.rng, ts, 60, 3 * 1440)
    return out


def refunded(p: Payment) -> Payment:
    """A background refunded payment (SPEC.md §1 convention note)."""
    return replace(p, status=PaymentStatus.REFUNDED, note=REFUND_NOTE)


# ---------------------------------------------------------------------------------------------
# Units shared by several families
# ---------------------------------------------------------------------------------------------


def z_unit(d: Draft) -> UnitSpec:
    return UnitSpec("z", 1, lambda _m: [d.customer()])


def first_name_unit(d: Draft, c: Customer) -> UnitSpec:
    """``f``: a customer sharing ``c``'s first name (1 rec)."""
    return UnitSpec("f", 1, lambda _m: [d.customer(first=first_name(c))])


def paid_installments(d: Draft, c: Customer, period: str, lines: list[LineItem]) -> list[Record]:
    """An invoice paid in full by 2-3 unequal installments (3-4 records)."""
    billed = line_sum(lines)
    inv = d.invoice(c, period, lines, paid=billed)
    return [inv, *installments(d, inv, split_amount(d.rng, billed, d.rng.randint(2, 3)), d.pay_time(period))]
