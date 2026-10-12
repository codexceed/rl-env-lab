"""Shared building blocks for the F1–F5 task generators (SPEC §4).

A template generator builds a ``Draft``: the core records, the oracle call and a list of
distractor *units* (SPEC §4 "Units"). ``finalize`` then trims units to the prompt budget
(§10), shuffles every record tuple, builds the ledger with the simulator, and computes the
oracle final state. All randomness comes from the ``random.Random`` passed in; nothing here
touches global RNG state.
"""

import dataclasses
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from envlab.calls import ToolCall
from envlab.sim.entities import (
    Case,
    CaseStatus,
    ContractBasis,
    Customer,
    Invoice,
    InvoiceStatus,
    LineItem,
    Payment,
    PaymentStatus,
    State,
)
from envlab.tasks.instance import TaskInstance

SPEC_VERSION = "1.0"

Record = Customer | Invoice | Payment | Case

# ---------------------------------------------------------------------------------------------
# Fixed vocab (synthetic only)
# ---------------------------------------------------------------------------------------------

PERIODS: tuple[str, ...] = (
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
    "2026-07",
    "2026-08",
    "2026-09",
)
MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DOMAINS: tuple[str, ...] = ("example.com", "example.net", "example.org")

FIRST_NAMES: tuple[str, ...] = (
    "Ada", "Alan", "Alma", "Amir", "Anya", "Ben", "Bea", "Carl", "Cora", "Dana",
    "Dev", "Eli", "Emma", "Ezra", "Faye", "Finn", "Gia", "Gus", "Hana", "Hugo",
    "Ines", "Ivan", "Jade", "Joel", "Kai", "Kira", "Leo", "Lena", "Luca", "Maya",
    "Max", "Mila", "Nate", "Nia", "Noah", "Nora", "Omar", "Owen", "Pia", "Raj",
    "Rosa", "Ravi", "Sam", "Sara", "Tara", "Theo", "Uma", "Vera", "Wes", "Yara",
    "Zane", "Zoe", "Iris", "Jon", "Lara", "Milo", "Petra", "Quinn", "Reza", "Sofia",
)  # fmt: skip
LAST_NAMES: tuple[str, ...] = (
    "Abbott", "Banks", "Bauer", "Chen", "Cruz", "Dahl", "Diaz", "Ellis", "Ford", "Garcia",
    "Grant", "Hale", "Ito", "Jensen", "Khan", "Kim", "Lane", "Lopez", "Meyer", "Mora",
    "Nash", "Novak", "Okafor", "Ortiz", "Park", "Patel", "Quinn", "Reyes", "Rossi", "Sato",
    "Shah", "Silva", "Stone", "Tan", "Torres", "Vega", "Walsh", "Weber", "Young", "Zhou",
)  # fmt: skip
STREETS: tuple[str, ...] = (
    "Oak", "Maple", "Cedar", "Pine", "Elm", "Birch", "Lake", "Hill", "River", "Park",
    "Mill", "Main", "Spring", "Ridge", "Grove", "Willow",
)  # fmt: skip
STREET_SUFFIXES: tuple[str, ...] = ("St", "Ave", "Rd", "Ln", "Dr", "Way")
CITIES: tuple[tuple[str, str], ...] = (
    ("Salem", "OR"),
    ("Dayton", "OH"),
    ("Austin", "TX"),
    ("Boise", "ID"),
    ("Tulsa", "OK"),
    ("Reno", "NV"),
    ("Mesa", "AZ"),
    ("Provo", "UT"),
    ("Fargo", "ND"),
    ("Akron", "OH"),
    ("Macon", "GA"),
    ("Ogden", "UT"),
)
AREA_CODES: tuple[int, ...] = (
    206, 212, 303, 312, 404, 415, 503, 512, 602, 617, 702, 713, 801, 816, 919, 971,
)  # fmt: skip

FLAT_PLANS: tuple[str, ...] = ("Plan: Basic", "Plan: Plus", "Plan: Pro", "Plan: Lite")
SEAT_PLAN = "Plan: Team seats"
ADDONS: tuple[str, ...] = (
    "Add-on: Support",
    "Add-on: Storage",
    "Add-on: SSO",
    "Add-on: Audit log",
    "Add-on: API",
    "Add-on: Sandbox",
    "Add-on: Export",
    "Add-on: Backup",
)
DISCOUNTS: tuple[str, ...] = ("Discount: Annual prepay", "Discount: Loyalty", "Discount: Nonprofit")
USAGE_LINE = "Usage: API calls (per 1k)"

REJECTED_CASE_NOTE = "Dispute rejected: charge is valid"
REFUND_NOTE = "Refund duplicate payment"


def month_name(period: str) -> str:
    """``"2026-08"`` -> ``"August 2026"``."""
    year, month = period.split("-")
    return f"{MONTH_NAMES[int(month) - 1]} {year}"


def shift_period(period: str, months: int) -> str:
    year, month = (int(x) for x in period.split("-"))
    idx = year * 12 + (month - 1) + months
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def dollars(cents: int) -> str:
    """SPEC §4: ``f"${rate // 100}.{rate % 100:02d}"``, e.g. ``$49.99``."""
    return f"${cents // 100}.{cents % 100:02d}"


def fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M")


def line_sum(lines: Iterable[LineItem]) -> int:
    return sum(li.total for li in lines)


def derive_status(amount_due: int) -> InvoiceStatus:
    """S1 for a non-void invoice (mirrors ``sim.ledger.derive_invoice_status``)."""
    return InvoiceStatus.OPEN if amount_due > 0 else InvoiceStatus.PAID


# ---------------------------------------------------------------------------------------------
# Draft: id/name allocation and record factories
# ---------------------------------------------------------------------------------------------


@dataclass
class Unit:
    """A distractor unit: a named group of records added or dropped together."""

    name: str
    records: list[Record]
    required: bool


class Draft:
    """Allocates unique ids, names, emails and phones and builds well-formed records.

    ``existing`` registers an existing state's records so new records never collide with them
    (used by ``broken.py``).
    """

    def __init__(self, rng: random.Random, existing: State | None = None) -> None:
        self.rng = rng
        self.ids: set[str] = set()
        self.full_names: set[str] = set()
        self.first_names: set[str] = set()
        self.emails: set[str] = set()
        self.phones: set[str] = set()
        self.addresses: set[str] = set()
        # customer id -> periods already used by that customer's invoices
        self.periods: dict[str, set[str]] = {}
        if existing is not None:
            for c in existing.customers:
                self._register_customer(c)
            for rec in (*existing.invoices, *existing.payments, *existing.cases):
                self.ids.add(rec.id)
            for inv in existing.invoices:
                self.periods.setdefault(inv.customer_id, set()).add(inv.period)

    # -- ids --------------------------------------------------------------------------------

    def _new_id(self, prefix: str, digits: int) -> str:
        lo, hi = 10 ** (digits - 1), 10**digits - 1
        while True:
            rid = f"{prefix}{self.rng.randint(lo, hi)}"
            if rid not in self.ids:
                self.ids.add(rid)
                return rid

    def customer_id(self) -> str:
        return self._new_id("C", 4)

    def invoice_id(self) -> str:
        return self._new_id("INV-", 5)

    def payment_id(self) -> str:
        return self._new_id("PAY-", 5)

    def case_id(self) -> str:
        return self._new_id("CS-", 4)

    # -- contact details --------------------------------------------------------------------

    def first_name(self) -> str:
        """A first name no customer in the draft has yet."""
        free = [n for n in FIRST_NAMES if n not in self.first_names]
        return self.rng.choice(free)

    def last_name(self, first: str) -> str:
        free = [n for n in LAST_NAMES if f"{first} {n}" not in self.full_names]
        return self.rng.choice(free)

    def email(self, local: str, domain: str | None = None) -> str:
        """A fresh email for ``local``; tries the given or a random domain, then suffixes."""
        domains = [domain] if domain else list(self.rng.sample(DOMAINS, len(DOMAINS)))
        for n in range(100):
            loc = local if n == 0 else f"{local}{n + 1}"
            for d in domains:
                e = f"{loc}@{d}"
                if e not in self.emails:
                    self.emails.add(e)
                    return e
        raise RuntimeError("email space exhausted")  # pragma: no cover

    def phone(self) -> str:
        while True:
            p = f"+1-{self.rng.choice(AREA_CODES)}-555-{self.rng.randint(0, 9999):04d}"
            if p not in self.phones:
                self.phones.add(p)
                return p

    def address(self) -> str:
        while True:
            city, st = self.rng.choice(CITIES)
            a = f"{self.rng.randint(10, 999)} {self.rng.choice(STREETS)} {self.rng.choice(STREET_SUFFIXES)}, {city} {st}"
            if a not in self.addresses:
                self.addresses.add(a)
                return a

    def flat_rate(self) -> int:
        """Shared range: ``100·j − 1``, j ∈ [20, 300]."""
        return 100 * self.rng.randint(20, 300) - 1

    def _register_customer(self, c: Customer) -> None:
        self.ids.add(c.id)
        self.full_names.add(c.name)
        self.first_names.add(c.name.split(" ")[0])
        self.emails.add(c.email)
        self.phones.add(c.phone)
        self.addresses.add(c.address)

    def customer(
        self,
        *,
        first: str | None = None,
        last: str | None = None,
        email: str | None = None,
        basis: ContractBasis = ContractBasis.FLAT,
        rate: int | None = None,
    ) -> Customer:
        """A new customer. Distinct first name and full name unless ``first``/``last`` given."""
        first = first or self.first_name()
        last = last or self.last_name(first)
        c = Customer(
            id=self.customer_id(),
            name=f"{first} {last}",
            email=email or self.email(f"{first}.{last}".lower()),
            phone=self.phone(),
            address=self.address(),
            contract_basis=basis,
            contract_rate=self.flat_rate() if rate is None else rate,
            credit_balance=0,
        )
        self._register_customer(c)
        return c

    # -- billing records --------------------------------------------------------------------

    def free_period(self, customer_id: str, choices: Iterable[str] = PERIODS) -> str | None:
        used = self.periods.get(customer_id, set())
        free = [p for p in choices if p not in used]
        return self.rng.choice(free) if free else None

    def plan_line(self, c: Customer, *, seats: int | None = None, unit: int | None = None):
        """The customer's plan line, priced at ``contract_rate`` unless ``unit`` is given."""
        price = c.contract_rate if unit is None else unit
        if c.contract_basis is ContractBasis.PER_SEAT:
            return LineItem(SEAT_PLAN, seats or self.rng.randint(3, 40), price)
        plan = FLAT_PLANS[int(c.id[1:]) % len(FLAT_PLANS)]  # stable per customer
        return LineItem(plan, 1, price)

    def addons(self, n: int, *, qty: tuple[int, int] = (1, 1), exclude: Iterable[str] = ()):
        names = [a for a in ADDONS if a not in set(exclude)]
        return [
            LineItem(name, self.rng.randint(*qty), self.rng.randint(500, 5000))
            for name in self.rng.sample(names, n)
        ]

    def background_lines(self, c: Customer) -> list[LineItem]:
        """Shared range: 1–3 lines, the plan first, add-ons qty 1."""
        return [self.plan_line(c), *self.addons(self.rng.randint(0, 2))]

    def invoice(
        self,
        c: Customer,
        period: str,
        lines: list[LineItem],
        *,
        billed: int | None = None,
        paid: int = 0,
    ) -> Invoice:
        """An invoice with ``amount_due = billed − paid`` (paid = Σ succeeded payments)."""
        billed = line_sum(lines) if billed is None else billed
        due = billed - paid
        self.periods.setdefault(c.id, set()).add(period)
        return Invoice(
            id=self.invoice_id(),
            customer_id=c.id,
            period=period,
            line_items=tuple(lines),
            amount_billed=billed,
            amount_due=due,
            status=derive_status(due),
        )

    def pay_time(self, period: str, *, latest_hour: int = 20) -> datetime:
        """A timestamp within the month after ``period`` (day 2–26, so ±1 day stays inside)."""
        year, month = (int(x) for x in shift_period(period, 1).split("-"))
        return datetime(
            year,
            month,
            self.rng.randint(2, 26),
            self.rng.randint(8, latest_hour),
            self.rng.randint(0, 59),
        )

    def payment(
        self,
        inv: Invoice,
        amount: int,
        ts: datetime,
        status: PaymentStatus = PaymentStatus.SUCCEEDED,
        note: str = "",
    ) -> Payment:
        return Payment(
            id=self.payment_id(),
            invoice_id=inv.id,
            customer_id=inv.customer_id,
            amount=amount,
            method=self.rng.choice(("card", "ach")),
            created_at=fmt_ts(ts),
            status=status,
            note=note,
        )

    def case(
        self,
        inv: Invoice,
        summary: str,
        disputed: int,
        *,
        resolved: bool = False,
    ) -> Case:
        """An open case, or a background resolved one (resolution 0, rejection note)."""
        return Case(
            id=self.case_id(),
            customer_id=inv.customer_id,
            invoice_id=inv.id,
            summary=summary,
            disputed_amount=disputed,
            status=CaseStatus.RESOLVED if resolved else CaseStatus.OPEN,
            resolution=0 if resolved else None,
            note=REJECTED_CASE_NOTE if resolved else "",
        )

    # -- common composite records -----------------------------------------------------------

    def paid_once(self, c: Customer, period: str, lines: list[LineItem]) -> list[Record]:
        """An invoice paid in full by one succeeded payment (2 records)."""
        billed = line_sum(lines)
        inv = self.invoice(c, period, lines, paid=billed)
        return [inv, self.payment(inv, billed, self.pay_time(period))]

    def background_invoice(self, c: Customer, period: str, *, allow_paid: bool, max_records: int):
        """A correctly priced invoice of ``c``: open (1 rec) or paid once (2 rec)."""
        lines = self.background_lines(c)
        if allow_paid and max_records >= 2 and self.rng.random() < 0.5:
            return self.paid_once(c, period, lines)
        return [self.invoice(c, period, lines)]

    def dispute_summary(self, line: LineItem) -> str:
        return f"Customer disputes the '{line.description}' charge."


# ---------------------------------------------------------------------------------------------
# Distractor units
# ---------------------------------------------------------------------------------------------

# A unit factory gets the maximum record count it may use and returns the records, or None
# if it cannot build a unit within that size (or at all, e.g. periods exhausted).
UnitFactory = Callable[[int], list[Record] | None]


@dataclass
class UnitSpec:
    name: str
    min_records: int
    make: UnitFactory
    hard_only: bool = False


def z_unit(d: Draft) -> UnitSpec:
    """``z``: an unrelated customer with no other records (1 rec)."""
    return UnitSpec("z", 1, lambda _m: [d.customer()])


def other_customer_unit(d: Draft, name: str, n: int, build: Callable[[Customer], list[Record]]):
    """A unit that is a new customer plus ``n − 1`` records built by ``build``."""

    def make(_max: int) -> list[Record]:
        c = d.customer()
        return [c, *build(c)]

    return UnitSpec(name, n, make)


def add_units(
    d: Draft,
    k: int,
    required: list[UnitSpec],
    allowed: list[UnitSpec],
    *,
    hard: bool,
) -> list[Unit]:
    """SPEC §4: required units first, then random allowed units that fit, then ``z`` to k.

    Most literal reading of "random allowed units that fit": draw uniformly among the
    allowed units (``z`` excluded; hard-only units only for ``hard``) whose minimum size fits
    the remaining count, until none fits; then pad with ``z``. A factory that cannot build its
    unit (returns None) is dropped from the candidates.
    """
    units: list[Unit] = []
    remaining = k
    for spec in required:
        # Required units use their smallest form, which leaves the most room in the prompt
        # budget for the random units (e.g. F1 ``a`` is then an open invoice, not a paid one).
        recs = spec.make(spec.min_records)
        if not recs:
            raise RuntimeError(f"required unit {spec.name!r} could not be built")
        units.append(Unit(spec.name, recs, required=True))
        remaining -= len(recs)
    candidates = [s for s in allowed if s.name != "z" and (hard or not s.hard_only)]
    while remaining > 0:
        fits = [s for s in candidates if s.min_records <= remaining]
        if not fits:
            break
        spec = d.rng.choice(fits)
        recs = spec.make(remaining)
        if not recs:
            candidates.remove(spec)
            continue
        units.append(Unit(spec.name, recs, required=False))
        remaining -= len(recs)
    z = z_unit(d)
    while remaining > 0:
        units.append(Unit("z", z.make(1) or [], required=False))
        remaining -= 1
    return units


def draw_k(rng: random.Random, variant: str) -> int:
    """Distractor record count: base {2, 3, 4}, hard {5, 6}."""
    return rng.choice((5, 6)) if variant == "hard" else rng.choice((2, 3, 4))


# ---------------------------------------------------------------------------------------------
# Built template -> TaskInstance
# ---------------------------------------------------------------------------------------------


@dataclass
class Built:
    """What a template generator returns, before trimming, shuffling and the ledger."""

    family: str
    template: str
    variant: str
    request: str
    params: dict[str, Any]
    oracle_call: ToolCall
    core: list[Record]
    units: list[Unit] = field(default_factory=list)


def split_records(records: Iterable[Record]) -> State:
    """Group records by type into a ledger-less State (order kept)."""
    recs = list(records)
    return State(
        customers=tuple(r for r in recs if isinstance(r, Customer)),
        invoices=tuple(r for r in recs if isinstance(r, Invoice)),
        payments=tuple(r for r in recs if isinstance(r, Payment)),
        cases=tuple(r for r in recs if isinstance(r, Case)),
    )


def with_ledger(state: State) -> State:
    """Rebuild ``state.ledger`` from its customers, invoices and payments (§2)."""
    from envlab.sim.ledger import build_ledger

    return dataclasses.replace(
        state, ledger=build_ledger(state.customers, state.invoices, state.payments)
    )


def oracle_final(state: State, call: ToolCall) -> State:
    """``apply_tool(state, call)[0]``; raises if the oracle call fails (§4)."""
    from envlab.sim.tools import apply_tool

    final, result = apply_tool(state, call)
    if not result.ok:
        raise ValueError(f"oracle call failed: {result.error} {call}")
    return final


class BudgetError(RuntimeError):
    """Prompt estimate stays above budget even with the minimum distractor count (§10)."""


def finalize(b: Built, rng: random.Random, seed: int, split: str) -> TaskInstance:
    """Trim to the prompt budget, shuffle, build the ledger and the oracle final state."""
    from envlab.prompts import ESTIMATE_BUDGET, estimate_tokens

    units = list(b.units)
    floor = max(2, sum(len(u.records) for u in units if u.required))

    def records() -> list[Record]:
        return [*b.core, *(r for u in units for r in u.records)]

    # §10: drop the last-added non-required units until the estimate fits.
    while estimate_tokens(b.request, split_records(records())) > ESTIMATE_BUDGET:
        idx = max((i for i, u in enumerate(units) if not u.required), default=None)
        n_now = sum(len(u.records) for u in units)
        if idx is None or n_now - len(units[idx].records) < floor:
            raise BudgetError(f"{b.template}/{b.variant} seed={seed}: over prompt budget")
        del units[idx]

    grouped = split_records(records())
    shuffled = []
    for group in (grouped.customers, grouped.invoices, grouped.payments, grouped.cases):
        lst = list(group)
        rng.shuffle(lst)
        shuffled.append(tuple(lst))
    initial = with_ledger(State(*shuffled))
    return TaskInstance(
        id=f"{b.template}-{split}-seed{seed}",
        family=b.family,
        template=b.template,
        variant=b.variant,
        split=split,
        seed=seed,
        request=b.request,
        initial_state=initial,
        oracle_call=b.oracle_call,
        oracle_final_state=oracle_final(initial, b.oracle_call),
        params=b.params,
        broken_type=None,
        meta={
            "n_distractors": sum(len(u.records) for u in units),
            # Ids of the core records (§4 "Core"), in generation order.
            "target_ids": [r.id for r in b.core],
            "spec_version": SPEC_VERSION,
        },
    )
