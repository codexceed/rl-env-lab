"""Broken variants: deliberately invalid instances the quality gates must reject (SPEC §8)."""

import dataclasses
import random

from envlab.calls import ToolCall
from envlab.sim.entities import (
    Case,
    CaseStatus,
    Customer,
    Invoice,
    InvoiceStatus,
    Payment,
    PaymentStatus,
    State,
)
from envlab.tasks import f1, f2, f4, f5
from envlab.tasks.common import PERIODS, Draft, oracle_final, with_ledger
from envlab.tasks.generate import BROKEN_SEED, derive_seed, generate
from envlab.tasks.instance import TaskInstance

BROKEN_TYPES: tuple[str, ...] = ("impossible", "ambiguous", "grader_bug", "env_bug")
PER_TEMPLATE = 5  # 5 per family per type -> 25 per type, 100 in all

# Source templates per type (§8). Train templates only, so held-out templates are never mutated.
SOURCES: dict[str, tuple[str, ...]] = {
    "impossible": ("F1T1", "F2T1", "F3T1", "F4T1", "F5T1"),
    "ambiguous": ("F1T2", "F2T2", "F3T1", "F4T2", "F5T2"),
    "grader_bug": ("F1T2", "F2T2", "F3T2", "F4T2", "F5T2"),
    "env_bug": ("F1T1", "F2T1", "F3T1", "F4T1", "F5T1"),
}


def _replace_at(records: tuple, old: object, new: object) -> tuple:
    """Replace the first record identical to ``old``, keeping its tuple position."""
    i = next(i for i, r in enumerate(records) if r is old)
    return (*records[:i], new, *records[i + 1 :])


def _require[T](value: T | None, what: str) -> T:
    if value is None:
        raise ValueError(f"base instance has no {what}")
    return value


def _target_invoice(state: State, call: ToolCall) -> Invoice:
    return _require(state.invoice(str(call.arguments["invoice_id"])), "target invoice")


def _target_payment(state: State, call: ToolCall) -> Payment:
    return _require(state.payment(str(call.arguments["payment_id"])), "target payment")


def _target_case(state: State, call: ToolCall) -> Case:
    return _require(state.case(str(call.arguments["case_id"])), "target case")


def _target_customer(state: State, call: ToolCall) -> Customer:
    return _require(state.customer(str(call.arguments["customer_id"])), "target customer")


def _impossible(base: TaskInstance) -> TaskInstance:
    """Remove the referenced record and everything referencing it; keep the stale final state."""
    s, call = base.initial_state, base.oracle_call
    if base.family in ("F1", "F4", "F2"):
        inv_id = _target_invoice(s, call).id if base.family != "F2" else _target_payment(s, call).invoice_id
        state = dataclasses.replace(
            s,
            invoices=tuple(r for r in s.invoices if r.id != inv_id),
            payments=tuple(r for r in s.payments if r.invoice_id != inv_id),
            cases=tuple(r for r in s.cases if r.invoice_id != inv_id),
        )
    elif base.family == "F3":
        case_id = _target_case(s, call).id
        state = dataclasses.replace(s, cases=tuple(r for r in s.cases if r.id != case_id))
    else:
        cid = _target_customer(s, call).id
        state = dataclasses.replace(
            s,
            customers=tuple(r for r in s.customers if r.id != cid),
            invoices=tuple(r for r in s.invoices if r.customer_id != cid),
            payments=tuple(r for r in s.payments if r.customer_id != cid),
            cases=tuple(r for r in s.cases if r.customer_id != cid),
        )
    return dataclasses.replace(base, initial_state=with_ledger(state))


def _ambiguous(base: TaskInstance, rng: random.Random) -> TaskInstance:
    """Add a second valid target under the params; recompute the oracle final state."""
    s, call = base.initial_state, base.oracle_call
    d = Draft(rng, existing=s)
    request, params = base.request, base.params
    new_inv: list[Invoice] = []
    new_pay: list[Payment] = []
    new_case: list[Case] = []
    new_cust: list[Customer] = []
    if base.family == "F1":
        t = _target_invoice(s, call)
        c = _require(s.customer(t.customer_id), "customer")
        new_inv.append(d.invoice(c, t.period, f1.overbilled_lines(d, c, rng.randint(0, 1))))
    elif base.family == "F2":
        c = _require(s.customer(_target_payment(s, call).customer_id), "customer")
        period = d.free_period(c.id) or rng.choice(PERIODS)
        inv, p1, p2 = f2.duplicate_invoice(d, c, period, d.background_lines(c))
        new_inv.append(inv)
        new_pay += [p1, p2]
    elif base.family == "F3":
        k = _target_case(s, call)
        t = _require(s.invoice(k.invoice_id), "case invoice")
        request = f"Resolve the open dispute on invoice {t.id}. Approve it in full."
        params = {"invoice_id": t.id, "rule": "percent", "percent": 100}
        others = [li for li in t.line_items if f"'{li.description}'" not in k.summary]
        line = rng.choice(others)
        new_case.append(d.case(t, d.dispute_summary(line), line.total))
    elif base.family == "F4":
        c = _require(s.customer(_target_invoice(s, call).customer_id), "customer")
        lines = f4.mis_summed_lines(d, c, "F4T2", "base")
        period = d.free_period(c.id) or rng.choice(PERIODS)
        new_inv.append(d.invoice(c, period, lines, billed=f4.inject_error(rng, lines)))
    else:
        c = _target_customer(s, call)
        twin = d.customer()
        new_cust.append(dataclasses.replace(twin, name=c.name, email=c.email))
    state = with_ledger(
        State(
            customers=(*s.customers, *new_cust),
            invoices=(*s.invoices, *new_inv),
            payments=(*s.payments, *new_pay),
            cases=(*s.cases, *new_case),
        )
    )
    return dataclasses.replace(
        base,
        request=request,
        params=params,
        initial_state=state,
        oracle_final_state=oracle_final(state, call),
    )


def perturbed_call(base: TaskInstance) -> ToolCall:
    """The grader_bug call (§8): the oracle call nudged to a plausible wrong value."""
    s, call = base.initial_state, base.oracle_call
    args = dict(call.arguments)
    if base.family == "F1":
        args["amount"] = int(args["amount"]) + 1
    elif base.family == "F2":
        dup = _target_payment(s, call)
        original = min(
            (
                p
                for p in s.payments
                if p.invoice_id == dup.invoice_id
                and p.status is PaymentStatus.SUCCEEDED
                and p.amount == dup.amount
                and p.id != dup.id
            ),
            key=lambda p: p.created_at,
        )
        args["payment_id"] = original.id
    elif base.family == "F3":
        args["adjustment"] = int(args["adjustment"]) + 1
    elif base.family == "F4":
        t = _target_invoice(s, call)
        new = int(args["new_amount"]) + 1
        args["new_amount"] = new + 1 if new == t.amount_billed else new
    else:
        c = _target_customer(s, call)
        field = str(args["field"])
        args["value"] = f5.wrong_value(field, str(args["value"]))
        assert args["value"] != getattr(c, field)
    return ToolCall(call.name, args)


def _grader_bug(base: TaskInstance) -> TaskInstance:
    """The stored oracle final state comes from a perturbed call (§8)."""
    return dataclasses.replace(base, oracle_final_state=oracle_final(base.initial_state, perturbed_call(base)))


def _env_bug(base: TaskInstance, rng: random.Random) -> TaskInstance:
    """Corrupt the target so the valid call breaks; ledger and oracle final state are kept."""
    s, call = base.initial_state, base.oracle_call
    if base.family in ("F1", "F4"):
        t = _target_invoice(s, call)
        state = dataclasses.replace(
            s, invoices=_replace_at(s.invoices, t, dataclasses.replace(t, status=InvoiceStatus.VOID))
        )
    elif base.family == "F2":
        p = _target_payment(s, call)
        state = dataclasses.replace(
            s, payments=_replace_at(s.payments, p, dataclasses.replace(p, status=PaymentStatus.FAILED))
        )
    elif base.family == "F3":
        k = _target_case(s, call)
        bad = dataclasses.replace(k, status=CaseStatus.RESOLVED, resolution=None)
        state = dataclasses.replace(s, cases=_replace_at(s.cases, k, bad))
    else:
        c = _target_customer(s, call)
        clone = dataclasses.replace(Draft(rng, existing=s).customer(), id=c.id)
        state = dataclasses.replace(s, customers=(*s.customers, clone))
    return dataclasses.replace(base, initial_state=state)


def make_broken(base: TaskInstance, kind: str, seed: int) -> TaskInstance:
    """A broken variant of a valid ``base`` instance (§8). The id is provisional."""
    if kind not in BROKEN_TYPES:
        raise ValueError(f"unknown broken type {kind!r}")
    rng = random.Random(derive_seed(seed, "mutate"))
    if kind == "impossible":
        out = _impossible(base)
    elif kind == "ambiguous":
        out = _ambiguous(base, rng)
    elif kind == "grader_bug":
        out = _grader_bug(base)
    else:
        out = _env_bug(base, rng)
    return dataclasses.replace(
        out,
        id=f"{base.template}-broken-{kind}",
        split="broken",
        variant="broken",
        broken_type=kind,
        seed=seed,
    )


def broken_id(template: str, kind: str, index: int) -> str:
    return f"{template}-broken-{kind}-{index:03d}"


def generate_broken() -> list[TaskInstance]:
    """The 100 broken instances: 4 types x 5 families x 5, ordered by type, template, index."""
    out = []
    for kind in BROKEN_TYPES:
        for template in SOURCES[kind]:
            for index in range(PER_TEMPLATE):
                seed = derive_seed(BROKEN_SEED, kind, template, index)
                inst = make_broken(generate(template, seed), kind, seed)
                out.append(dataclasses.replace(inst, id=broken_id(template, kind, index)))
    return out
