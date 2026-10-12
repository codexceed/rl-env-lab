"""Per-template generation (SPEC §4, §5, §10, §13 "tasks"). Needs the simulator."""

from __future__ import annotations

import pytest

pytest.importorskip("envlab.sim.ledger")
pytest.importorskip("envlab.sim.tools")
pytest.importorskip("envlab.sim.state_hash")

from envlab.prompts import ESTIMATE_BUDGET, estimate_prompt_tokens
from envlab.sim.entities import CaseStatus, InvoiceStatus, PaymentStatus, State
from envlab.sim.ledger import check_invariants
from envlab.sim.state_hash import state_hash
from envlab.sim.tools import apply_tool
from envlab.tasks.common import dollars, line_sum
from envlab.tasks.generate import HELDOUT_TEMPLATES, TEMPLATES, TRAIN_TEMPLATES, derive_seed, generate
from envlab.tasks.instance import TaskInstance, dumps_line

N_SEEDS = 25
CASES = [(t, "base") for t in TEMPLATES] + [(t, "hard") for t in TRAIN_TEMPLATES]

PARAM_KEYS = {
    "F1T1": {"customer_id", "invoice_id"},
    "F1T2": {"customer_name", "period"},
    "F1T3": {"customer_id", "invoice_id"},
    "F1T4": {"customer_id", "invoice_id"},
    "F1T5": {"customer_id", "invoice_id"},
    "F2T1": {"customer_id", "invoice_id"},
    "F2T2": {"customer_name"},
    "F2T3": {"customer_id", "invoice_id"},
    "F3T1": {"case_id", "rule", "percent"},
    "F3T2": {"case_id", "rule", "percent"},
    "F3T3": {"case_id", "rule", "used_units", "line_description"},
    "F4T1": {"invoice_id"},
    "F4T2": {"customer_name"},
    "F4T3": {"invoice_id"},
    "F4T4": {"invoice_id"},
    "F4T5": {"invoice_id"},
    "F5T1": {"customer_id", "field", "new_value"},
    "F5T2": {"lookup", "field", "new_value"},
    "F5T3": {"lookup", "field", "new_value"},
}
ORACLE_TOOL = {
    "F1": "apply_credit",
    "F2": "refund_payment",
    "F3": "resolve_case",
    "F4": "adjust_invoice",
    "F5": "update_customer",
}


def _instances(template: str, variant: str) -> list[TaskInstance]:
    return [generate(template, derive_seed(77, template, variant, i), variant=variant) for i in range(N_SEEDS)]


def _n_records(s: State) -> int:
    return len(s.customers) + len(s.invoices) + len(s.payments) + len(s.cases)


@pytest.fixture(scope="module", params=CASES, ids=[f"{t}-{v}" for t, v in CASES])
def batch(request) -> tuple[str, str, list[TaskInstance]]:
    template, variant = request.param
    return template, variant, _instances(template, variant)


def test_same_seed_is_byte_identical(batch):
    template, variant, insts = batch
    again = _instances(template, variant)
    assert [dumps_line(i) for i in insts] == [dumps_line(i) for i in again]
    assert len({dumps_line(i) for i in insts}) == len(insts)  # different seeds differ


def test_fields_and_params(batch):
    template, variant, insts = batch
    for inst in insts:
        assert inst.template == template
        assert inst.family == template[:2]
        assert inst.variant == variant
        assert inst.split == (
            "train_hard" if variant == "hard" else "heldout" if template in HELDOUT_TEMPLATES else "train"
        )
        keys = PARAM_KEYS[template]
        if template == "F3T1" and variant == "hard":  # §5: invoice_id instead of case_id
            keys = {"invoice_id", "rule", "percent"}
        assert set(inst.params) == keys
        assert inst.oracle_call.name == ORACLE_TOOL[inst.family]
        assert inst.broken_type is None
        assert inst.meta["spec_version"] == "1.0"


def test_oracle_succeeds_and_invariants_hold(batch):
    _, _, insts = batch
    for inst in insts:
        assert check_invariants(inst.initial_state) == []
        final, result = apply_tool(inst.initial_state, inst.oracle_call)
        assert result.ok, (inst.id, result)
        assert state_hash(final) == state_hash(inst.oracle_final_state)
        assert check_invariants(inst.oracle_final_state) == []


def test_distractor_counts(batch):
    _, variant, insts = batch
    lo, hi = (2, 6) if variant == "hard" else (2, 4)
    for inst in insts:
        n = inst.meta["n_distractors"]
        assert lo <= n <= hi, (inst.id, n)
        assert n == _n_records(inst.initial_state) - len(inst.meta["target_ids"])


def test_prompt_budget(batch):
    _, _, insts = batch
    for inst in insts:
        assert estimate_prompt_tokens(inst) <= ESTIMATE_BUDGET, inst.id


def test_no_hard_variant_for_heldout():
    for template in HELDOUT_TEMPLATES:
        with pytest.raises(ValueError):
            generate(template, 1, variant="hard")


# --- required units (§4/§5 tables), checked structurally ---------------------------------------


def _core(inst: TaskInstance):
    s = inst.initial_state
    cust = s.customer(inst.meta["target_ids"][0])
    assert cust is not None
    target = s.invoice(inst.meta["target_ids"][1]) if len(inst.meta["target_ids"]) > 1 else None
    return s, cust, target


def _other_invoices(s, cust, target):
    return [i for i in s.invoices if i.customer_id == cust.id and (target is None or i.id != target.id)]


def _required_present(inst: TaskInstance) -> bool:
    s, c, t = _core(inst)
    hard = inst.variant == "hard"
    others = _other_invoices(s, c, t)
    first = c.name.split(" ")[0]
    same_first = [o for o in s.customers if o.id != c.id and o.name.split(" ")[0] == first and o.name != c.name]

    def succeeded(inv):
        return [p for p in s.payments if p.invoice_id == inv.id and p.status is PaymentStatus.SUCCEEDED]

    match inst.template, hard:
        case "F1T2", False:
            assert t is not None
            return any(o.period != t.period for o in others)
        case "F1T1" | "F1T4" | "F1T5", True:
            same_rate = [
                o
                for o in s.customers
                if o.id != c.id
                and o.contract_rate == c.contract_rate
                and any(
                    i.customer_id == o.id
                    and i.status is InvoiceStatus.OPEN
                    and i.line_items[0].unit_price > o.contract_rate
                    for i in s.invoices
                )
            ]
            return bool(others) and bool(same_rate)
        case "F1T2", True:
            assert t is not None
            y, m = (int(x) for x in t.period.split("-"))
            want = {f"{y}-{m - 1:02d}", f"{y}-{m + 1:02d}"}
            return want <= {o.period for o in others} and bool(same_first)
        case "F2T2", _:
            assert t is not None
            b_ok = any(len(succeeded(o)) == 1 and succeeded(o)[0].amount == t.amount_billed for o in others)
            if not hard:
                return b_ok
            e_ok = any(len({p.amount for p in succeeded(o)}) == 2 and len(succeeded(o)) == 2 for o in others)
            return b_ok and e_ok
        case "F2T3", _:
            assert t is not None
            dup = s.payment(str(inst.oracle_call.arguments["payment_id"]))
            assert dup is not None
            first_ok = min(p.created_at for p in succeeded(t))
            return any(
                p.status is PaymentStatus.FAILED and p.amount == dup.amount and p.created_at < first_ok
                for p in s.payments
                if p.invoice_id == t.id
            )
        case "F2T1", True:
            assert t is not None
            d_ok = any(
                any(p.invoice_id == o.id and p.status is PaymentStatus.REFUNDED for p in s.payments) for o in others
            )
            a_ok = any(p.invoice_id == t.id and p.status is PaymentStatus.FAILED for p in s.payments)
            return d_ok and a_ok
        case "F3T1" | "F3T2", True:
            assert t is not None
            d_ok = any(k.customer_id == c.id and k.invoice_id != t.id and k.status is CaseStatus.OPEN for k in s.cases)
            e_ok = any(k.invoice_id == t.id and k.status is CaseStatus.RESOLVED for k in s.cases)
            return d_ok and (e_ok or inst.template == "F3T2")
        case "F4T1" | "F4T2" | "F4T4" | "F4T5", True:
            assert t is not None
            descs = [li.description for li in t.line_items]
            c_ok = any(
                [li.description for li in o.line_items] == descs
                and o.status is InvoiceStatus.OPEN
                and o.amount_billed == line_sum(o.line_items)
                for o in others
            )
            a_ok = sum(o.status is InvoiceStatus.OPEN and o.amount_billed == line_sum(o.line_items) for o in others)
            return c_ok and (inst.template != "F4T2" or a_ok >= 2)
        case "F4T2", False:
            return any(o.status is InvoiceStatus.OPEN and o.amount_billed == line_sum(o.line_items) for o in others)
        case "F5T1", True:
            local, dom = c.email.split("@")
            a_ok = any(
                o.id != c.id and o.email.split("@")[0] == local and not o.email.endswith(dom) for o in s.customers
            )
            return a_ok and bool(same_first)
        case "F5T2", _:
            c_ok = any(o.id != c.id and o.name == c.name and o.email != c.email for o in s.customers)
            if not hard:
                return c_ok
            local, dom = c.email.split("@")
            return c_ok and any(
                o.id != c.id and o.email.split("@")[0] == local and not o.email.endswith(dom) for o in s.customers
            )
        case "F5T3", _:
            return bool(same_first)
    return True


def test_required_units_present(batch):
    _, _, insts = batch
    for inst in insts:
        assert _required_present(inst), inst.id


# --- medium-difficulty candidates (SPEC §4: F1T4, F1T5, F4T4, F4T5; §5 hard rows) ---------------

CANDIDATE_CASES = [(t, v) for t in ("F1T4", "F1T5", "F4T4", "F4T5") if t in TRAIN_TEMPLATES for v in ("base", "hard")]


def _f1_candidate_request(inst: TaskInstance) -> str:
    _, c, t = _core(inst)
    assert t is not None
    if inst.template == "F1T4":
        return (
            f"{c.name} ({c.id}) says the plan price on invoice {t.id} is above their contract rate. "
            "Credit the difference."
        )
    return (
        f"{c.name} ({c.id}) was promised {dollars(c.contract_rate)} per month, but invoice {t.id} "
        f"charged {dollars(t.line_items[0].unit_price)} for the plan. Credit the difference."
    )


def _f4_candidate_request(inst: TaskInstance) -> str:
    _, c, t = _core(inst)
    assert t is not None
    if inst.template == "F4T4":
        return (
            f"The total on {c.name}'s invoice {t.id} doesn't equal the sum of its line items. "
            "Correct the invoice total."
        )
    return f"Invoice {t.id} for {c.name} billed one of its line items twice. Correct the invoice total."


@pytest.mark.parametrize(("template", "variant"), CANDIDATE_CASES, ids=[f"{t}-{v}" for t, v in CANDIDATE_CASES])
def test_candidate_shapes_and_requests(template: str, variant: str):
    hard = variant == "hard"
    for inst in _instances(template, variant):
        _, c, t = _core(inst)
        assert t is not None
        lines = t.line_items
        assert lines[0].description.startswith("Plan: ") and lines[0].quantity == 1
        if template in ("F1T4", "F1T5"):
            amount = inst.oracle_call.arguments["amount"]
            assert amount == lines[0].unit_price - c.contract_rate
            assert (2 <= len(lines) <= 3) if hard else len(lines) == 1
            if template == "F1T5" and hard:
                assert isinstance(amount, int) and 100 <= amount <= 5000
            else:  # whole-dollar overcharge
                assert isinstance(amount, int) and amount % 100 == 0 and 100 <= amount <= 2000
            assert inst.request == _f1_candidate_request(inst)
            assert inst.params == {"customer_id": c.id, "invoice_id": t.id}
        else:
            assert (3 <= len(lines) <= 4) if hard else len(lines) == 2
            assert lines[0].unit_price == c.contract_rate
            for li in lines[1:]:
                assert li.unit_price % 100 == 0 and 500 <= li.unit_price <= 5000
                assert 1 <= li.quantity <= (3 if hard else 1)
            total = line_sum(lines)
            assert inst.oracle_call.arguments["new_amount"] == total != t.amount_billed
            if template == "F4T5":
                assert c.contract_rate % 100 == 0  # every line is whole dollars
                assert t.amount_billed - total in {li.total for li in lines}  # one line counted twice
            assert inst.request == _f4_candidate_request(inst)
            assert inst.params == {"invoice_id": t.id}
