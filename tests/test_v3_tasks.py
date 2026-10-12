"""v3 generators (SPEC_v3.md §4, §9, §10): known answers, determinism, params, invariants, budget."""

from __future__ import annotations

import random
import re

import pytest

from envlab.prompts import PROMPT_TOKEN_LIMIT, V3_ESTIMATE_BUDGET, V3_PROMPT_TOKEN_LIMIT, estimate_prompt_tokens
from envlab.sim.entities import CaseStatus, InvoiceStatus, PaymentStatus, State
from envlab.sim.ledger import check_invariants
from envlab.sim.state_hash import state_hash
from envlab.sim.tools import PHONE_RE, apply_tool
from envlab.tasks.common import Draft
from envlab.tasks.generate import derive_seed
from envlab.tasks.instance import TaskInstance, dumps_line
from envlab.tasks.v3 import (
    FAR_TEMPLATES,
    HELDOUT_TEMPLATES,
    NEAR_TEMPLATES,
    SPEC_VERSION,
    TEMPLATES,
    TRAIN_TEMPLATES,
    f1,
    f3,
    f4,
    f5,
)
from envlab.tasks.v3 import common as v3common
from envlab.tasks.v3.common import K_CHOICES, MIN_DISTRACTORS
from envlab.tasks.v3.generate import default_split, generate

N_SEEDS = 30

PARAM_KEYS: dict[str, set[str]] = {
    "F1A": {"customer_name", "period", "rule"},
    "F1B": {"customer_id", "period", "rule", "percent"},
    "F1C": {"customer_id", "invoice_id", "rule", "downgrade_date"},
    "F1N": {"customer_id", "period", "rule", "percent"},
    "F1X": {"customer_id", "invoice_id", "rule", "tier_seats", "tier1_price", "tier2_price"},
    "F2A": {"customer_id", "rule"},
    "F2B": {"customer_name", "period", "rule"},
    "F2C": {"lookup", "rule"},
    "F2N": {"lookup", "rule"},
    "F2X": {"customer_id", "invoice_id", "rule"},
    "F3A": {"invoice_id", "rule", "percent", "cap"},
    "F3B": {"customer_name", "period", "rule", "used_units", "line_description"},
    "F3C": {"case_id", "rule", "hours"},
    "F3N": {"case_id", "rule", "used_units", "line_description", "percent"},
    "F3X": {"case_id", "rule", "used_units", "allowance_units", "line_description"},
    "F4A": {"customer_name", "period", "rule"},
    "F4B": {"invoice_id", "rule", "tax_bp"},
    "F4C": {"customer_name", "rule"},
    "F4N": {"invoice_id", "rule", "tax_bp"},
    "F4X": {"invoice_id", "rule", "minimum"},
    "F5A": {"lookup", "field", "new_value"},
    "F5B": {"lookup", "field", "new_value"},
    "F5C": {"lookup", "field", "new_value"},
    "F5N": {"lookup", "field", "new_value"},
    "F5X": {"lookup", "field", "new_value"},
}
RULES: dict[str, str] = {
    "F1A": "per_seat",
    "F1B": "percent_off",
    "F1C": "prorate_days",
    "F1N": "per_seat_percent",
    "F1X": "tiered",
    "F2A": "duplicate",
    "F2B": "duplicate",
    "F2C": "duplicate",
    "F2N": "duplicate",
    "F2X": "overpayment",
    "F3A": "percent_cap",
    "F3B": "unused_units",
    "F3C": "outage_hours",
    "F3N": "unused_units_percent",
    "F3X": "allowance",
    "F4A": "sum",
    "F4B": "tax",
    "F4C": "sum",
    "F4N": "tax",
    "F4X": "minimum",
}
ORACLE_TOOL = {"F1": "apply_credit", "F2": "refund_payment", "F3": "resolve_case", "F4": "adjust_invoice"}
ORACLE_TOOL["F5"] = "update_customer"


def _instances(template: str) -> list[TaskInstance]:
    return [generate(template, derive_seed(99, template, i)) for i in range(N_SEEDS)]


@pytest.fixture(scope="module", params=TEMPLATES)
def batch(request) -> tuple[str, list[TaskInstance]]:
    return request.param, _instances(request.param)


# --- registry -----------------------------------------------------------------------------------


def test_registry_covers_every_template_once():
    assert len(TEMPLATES) == len(set(TEMPLATES)) == 25
    assert set(PARAM_KEYS) == set(TEMPLATES)
    assert [t[:2] for t in NEAR_TEMPLATES] == [t[:2] for t in FAR_TEMPLATES] == ["F1", "F2", "F3", "F4", "F5"]
    assert all(default_split(t) == "train" for t in TRAIN_TEMPLATES)
    assert all(default_split(t) == "heldout" for t in HELDOUT_TEMPLATES)
    assert SPEC_VERSION == "v3.0"


def test_near_templates_only_use_trained_rules():
    """A near template recombines trained steps: its rule (or both halves of it) appears in a train template."""
    train_rules = {RULES[t] for t in TRAIN_TEMPLATES if t in RULES}
    assert RULES["F2N"] in train_rules and RULES["F4N"] in train_rules
    assert {"per_seat", "percent_off"} <= train_rules  # F1N = per-seat + percent off
    assert {"unused_units", "percent_cap"} <= train_rules  # F3N = unused units, then a percent
    far_rules = {RULES[t] for t in FAR_TEMPLATES if t in RULES}
    assert not far_rules & train_rules  # far templates need a rule never trained


def test_unknown_template_is_rejected():
    with pytest.raises(ValueError):
        generate("F1T1", 1)


def test_prompt_limits_keep_v1_unchanged():
    assert PROMPT_TOKEN_LIMIT == 1536
    assert V3_PROMPT_TOKEN_LIMIT == 2048 and V3_ESTIMATE_BUDGET < V3_PROMPT_TOKEN_LIMIT


# --- known answers: oracle rules on hand-checked numbers ------------------------------------------


def test_f1_rules_known_answers():
    assert f1.per_seat_credit(37, 5250, 4999) == 9287  # 37 x 251
    assert f1.percent_off_credit(1, 12999, 15) == 1949  # 1949.85 rounded down
    assert f1.percent_off_credit(25, 4999, 22) == 27494  # 124975 x 22 / 100 = 27494.5
    assert f1.prorate_credit(14199, 2699, 24, 30) == 9200  # 11500 x 24 / 30
    assert f1.prorate_credit(10000, 2999, 13, 31) == 2935  # 91013 / 31 = 2935.9
    # 23 seats at 18.99 billed; contract 17.49 less 12%: 40227 - 4827 = 35400; 43677 - 35400
    assert f1.per_seat_percent_credit(23, 1899, 1749, 12) == 8277
    assert f1.tiered_credit(57, 4000, 25, 4000, 3200) == 25600  # 32 seats x 8.00
    assert f1.tiered_credit(20, 4000, 25, 4000, 3200) == 0  # all seats inside the first tier


def test_f3_rules_known_answers():
    assert f3.percent_cap_adjustment(12345, 35, 5000) == 4320  # 4320.75 rounded down, under the cap
    assert f3.percent_cap_adjustment(12345, 35, 4000) == 4000  # capped
    assert f3.unused_adjustment(1500, 1123, 37) == 13949  # 377 x 37
    assert f3.outage_adjustment(14999, 7, 31) == 141  # 104993 / 744 = 141.1
    assert f3.unused_percent_adjustment(1500, 1123, 37, 45) == 6277  # 13949 x 45 / 100 = 6277.05
    assert f3.allowance_adjustment(2000, 1800, 500, 12) == 8400  # billable 1300: (2000 - 1300) x 12


def test_f4_rules_known_answers():
    assert f4.with_tax(54321, 825) == 58802  # tax 4481.48 rounded down
    assert f4.with_tax(10000, 600) == 10600
    assert f4.with_minimum(18000, 25000) == 25000
    assert f4.with_minimum(31000, 25000) == 31000


def test_text_helpers():
    assert v3common.percent_text(825) == "8.25%"
    assert v3common.percent_text(650) == "6.5%"
    assert v3common.percent_text(600) == "6%"
    assert v3common.long_date("2026-08", 19) == "August 19, 2026"
    assert v3common.days_in_month("2026-04") == 30 and v3common.days_in_month("2026-08") == 31
    rng = random.Random(3)
    for _ in range(200):
        shown = f5.format_phone(rng, "+1-415-555-0123")
        assert shown != "+1-415-555-0123"
        assert re.sub(r"\D", "", shown).endswith("4155550123")


def test_split_amount_and_typo():
    rng = random.Random(5)
    for total in (1999, 5000, 123457):
        for parts in (2, 3, 4):
            amounts = v3common.split_amount(rng, total, parts)
            assert sum(amounts) == total and len(set(amounts)) == parts and min(amounts) >= total // 10
    for digits in ("724", "11", "100", "0450", "9999"):
        wrong = f5.typo(rng, digits)
        assert wrong != digits and len(wrong) == len(digits)


def test_similar_phone_is_fresh_and_alike():
    d = Draft(random.Random(1))
    phone = d.phone()
    for _ in range(20):  # at most 39 numbers are one swap or one digit away
        other = v3common.similar_phone(d, phone)
        assert other != phone and other[:-4] == phone[:-4] and PHONE_RE.fullmatch(other)


# --- per-template generation --------------------------------------------------------------------


def test_same_seed_is_byte_identical(batch):
    template, insts = batch
    assert [dumps_line(i) for i in insts] == [dumps_line(i) for i in _instances(template)]
    assert len({dumps_line(i) for i in insts}) == len(insts)


def test_fields_params_and_meta(batch):
    template, insts = batch
    for inst in insts:
        assert (inst.template, inst.family, inst.variant) == (template, template[:2], "base")
        assert inst.split == default_split(template)
        assert set(inst.params) == PARAM_KEYS[template]
        if template in RULES:
            assert inst.params["rule"] == RULES[template]
        assert inst.oracle_call.name == ORACLE_TOOL[inst.family]
        assert inst.meta["spec_version"] == SPEC_VERSION
        assert MIN_DISTRACTORS <= inst.meta["n_distractors"] <= max(K_CHOICES) + 2  # +2: a required unit's size


def test_oracle_succeeds_and_invariants_hold(batch):
    _, insts = batch
    for inst in insts:
        assert check_invariants(inst.initial_state) == []
        final, result = apply_tool(inst.initial_state, inst.oracle_call)
        assert result.ok, result
        assert state_hash(final, mask_text=True) == state_hash(inst.oracle_final_state, mask_text=True)
        assert check_invariants(inst.oracle_final_state) == []


def test_prompt_budget(batch):
    _, insts = batch
    assert all(estimate_prompt_tokens(i) <= V3_ESTIMATE_BUDGET for i in insts)


def _n(s: State) -> int:
    return len(s.customers) + len(s.invoices) + len(s.payments) + len(s.cases)


def _f1_expected(p: dict, rate: int, qty: int, unit: int, period: str) -> int:
    """F1 credit recomputed from the plan line, the contract rate and the request numbers."""
    rule = p["rule"]
    if rule == "per_seat":
        return qty * (unit - rate)
    if rule == "percent_off":
        return qty * unit * p["percent"] // 100
    if rule == "per_seat_percent":
        return qty * unit - (qty * rate - qty * rate * p["percent"] // 100)
    if rule == "tiered":
        return (qty - p["tier_seats"]) * (p["tier1_price"] - p["tier2_price"])
    days = v3common.days_in_month(period)
    return (unit - rate) * (days - int(p["downgrade_date"][-2:]) + 1) // days


def test_oracle_recomputed_from_records(batch):
    """Each oracle value, recomputed here from the shown records and the request numbers."""
    template, insts = batch
    for inst in insts:
        s, p, args = inst.initial_state, inst.params, inst.oracle_call.arguments
        if inst.family == "F1":
            inv = s.invoice(str(args["invoice_id"]))
            assert inv is not None and inv.status is InvoiceStatus.OPEN
            c = s.customer(inv.customer_id)
            assert c is not None
            plan = inv.line_items[0]
            assert plan.description.startswith("Plan: ")
            assert args["amount"] == _f1_expected(p, c.contract_rate, plan.quantity, plan.unit_price, inv.period)
        elif inst.family == "F2":
            dup = s.payment(str(args["payment_id"]))
            assert dup is not None and dup.status is PaymentStatus.SUCCEEDED
            inv = s.invoice(dup.invoice_id)
            assert inv is not None and inv.amount_due == -dup.amount  # refunding it settles the invoice
            ok = [x for x in s.payments if x.invoice_id == inv.id and x.status is PaymentStatus.SUCCEEDED]
            if p["rule"] == "duplicate":
                assert any(q.amount == dup.amount and q.created_at < dup.created_at for q in ok)
            else:
                assert len({x.amount for x in ok}) == len(ok)  # no equal-amount twin
                assert dup.created_at == max(x.created_at for x in ok)
            if template == "F2A":
                assert {x.status for x in s.payments if x.invoice_id == inv.id} == set(PaymentStatus)
        elif inst.family == "F3":
            k = s.case(str(args["case_id"]))
            assert k is not None and k.status is CaseStatus.OPEN
            case_inv = s.invoice(k.invoice_id)
            assert case_inv is not None and 1 <= int(args["adjustment"]) < case_inv.amount_due
        elif inst.family == "F4":
            inv = s.invoice(str(args["invoice_id"]))
            assert inv is not None
            total = sum(li.total for li in inv.line_items)
            expected = {
                "sum": total,
                "tax": total + total * p.get("tax_bp", 0) // 10000,
                "minimum": max(total, p.get("minimum", 0)),
            }[p["rule"]]
            assert args["new_amount"] == expected != inv.amount_billed
        else:
            c = s.customer(str(args["customer_id"]))
            assert c is not None
            assert all(getattr(c, k) == v for k, v in p["lookup"].items())
            assert args["value"] == p["new_value"] != getattr(c, p["field"])
        assert _n(s) >= 2


def test_twins_are_present_where_the_template_says(batch):
    """Name-based templates always show a different customer with the target's exact name."""
    template, insts = batch
    if template not in {"F1A", "F2B", "F2C", "F2N", "F3B", "F4A", "F4C", "F5A", "F5B", "F5C", "F5N"}:
        return
    for inst in insts:
        names = [c.name for c in inst.initial_state.customers]
        target = inst.initial_state.customer(inst.meta["target_ids"][0])  # core records start with C
        assert target is not None and names.count(target.name) >= 2, inst.request


def test_f5x_request_shows_a_non_canonical_phone(batch):
    template, insts = batch
    if template != "F5X":
        return
    for inst in insts:
        assert inst.params["new_value"] not in inst.request
        assert re.fullmatch(r"\+1-\d{3}-555-\d{4}", inst.params["new_value"])
