"""Gates G1-G3, solvers and wrong-value calls (SPEC §7, §8, §13)."""

from __future__ import annotations

import dataclasses

import pytest
from verify_builders import BROKEN, BROKEN_SOURCE, BUILDERS, call, case, customer, instance, invoice, line, state

from envlab.calls import ToolCall
from envlab.tasks.instance import TaskInstance
from envlab.verify.gates import accepts, gate_g1, gate_g2, gate_g3, run_gates
from envlab.verify.solvers import solve
from envlab.verify.wrong import wrong_value_call

FAMILIES = ("F1", "F2", "F3", "F4", "F5")


def _reasons(inst: TaskInstance) -> dict[str, str]:
    return {g.gate: g.reason for g in run_gates(inst)}


# --- valid instances ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", sorted(BUILDERS))
def test_valid_instance_passes_all_gates(template: str):
    inst = BUILDERS[template]()
    results = run_gates(inst)
    assert [g.gate for g in results] == ["G1", "G2", "G3"]
    assert all(g.passed for g in results), results
    assert accepts(inst)


@pytest.mark.parametrize("template", sorted(BUILDERS))
def test_solver_returns_exactly_the_oracle(template: str):
    inst = BUILDERS[template]()
    assert solve(inst) == [inst.oracle_call]


@pytest.mark.parametrize("template", sorted(BUILDERS))
def test_gates_ignore_labels(template: str):
    """Gates must not read variant, broken_type, split or meta (SPEC §7)."""
    inst = BUILDERS[template]()
    relabelled = dataclasses.replace(inst, variant="broken", split="broken", broken_type="impossible", meta={})
    assert run_gates(relabelled) == run_gates(inst)


# --- broken variants: each type is rejected by its expected gate (SPEC §8) -----------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_impossible_fails_g1b_and_g2(family: str):
    inst = BROKEN["impossible"](BROKEN_SOURCE["impossible"][family]())
    g1, g2, _ = run_gates(inst)
    assert not g1.passed and "oracle_tool_error" in g1.detail
    assert (g2.passed, g2.reason) == (False, "no_solution")
    assert not accepts(inst)


@pytest.mark.parametrize("family", FAMILIES)
def test_ambiguous_fails_g2_only(family: str):
    inst = BROKEN["ambiguous"](BROKEN_SOURCE["ambiguous"][family]())
    g1, g2, g3 = run_gates(inst)
    assert g1.passed and g3.passed
    assert (g2.passed, g2.reason) == (False, "multiple_solutions")
    assert not accepts(inst)


@pytest.mark.parametrize("family", FAMILIES)
def test_grader_bug_fails_g1c(family: str):
    inst = BROKEN["grader_bug"](BROKEN_SOURCE["grader_bug"][family]())
    g1, g2, _ = run_gates(inst)
    assert (g1.passed, g1.reason) == (False, "oracle_state_mismatch")
    assert "oracle_not_robust" in g1.detail
    assert g2.passed
    assert not accepts(inst)


@pytest.mark.parametrize("family", FAMILIES)
def test_env_bug_fails_g1a(family: str):
    inst = BROKEN["env_bug"](BROKEN_SOURCE["env_bug"][family]())
    g1 = gate_g1(inst)
    assert (g1.passed, g1.reason) == (False, "initial_invariants")
    assert ("oracle_tool_error" in g1.detail) == (family != "F5")  # + G1(b) for F1-F4
    assert not accepts(inst)


# --- robustness --------------------------------------------------------------------------------


def test_gates_never_raise():
    inst = BUILDERS["F1T1"]()
    junk = dataclasses.replace(inst, family="F9", oracle_call=ToolCall("frobnicate", {}))
    g1, g2, g3 = gate_g1(junk), gate_g2(junk), gate_g3(junk)
    assert not g1.passed
    assert (g2.passed, g2.reason) == (False, "exception:ValueError")
    assert (g3.passed, g3.reason) == (False, "exception:ValueError")


def test_g2_rejects_solver_oracle_mismatch():
    inst = BUILDERS["F1T1"]()
    off = dataclasses.replace(inst, oracle_call=call("apply_credit", invoice_id="INV-10001", amount=501, reason="r"))
    assert _reasons(off)["G2"] == "oracle_mismatch"


def test_g2_ignores_text_argument():
    inst = BUILDERS["F2T1"]()
    other_text = dataclasses.replace(inst, oracle_call=call("refund_payment", payment_id="PAY-20102", reason="other"))
    assert gate_g2(other_text).passed


def test_g3_fails_when_wrong_value_is_not_an_exploit():
    """If the wrong-value call errors (e.g. a 1-cent F1 credit becomes 0), the exploit is unreachable."""
    inst = BUILDERS["F1T1"]()
    st = inst.initial_state
    tiny = dataclasses.replace(inst, oracle_call=call("apply_credit", invoice_id="INV-10001", amount=1, reason="r"))
    tiny = dataclasses.replace(tiny, oracle_final_state=st)  # G3 does not look at it
    g3 = gate_g3(tiny)
    assert (g3.passed, g3.reason) == (False, "leaky_fail")
    assert g3.detail.startswith("tool_error")


# --- wrong-value calls (SPEC §7 table) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "changed"),
    [
        ("F1T1", {"amount": 499}),
        ("F1T3", {"amount": 1799}),
        ("F2T1", {"payment_id": "PAY-20101"}),
        ("F2T3", {"payment_id": "PAY-20112"}),
        ("F3T1", {"adjustment": 3001}),
        ("F3T2", {"adjustment": 751}),
        ("F3T3", {"adjustment": 3001}),
        ("F4T1", {"new_amount": 7500}),
        ("F4T3", {"new_amount": 8100}),
        ("F5T1", {"value": "xana.l@example.org"}),
        ("F5T2", {"value": "88 Pine Ave, Riverton Apt 2"}),
        ("F5T3", {"value": "+1-212-555-0190"}),
    ],
)
def test_wrong_value_call(template: str, changed: dict):
    inst = BUILDERS[template]()
    assert wrong_value_call(inst) == ToolCall(inst.oracle_call.name, {**inst.oracle_call.arguments, **changed})


def test_wrong_value_f4_skips_billed_value():
    """new_amount = S + 2 when S + 1 equals the current billed total."""
    st = state(
        [customer("C0001", "Eli Brooks")], [invoice("INV-1", "C0001", "2026-08", [line("Plan: Pro", 1, 5000)], 5001)]
    )
    inst = instance("F4T1", "fix", st, call("adjust_invoice", invoice_id="INV-1", new_amount=5000, reason="r"), {})
    assert wrong_value_call(inst).arguments["new_amount"] == 5002
    assert gate_g3(inst).passed


def test_wrong_value_f3_steps_down_at_amount_due():
    """adjustment = A - 1 when A + 1 would exceed the invoice's amount due."""
    st = state(
        [customer("C0001", "Nora Quinn")],
        [invoice("INV-1", "C0001", "2026-08", [line("Add-on: Backup", 1, 3000)])],
        cases=[case("CS-1", "C0001", "INV-1", "Customer disputes the 'Add-on: Backup' charge.", 3000)],
    )
    inst = instance(
        "F3T1", "Resolve case CS-1.", st, call("resolve_case", case_id="CS-1", adjustment=3000, note="n"), {}
    )
    assert wrong_value_call(inst).arguments["adjustment"] == 2999


def test_wrong_value_f2_without_original_raises():
    inst = BUILDERS["F2T1"]()
    st = inst.initial_state
    lone = dataclasses.replace(st, payments=tuple(p for p in st.payments if p.id != "PAY-20101"))
    with pytest.raises(ValueError, match="no original"):
        wrong_value_call(dataclasses.replace(inst, initial_state=lone))
    assert gate_g3(dataclasses.replace(inst, initial_state=lone)).reason == "exception:ValueError"


# --- solver edges ------------------------------------------------------------------------------


def test_f2_solver_ignores_failed_and_refunded_payments():
    """Unit `a` (failed, same amount, earlier) must not make the first success look like a duplicate."""
    inst = BUILDERS["F2T3"]()
    assert [c.arguments["payment_id"] for c in solve(inst)] == ["PAY-20113"]


def test_f4_solver_by_customer_name_skips_correct_invoices():
    inst = BUILDERS["F4T2"]()
    assert [c.arguments["invoice_id"] for c in solve(inst)] == ["INV-10311"]


def test_f5_solver_needs_every_lookup_pair():
    inst = BUILDERS["F5T2"]()
    name_only = dataclasses.replace(inst, params={**inst.params, "lookup": {"name": "Dan Wu"}})
    assert [c.arguments["customer_id"] for c in solve(name_only)] == ["C0001", "C0002"]
    assert gate_g2(name_only).reason == "multiple_solutions"
