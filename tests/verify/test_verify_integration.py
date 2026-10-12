"""Fixture sanity on the real simulator, plus checks that turn on once the tasks workstream merges.

- The hand-built fixtures are valid SPEC states: ledgers match `build_ledger`, invariants hold before and after.
- `requires_real_tasks`: generated instances pass every gate; every generated broken instance is rejected.
"""

from __future__ import annotations

import importlib

import pytest
from verify_builders import BUILDERS, native
from verify_support import requires_real_tasks

from envlab.sim.ledger import build_ledger, check_invariants
from envlab.verify.gates import accepts, run_gates
from envlab.verify.rollout import execute, score_completion
from envlab.verify.wrong import wrong_value_call

SEEDS = range(5)


@pytest.mark.parametrize("template", sorted(BUILDERS))
def test_fixtures_are_valid_states(template: str):
    inst = BUILDERS[template]()
    st = inst.initial_state
    assert build_ledger(st.customers, st.invoices, st.payments) == st.ledger
    assert check_invariants(st) == []
    assert check_invariants(inst.oracle_final_state) == []
    wrong = execute(inst, wrong_value_call(inst))
    assert wrong.result is not None and wrong.result.ok
    assert check_invariants(wrong.final_state) == []


@requires_real_tasks
@pytest.mark.parametrize("variant", ["base", "hard"])
def test_generated_instances_pass_all_gates(variant: str):
    generate = importlib.import_module("envlab.tasks.generate")
    templates = list(generate.TRAIN_TEMPLATES)
    if variant == "base":
        templates += list(generate.HELDOUT_TEMPLATES)
    failures = []
    for template in templates:
        for seed in SEEDS:
            inst = generate.generate(template, seed, variant=variant)
            if not accepts(inst):
                failures.append((template, seed, [(g.gate, g.reason, g.detail) for g in run_gates(inst)]))
    assert failures == []


@requires_real_tasks
def test_generated_correct_completion_scores_both():
    generate = importlib.import_module("envlab.tasks.generate")
    prompts = pytest.importorskip("envlab.prompts")
    for template in (*generate.TRAIN_TEMPLATES, *generate.HELDOUT_TEMPLATES):
        inst = generate.generate(template, 0)
        for completion in (native(inst.oracle_call), prompts.render_call(inst.oracle_call)):
            sample = score_completion(inst, completion)
            assert (sample.robust_pass, sample.leaky_pass) == (True, True), (template, sample)
        exploit = score_completion(inst, prompts.render_call(wrong_value_call(inst)))
        assert (exploit.leaky_pass, exploit.robust_pass) == (True, False), template


EXPECTED_GATE = {
    "impossible": lambda g: not g["G1"].passed and "oracle_tool_error" in g["G1"].detail and not g["G2"].passed,
    "ambiguous": lambda g: not g["G2"].passed,
    "grader_bug": lambda g: not g["G1"].passed and "oracle_state_mismatch" in g["G1"].detail,
    "env_bug": lambda g: g["G1"].reason == "initial_invariants",
}


@requires_real_tasks
def test_generated_broken_rejected_by_expected_gate():
    broken = importlib.import_module("envlab.tasks.broken")
    instances = broken.generate_broken()
    assert len(instances) == 100
    wrong = []
    for inst in instances:
        gates = {g.gate: g for g in run_gates(inst)}
        if all(g.passed for g in gates.values()) or not EXPECTED_GATE[inst.broken_type](gates):
            wrong.append((inst.id, {k: (g.reason, g.detail) for k, g in gates.items()}))
    assert wrong == []
