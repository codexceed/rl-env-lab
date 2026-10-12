"""Checks that turn on after the sim and tasks workstreams merge (SPEC §11 merge order: sim -> tasks -> verify).

- `requires_real_sim`: the test stand-in in `verify_fakesim` agrees with the real simulator.
- `requires_real_tasks`: generated instances pass every gate; every generated broken instance is rejected.
"""

import importlib

import pytest
import verify_fakesim
from verify_builders import BUILDERS, native
from verify_support import requires_real_sim, requires_real_tasks

from envlab.calls import ParseError, ParseFailure, ToolCall
from envlab.verify.gates import accepts, run_gates
from envlab.verify.rollout import score_completion
from envlab.verify.wrong import wrong_value_call

SEEDS = range(5)


@requires_real_sim
@pytest.mark.parametrize("template", sorted(BUILDERS))
def test_fake_sim_agrees_with_real_sim(template: str):
    tools = importlib.import_module("envlab.sim.tools")
    ledger = importlib.import_module("envlab.sim.ledger")
    hashing = importlib.import_module("envlab.sim.state_hash")
    inst = BUILDERS[template]()
    st = inst.initial_state
    for masked in (False, True):
        assert hashing.state_hash(st, mask_text=masked) == verify_fakesim.state_hash(st, mask_text=masked)
    assert ledger.build_ledger(st.customers, st.invoices, st.payments) == st.ledger
    for c in (inst.oracle_call, wrong_value_call(inst)):
        real_state, real_result = tools.apply_tool(st, c)
        fake_state, fake_result = verify_fakesim.apply_tool(st, c)
        assert (real_result.ok, real_result.error) == (fake_result.ok, fake_result.error)
        assert real_state == fake_state
        assert ledger.check_invariants(real_state) == []


PARSE_CASES = [
    *(native(b().oracle_call) for b in BUILDERS.values()),
    "",
    "I refunded it.",
    "<tool_call>x</tool_call>",
    "<tool_call>",
    '<tool_call>{"name": "apply_credit", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "refund_payment", "arguments": {"payment_id": "P", "reason": 5}}</tool_call>',
    "<think>hmm</think>" + native(ToolCall("void_invoice", {"invoice_id": "INV-1", "reason": "r"})),
    native(ToolCall("void_invoice", {"invoice_id": "INV-1", "reason": "r"})) * 2,
]


@requires_real_sim
@pytest.mark.parametrize("text", PARSE_CASES)
def test_fake_parser_agrees_with_real_parser(text: str):
    parse = importlib.import_module("envlab.parse")

    def norm(r: ToolCall | ParseFailure) -> ToolCall | ParseError:
        return r if isinstance(r, ToolCall) else r.reason

    assert norm(parse.parse_tool_call(text)) == norm(verify_fakesim.parse_tool_call(text))


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
