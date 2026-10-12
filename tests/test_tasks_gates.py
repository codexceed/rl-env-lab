"""Generated data against the verify workstream's gates (SPEC §7). Turns on once verify lands."""

from __future__ import annotations

import pytest

pytest.importorskip("envlab.sim.tools")
gates = pytest.importorskip("envlab.verify.gates")

from envlab.tasks.broken import generate_broken  # noqa: E402
from envlab.tasks.generate import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, derive_seed, generate  # noqa: E402

N = 20


def test_valid_instances_pass_all_gates():
    cases = [(t, "base") for t in (*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES)] + [(t, "hard") for t in TRAIN_TEMPLATES]
    failures = []
    total = 0
    for template, variant in cases:
        for i in range(N):
            inst = generate(template, derive_seed(55, template, variant, i), variant=variant)
            total += 1
            if not gates.accepts(inst):
                failures.append((template, variant, i, gates.run_gates(inst)))
    assert len(failures) <= total // 100, failures[:5]


@pytest.mark.parametrize("template", [t for t in ("F1T4", "F1T5", "F4T4", "F4T5") if t in TRAIN_TEMPLATES])
@pytest.mark.parametrize("variant", ["base", "hard"])
def test_candidate_templates_pass_every_gate(template: str, variant: str):
    """The medium-difficulty candidates (SPEC §4): G1-G3 hold on every generated instance."""
    for i in range(50):
        inst = generate(template, derive_seed(56, template, variant, i), variant=variant)
        assert [g.reason for g in gates.run_gates(inst)] == ["ok", "ok", "ok"], inst.id


def test_every_broken_instance_fails_a_gate():
    accepted = [inst.id for inst in generate_broken() if gates.accepts(inst)]
    assert accepted == []
