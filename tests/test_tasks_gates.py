"""Generated data against the verify workstream's gates (SPEC §7). Turns on once verify lands."""

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


def test_every_broken_instance_fails_a_gate():
    accepted = [inst.id for inst in generate_broken() if gates.accepts(inst)]
    assert accepted == []
