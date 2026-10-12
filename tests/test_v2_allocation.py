"""The ADR 0005 v2 runs in configs/allocation.yaml (added after the go, ADR 0006): valid for the
coordinator and for grpo.py, on node1 after its 2B block."""

from __future__ import annotations

import shlex
from pathlib import Path

from train import grpo
from train.coordinator import build_plan, load_config, resolve_paths, train_command

REPO = Path(__file__).resolve().parents[1]


def test_v2_runs_build_a_valid_node1_plan():
    cfg = load_config(REPO / "configs" / "allocation.yaml")
    plan = build_plan(resolve_paths(cfg, REPO), "node1")
    v2 = {rid: spec for rid, spec in plan.runs.items() if rid.startswith("v2-2b-")}
    assert sorted(v2) == sorted(f"v2-2b-{r}-s{s}" for r in ("rob", "leak") for s in (1, 2, 3))
    for rid, spec in v2.items():
        assert spec.timeout_s == 36000 and spec.size == "2b"
        cmd = train_command(plan, spec, Path("/resume"))
        argv = shlex.split(cmd)
        assert argv[argv.index("--variant") + 1] == "v2-think" and argv[-2:] == ["--resume-from", "/resume"]
        args = grpo.parse_args(argv[argv.index("train/grpo.py") + 1 :])  # grpo.py accepts it as is
        assert (args.run_id, args.variant, args.gpus) == (rid, "v2-think", list(spec.gpus))
        assert (args.reward, args.seed) == (spec.reward, spec.seed)
    groups = {g.name: g for g in plan.groups}
    v2_2b = [n for n in groups if n.startswith("n1-v2-g")]
    assert {groups[n].gpus for n in v2_2b} == {(i,) for i in range(6)}
    assert all(set(groups[n].after) == {f"n1-g{i}" for i in range(6)} for n in v2_2b)
    assert not any(rid.startswith("v2-2b-") for rid in build_plan(resolve_paths(cfg, REPO), "node2").runs)


def test_v2_extension_to_08b_on_node1():
    cfg = load_config(REPO / "configs" / "allocation.yaml")
    n1, n2 = (build_plan(resolve_paths(cfg, REPO), node) for node in ("node1", "node2"))
    assert not any(rid.startswith("v2-4b-") for rid in {**n1.runs, **n2.runs})  # dropped: 4B gate no-go (ADR 0007)
    assert not any(rid.startswith("v2-") for rid in n2.runs)
    v2 = {rid for rid in n1.runs if rid.startswith("v2-0.8b-")}
    assert v2 == {f"v2-0.8b-{r}-s{s}" for r in ("rob", "leak") for s in (1, 2, 3)}
    for rid in v2:
        argv = shlex.split(train_command(n1, n1.runs[rid]))
        assert argv[argv.index("--variant") + 1] == "v2-think" and len(n1.runs[rid].gpus) == 1
    g1 = {g.name: g for g in n1.groups}
    v2_08 = [n for n in g1 if n.startswith("n1-v2-08-")]
    assert {g1[n].gpus for n in v2_08} == {(i,) for i in range(6)}  # GPU 6-7 stay eval GPUs
    assert all(set(g1[n].after) == {f"n1-v2-g{i}" for i in range(6)} for n in v2_08)  # after the v2 2B block
