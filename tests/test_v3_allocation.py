"""The ADR 0007 v3 runs in configs/allocation.yaml: defined, placed on node1 (4B pairs, then 2B),
and valid for grpo.py and the coordinator."""

from __future__ import annotations

import shlex
from pathlib import Path

from train import grpo
from train.coordinator import build_plan, load_config, resolve_paths, train_command

REPO = Path(__file__).resolve().parents[1]
V3 = [f"v3-{size}-{r}-s{s}" for size in ("4b", "2b") for r in ("rob", "leak") for s in (1, 2, 3)]


def test_v3_runs_are_defined_with_their_timeouts():
    runs = load_config(REPO / "configs" / "allocation.yaml")["runs"]
    assert {rid for rid in runs if rid.startswith("v3-")} == set(V3)
    for rid in V3:
        size, r, seed = rid.split("-")[1:]
        spec = runs[rid]
        assert (spec["size"], spec["reward"], spec["seed"]) == (
            size,
            {"rob": "robust", "leak": "leaky"}[r],
            int(seed[1:]),
        )
        assert spec["timeout_s"] == {"4b": 43200, "2b": 36000}[size]


def test_v3_placement():
    cfg = resolve_paths(load_config(REPO / "configs" / "allocation.yaml"), REPO)
    n1, n2 = build_plan(cfg, "node1"), build_plan(cfg, "node2")
    assert not any(r.startswith("v3-") for r in n2.runs)  # all of v3 on node1 (node2's disk was too slow)
    g = {grp.name: grp for grp in n1.groups}
    assert g["n1-v3-4b-rob"].queue == ("v3-4b-rob-s1", "v3-4b-rob-s2", "v3-4b-rob-s3")
    assert g["n1-v3-4b-leak"].queue == ("v3-4b-leak-s1", "v3-4b-leak-s2", "v3-4b-leak-s3")
    assert (g["n1-v3-4b-rob"].gpus, g["n1-v3-4b-leak"].gpus) == ((0, 1, 2, 3), (4, 5, 6, 7))
    for name in ("n1-v3-4b-rob", "n1-v3-4b-leak"):
        assert set(g[name].after) == {f"n1-v2-08-g{i}" for i in range(6)}  # after v2 0.8B and its evals
    two_b = [grp for name, grp in g.items() if name.startswith("n1-v3-2b-")]
    assert sorted(rid for grp in two_b for rid in grp.queue) == sorted(r for r in V3 if r.startswith("v3-2b-"))
    assert {grp.gpus for grp in two_b} == {(i,) for i in range(6)}  # GPU 6-7 evaluate
    assert all(set(grp.after) == {"n1-v3-4b-rob", "n1-v3-4b-leak"} for grp in two_b)
    assert {r for r in n1.runs if r.startswith("v3-")} == set(V3)


def test_v3_commands_parse_with_grpo():
    cfg = resolve_paths(load_config(REPO / "configs" / "allocation.yaml"), REPO)
    plans = {node: build_plan(cfg, node) for node in ("node1", "node2")}
    for rid in V3:
        plan = next(p for p in plans.values() if rid in p.runs)
        spec = plan.runs[rid]
        assert spec.timeout_s == (43200 if spec.size == "4b" else 36000)
        argv = shlex.split(train_command(plan, spec, Path("/resume")))
        assert argv[argv.index("--variant") + 1] == "v3" and argv[-2:] == ["--resume-from", "/resume"]
        args = grpo.parse_args(argv[argv.index("train/grpo.py") + 1 :])
        assert (args.run_id, args.variant, args.gpus) == (rid, "v3", list(spec.gpus))
        assert (args.size, args.reward, args.seed) == (spec.size, spec.reward, spec.seed)
        cfg_v3 = grpo.load_config(grpo.DEFAULT_CONFIG, args.size, args.variant)
        grpo.check_context(cfg_v3, grpo.resolve_layout(cfg_v3, args).vllm_max_model_length)
