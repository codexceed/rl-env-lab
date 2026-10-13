"""The ADR 0007 v3 runs in configs/allocation.yaml: defined, placed (4B robust + 2B on node2, 4B leaky on node1),
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
    g1, g2 = {g.name: g for g in n1.groups}, {g.name: g for g in n2.groups}
    assert g2["n2-v3-4b-rob"].queue == ("v3-4b-rob-s1", "v3-4b-rob-s2", "v3-4b-rob-s3")
    assert g2["n2-v3-4b-rob"].gpus == (0, 1, 2, 3)
    assert g1["n1-v3-4b-leak"].queue == ("v3-4b-leak-s1", "v3-4b-leak-s2", "v3-4b-leak-s3")
    assert set(g1["n1-v3-4b-leak"].after) == {f"n1-v2-08-g{i}" for i in range(6)}  # after v2 0.8B and its evals
    for s, gpu in ((1, 4), (2, 5), (3, 6)):
        assert g2[f"n2-v3-2b-s{s}"].queue == (f"v3-2b-rob-s{s}", f"v3-2b-leak-s{s}")
        assert g2[f"n2-v3-2b-s{s}"].gpus == (gpu,)
    used = {gpu for name, g in g2.items() if name.startswith("n2-v3-") for gpu in g.gpus}
    assert 7 not in used and n2.eval_gpus == (7,)  # GPU 7 evaluates on node2
    assert {r for r in {**n1.runs, **n2.runs} if r.startswith("v3-")} == set(V3)


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
