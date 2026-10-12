"""The ADR 0007 v3 runs in configs/allocation.yaml: defined, not yet queued, and valid for grpo.py and the
coordinator once the integrator places them."""

from __future__ import annotations

import copy
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


def test_v3_runs_are_not_scheduled_yet_and_existing_plans_are_unchanged():
    cfg = resolve_paths(load_config(REPO / "configs" / "allocation.yaml"), REPO)
    for node in cfg["nodes"]:
        plan = build_plan(cfg, node)
        assert not any(rid.startswith("v3-") for rid in plan.runs)
        assert not any(rid.startswith("v3-") for g in plan.groups for rid in g.queue)


def test_v3_commands_parse_with_grpo_once_placed():
    cfg = load_config(REPO / "configs" / "allocation.yaml")
    placed = copy.deepcopy(cfg)  # how the integrator would queue them: 4B on 4 GPUs, 2B on one
    placed["nodes"]["node2"]["groups"] += [
        {"name": "t-v3-4b", "gpus": [0, 1, 2, 3], "queue": [r for r in V3 if r.startswith("v3-4b-")]},
        {"name": "t-v3-2b", "gpus": [4], "queue": [r for r in V3 if r.startswith("v3-2b-")]},
    ]
    plan = build_plan(resolve_paths(placed, REPO), "node2")
    for rid in V3:
        spec = plan.runs[rid]
        assert spec.timeout_s == (43200 if spec.size == "4b" else 36000)
        argv = shlex.split(train_command(plan, spec, Path("/resume")))
        assert argv[argv.index("--variant") + 1] == "v3" and argv[-2:] == ["--resume-from", "/resume"]
        args = grpo.parse_args(argv[argv.index("train/grpo.py") + 1 :])
        assert (args.run_id, args.variant, args.gpus) == (rid, "v3", list(spec.gpus))
        assert (args.size, args.reward, args.seed) == (spec.size, spec.reward, spec.seed)
        cfg_v3 = grpo.load_config(grpo.DEFAULT_CONFIG, args.size, args.variant)
        grpo.check_context(cfg_v3, grpo.resolve_layout(cfg_v3, args).vllm_max_model_length)
