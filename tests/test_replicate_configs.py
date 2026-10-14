"""configs/replicate/v{1,2,3}.yaml (README, "Reproduce the study"): one clean two-node plan per project version,
resolved with the coordinator's own code. Each must queue every run of its version exactly once, never put two
groups that can run at the same time on one GPU, keep eval GPUs out of training, and launch commands that
train/grpo.py accepts."""

from __future__ import annotations

import shlex
from itertools import combinations
from pathlib import Path

import pytest

from train import grpo
from train.coordinator import NodePlan, build_plan, describe_plan, load_config, resolve_paths, train_command

REPO = Path(__file__).resolve().parents[1]
NODES = ("node1", "node2")


def _ids(prefix: str, sizes: tuple[str, ...]) -> set[str]:
    return {f"{prefix}{size}-{r}-s{s}" for size in sizes for r in ("rob", "leak") for s in (1, 2, 3)}


# version -> (expected run ids, --variant in the train command, timeout per size)
VERSIONS = {
    "v1": (_ids("", ("0.8b", "2b", "4b")), None, {"0.8b": 21600, "2b": 28800, "4b": 28800}),
    "v2": (_ids("v2-", ("2b", "0.8b")), "v2-think", {"0.8b": 21600, "2b": 36000}),
    "v3": (_ids("v3-", ("4b", "2b")), "v3", {"2b": 36000, "4b": 43200}),
}
GPUS_PER_RUN = {"0.8b": 1, "2b": 1, "4b": 4}


def _config(version: str) -> dict:
    return load_config(REPO / "configs" / "replicate" / f"{version}.yaml")


def _plans(version: str) -> dict[str, NodePlan]:
    cfg = resolve_paths(_config(version), REPO)
    return {node: build_plan(cfg, node) for node in NODES}


def _predecessors(plan: NodePlan) -> dict[str, set[str]]:
    """Every group that must drain before each group starts (transitive ``after``)."""
    direct = {g.name: set(g.after) for g in plan.groups}
    out: dict[str, set[str]] = {}
    for name in direct:
        seen: set[str] = set()
        todo = list(direct[name])
        while todo:
            g = todo.pop()
            if g not in seen:
                seen.add(g)
                todo.extend(direct[g])
        out[name] = seen
    return out


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_every_run_of_the_version_is_queued_exactly_once(version):
    expected = VERSIONS[version][0]
    cfg = _config(version)
    assert set(cfg["runs"]) == expected
    queued = [rid for n in cfg["nodes"].values() for g in n["groups"] for rid in g["queue"]]
    assert sorted(queued) == sorted(expected)  # once each, nothing from another version
    plans = _plans(version)
    assert set(plans["node1"].runs) | set(plans["node2"].runs) == expected
    assert not set(plans["node1"].runs) & set(plans["node2"].runs)
    for plan in plans.values():
        assert describe_plan(plan).startswith(f"node {plan.name}:")  # `--print-plan` works


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_no_gpu_is_double_booked_by_groups_that_can_overlap(version):
    for plan in _plans(version).values():
        before = _predecessors(plan)
        for a, b in combinations(plan.groups, 2):
            ordered = a.name in before[b.name] or b.name in before[a.name]
            shared = set(a.gpus) & set(b.gpus)
            assert ordered or not shared, f"{version}/{plan.name}: {a.name} and {b.name} can overlap on GPUs {shared}"


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_eval_gpus_are_reserved_on_every_node(version):
    for plan in _plans(version).values():
        training = {gpu for g in plan.groups for gpu in g.gpus}
        assert len(plan.eval_gpus) >= 2, f"{version}/{plan.name}: reserve eval GPUs (learnings §5)"
        assert not set(plan.eval_gpus) & training, f"{version}/{plan.name}: eval GPUs also train"
        assert set(plan.eval_gpus) | training <= set(plan.gpus)
        assert plan.settings.watcher_enabled and plan.settings.lend_idle_gpus


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_runs_have_their_gpus_timeouts_and_a_command_grpo_accepts(version):
    _, variant, timeouts = VERSIONS[version]
    for plan in _plans(version).values():
        for rid, spec in plan.runs.items():
            assert len(spec.gpus) == GPUS_PER_RUN[spec.size], rid
            assert spec.timeout_s == timeouts[spec.size], rid
            argv = shlex.split(train_command(plan, spec, Path("/resume")))
            assert argv[-2:] == ["--resume-from", "/resume"]
            args = grpo.parse_args(argv[argv.index("train/grpo.py") + 1 :])
            assert (args.run_id, args.size, args.reward, args.seed) == (rid, spec.size, spec.reward, spec.seed)
            assert (args.variant, args.gpus) == (variant, list(spec.gpus))
        assert plan.settings.deadline_ts is None and plan.settings.shutdown_cmd is None  # no cutoff, no release
        assert plan.settings.pruning_enabled and plan.settings.keep_last == 2


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_shared_sections_match_allocation(version):
    """Copied from configs/allocation.yaml, not reinvented: only the variant flag, cutoff and timeouts differ."""
    alloc, cfg = load_config(REPO / "configs" / "allocation.yaml"), _config(version)
    for key in ("version", "paths", "transfer", "checkpoints", "eval"):
        assert cfg[key] == alloc[key], key
    assert {k: v for k, v in cfg["coordinator"].items() if k != "node_hard_deadline"} == {
        k: v for k, v in alloc["coordinator"].items() if k != "node_hard_deadline"
    }
    variant = VERSIONS[version][1]
    suffix = f" --variant {variant}" if variant else ""
    assert cfg["commands"] == {**alloc["commands"], "train": alloc["commands"]["train"] + suffix}
    for size, sc in cfg["sizes"].items():
        assert sc["gpus_per_run"] == alloc["sizes"][size]["gpus_per_run"]
    for rid, rc in cfg["runs"].items():  # same ids, sizes, rewards and seeds as the study ran
        assert {k: alloc["runs"][rid][k] for k in ("size", "reward", "seed")} == rc
