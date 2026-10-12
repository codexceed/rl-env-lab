"""The ADR 0005 v2 allocation snippet the integrator adds after a go: valid for the coordinator and grpo.py.

configs/allocation.yaml does not hold the v2 runs yet (they are added only after the thinking-on base
check says go). This test merges the snippet into a copy and checks the node1 plan and its commands.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

from train import grpo
from train.coordinator import build_plan, load_config, resolve_paths, train_command

REPO = Path(__file__).resolve().parents[1]

V2_SNIPPET = """
runs:
  v2-2b-rob-s1:
    {size: 2b, reward: robust, seed: 1, timeout_s: 21600, train_command: &v2cmd "uv run python train/grpo.py
      --run-id {run_id} --size {size} --reward {reward} --seed {seed} --gpus {gpus} --output-dir {run_dir}
      --variant v2-think"}
  v2-2b-rob-s2: {size: 2b, reward: robust, seed: 2, timeout_s: 21600, train_command: *v2cmd}
  v2-2b-rob-s3: {size: 2b, reward: robust, seed: 3, timeout_s: 21600, train_command: *v2cmd}
  v2-2b-leak-s1: {size: 2b, reward: leaky, seed: 1, timeout_s: 21600, train_command: *v2cmd}
  v2-2b-leak-s2: {size: 2b, reward: leaky, seed: 2, timeout_s: 21600, train_command: *v2cmd}
  v2-2b-leak-s3: {size: 2b, reward: leaky, seed: 3, timeout_s: 21600, train_command: *v2cmd}
node1_groups:
  - {name: n1-v2-g0, gpus: [0], after: [n1-g0, n1-g1, n1-g2, n1-g3, n1-g4, n1-g5], queue: [v2-2b-rob-s1]}
  - {name: n1-v2-g1, gpus: [1], after: [n1-g0, n1-g1, n1-g2, n1-g3, n1-g4, n1-g5], queue: [v2-2b-leak-s1]}
  - {name: n1-v2-g2, gpus: [2], after: [n1-g0, n1-g1, n1-g2, n1-g3, n1-g4, n1-g5], queue: [v2-2b-rob-s2]}
  - {name: n1-v2-g3, gpus: [3], after: [n1-g0, n1-g1, n1-g2, n1-g3, n1-g4, n1-g5], queue: [v2-2b-leak-s2]}
  - {name: n1-v2-g4, gpus: [4], after: [n1-g0, n1-g1, n1-g2, n1-g3, n1-g4, n1-g5], queue: [v2-2b-rob-s3]}
  - {name: n1-v2-g5, gpus: [5], after: [n1-g0, n1-g1, n1-g2, n1-g3, n1-g4, n1-g5], queue: [v2-2b-leak-s3]}
"""


def test_v2_snippet_builds_a_valid_node1_plan():
    cfg = load_config(REPO / "configs" / "allocation.yaml")
    assert not any(rid.startswith("v2-") for rid in cfg["runs"])  # added by the integrator after a go
    snippet = yaml.safe_load(V2_SNIPPET)
    cfg["runs"].update(snippet["runs"])
    cfg["nodes"]["node1"]["groups"] += snippet["node1_groups"]
    plan = build_plan(resolve_paths(cfg, REPO), "node1")
    v2 = {rid: spec for rid, spec in plan.runs.items() if rid.startswith("v2-")}
    assert sorted(v2) == sorted(snippet["runs"])
    for rid, spec in v2.items():
        assert spec.timeout_s == 21600 and spec.size == "2b"
        cmd = train_command(plan, spec, Path("/resume"))
        argv = shlex.split(cmd)
        assert argv[argv.index("--variant") + 1] == "v2-think" and argv[-2:] == ["--resume-from", "/resume"]
        args = grpo.parse_args(argv[argv.index("train/grpo.py") + 1 :])  # grpo.py accepts it as is
        assert (args.run_id, args.variant, args.gpus) == (rid, "v2-think", list(spec.gpus))
        assert (args.reward, args.seed) == (spec.reward, spec.seed)
    groups = {g.name: g for g in plan.groups}
    assert {g.gpus for name, g in groups.items() if name.startswith("n1-v2-")} == {(i,) for i in range(6)}
    assert all(set(groups[n].after) == {f"n1-g{i}" for i in range(6)} for n in groups if n.startswith("n1-v2-"))
