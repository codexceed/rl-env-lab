from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from eval.watcher import (
    Candidate,
    Watcher,
    WatcherSettings,
    base_eval_id,
    pending_base_evals,
    pending_checkpoints,
    priority_key,
)
from train.checkpoints import (
    atomic_write_json,
    eval_done_path,
    eval_failed_path,
    eval_output_path,
    eval_step_dir,
    step_name,
)

PY = sys.executable

FAKE_EVAL = r"""
import json, os, sys, time
run_id, step, output, log, sleep_s, rc = sys.argv[1:7]
t0 = time.time()
time.sleep(float(sleep_s))
if int(rc) == 0:
    with open(output, "w") as fh:
        fh.write(json.dumps({"ok": True}) + "\n")
with open(log, "a") as fh:
    fh.write(json.dumps({"run": run_id, "step": int(step), "t0": t0, "t1": time.time(),
                         "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}) + "\n")
sys.exit(int(rc))
"""


def add_ckpt(runs: Path, run_id: str, step: int, final: bool = False) -> None:
    d = eval_step_dir(runs / run_id, step)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "COMPLETE").write_text(json.dumps({"step": step, "final": final}))


@pytest.fixture
def env(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "evals"
    runs.mkdir()
    script = tmp_path / "fake_eval.py"
    script.write_text(FAKE_EVAL)
    log = tmp_path / "eval_log.jsonl"
    return tmp_path, runs, evals, script, log


def settings(env, *, sleep=0.0, rc=0, gpus=(0,), **kw) -> WatcherSettings:
    tmp, runs, evals, script, log = env
    cmd = f"{PY} {script} {{run_id}} {{step}} {{output}} {log} {sleep} {rc}"
    return WatcherSettings(
        runs_root=runs,
        evals_root=evals,
        eval_command=cmd,
        workdir=tmp,
        static_gpus=list(gpus),
        poll_interval_s=0.02,
        kill_grace_s=0.5,
        **kw,
    )


def drive(*watchers: Watcher, until, timeout=15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        for w in watchers:
            w.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def eval_log(env) -> list[dict]:
    log = env[4]
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_priority_final_then_every_100_then_rest(env):
    _, runs, evals, _, _ = env
    for s in (50, 100, 150, 200):
        add_ckpt(runs, "a-rob-s1", s)
    add_ckpt(runs, "a-rob-s1", 230, final=True)
    add_ckpt(runs, "b-leak-s1", 100)
    add_ckpt(runs, "b-leak-s1", 120, final=True)
    eval_step_dir(runs / "a-rob-s1", 250).mkdir()  # no COMPLETE: ignored
    (runs / "_coordinator").mkdir()

    order = [(c.run_id, c.step) for c in pending_checkpoints(runs, evals)]

    assert order == [
        ("b-leak-s1", 120),
        ("a-rob-s1", 230),
        ("a-rob-s1", 100),
        ("b-leak-s1", 100),
        ("a-rob-s1", 200),
        ("a-rob-s1", 50),
        ("a-rob-s1", 150),
    ]
    assert priority_key(Candidate("x", 300, Path("."), True))[0] == 0


def test_watcher_evaluates_in_priority_order_and_writes_done(env):
    _, runs, evals, _, _ = env
    for s in (50, 100, 150):
        add_ckpt(runs, "a-rob-s1", s)
    add_ckpt(runs, "a-rob-s1", 170, final=True)
    w = Watcher(settings(env, gpus=(3,)))

    drive(w, until=lambda: len(eval_log(env)) == 4 and not w.jobs)

    assert [r["step"] for r in eval_log(env)] == [170, 100, 50, 150]
    assert all(r["gpus"] == "3" for r in eval_log(env))
    for s in (50, 100, 150, 170):
        done = json.loads(eval_done_path(evals, "a-rob-s1", s).read_text())
        assert done["step"] == s
        assert not (evals / "a-rob-s1" / f"{step_name(s)}.claim").exists()  # released on success too
    assert pending_checkpoints(runs, evals) == []


def test_multiple_eval_gpus_run_concurrently(env):
    _, runs, _, _, _ = env
    add_ckpt(runs, "a-rob-s1", 50)
    add_ckpt(runs, "a-rob-s1", 100)
    w = Watcher(settings(env, sleep=0.5, gpus=(6, 7)))

    drive(w, until=lambda: len(eval_log(env)) == 2 and not w.jobs)

    a, b = eval_log(env)
    assert {a["gpus"], b["gpus"]} == {"6", "7"}
    assert a["t0"] < b["t1"] and b["t0"] < a["t1"]  # overlapped


def test_gpus_per_eval_groups_gpus(env):
    _, runs, _, _, _ = env
    add_ckpt(runs, "a-rob-s1", 50)
    w = Watcher(settings(env, gpus=(6, 7), gpus_per_eval=2))
    drive(w, until=lambda: len(eval_log(env)) == 1 and not w.jobs)
    assert eval_log(env)[0]["gpus"] == "6,7"


def test_failure_retries_then_marks_failed_and_never_deletes(env):
    tmp, runs, evals, _, _ = env
    add_ckpt(runs, "a-rob-s1", 50)
    before = set(tmp.rglob("*"))
    w = Watcher(settings(env, rc=2, retries=1))

    drive(w, until=lambda: eval_failed_path(evals, "a-rob-s1", 50).exists() and not w.jobs)

    assert len(eval_log(env)) == 2  # first try + 1 retry
    assert not eval_done_path(evals, "a-rob-s1", 50).exists()
    d = evals / "a-rob-s1"
    assert len(list(d.glob(f"{step_name(50)}.fail-*"))) == 2
    assert len(list(d.glob(f"{step_name(50)}.claim.released-*"))) == 2
    assert before <= set(tmp.rglob("*"))  # nothing removed
    assert pending_checkpoints(runs, evals) == []


def test_eval_timeout_kills_and_records_failure(env):
    _, runs, evals, _, _ = env
    add_ckpt(runs, "a-rob-s1", 50)
    w = Watcher(settings(env, sleep=30, retries=0, timeout_s=0.3))
    drive(w, until=lambda: eval_failed_path(evals, "a-rob-s1", 50).exists() and not w.jobs)
    assert json.loads(eval_failed_path(evals, "a-rob-s1", 50).read_text())["reason"] == "timeout"


def test_two_watchers_never_evaluate_the_same_step(env):
    _, runs, _evals, _, _ = env
    for s in (50, 100, 150, 200):
        add_ckpt(runs, "a-rob-s1", s)
    w1 = Watcher(settings(env, sleep=0.2, gpus=(0,)))
    w2 = Watcher(settings(env, sleep=0.2, gpus=(1,)))

    drive(w1, w2, until=lambda: len(eval_log(env)) == 4 and not w1.jobs and not w2.jobs)

    steps = sorted(r["step"] for r in eval_log(env))
    assert steps == [50, 100, 150, 200]


def test_stale_claim_from_dead_process_is_taken_over(env):
    _, runs, evals, _, _ = env
    add_ckpt(runs, "a-rob-s1", 50)
    dead = subprocess.Popen([PY, "-c", "pass"])
    dead.wait()
    claim = evals / "a-rob-s1" / f"{step_name(50)}.claim"
    claim.parent.mkdir(parents=True)
    w = Watcher(settings(env))
    claim.write_text(json.dumps({"host": w.host, "pid": dead.pid, "token": "old"}))

    drive(w, until=lambda: eval_done_path(evals, "a-rob-s1", 50).exists())

    assert list(claim.parent.glob(f"{claim.name}.stale-*"))


def test_live_foreign_claim_is_respected(env):
    _, runs, evals, _, _ = env
    add_ckpt(runs, "a-rob-s1", 50)
    claim = evals / "a-rob-s1" / f"{step_name(50)}.claim"
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps({"host": "other-node", "pid": 1, "token": "t"}))
    w = Watcher(settings(env))
    for _ in range(5):
        w.tick()
    assert w.jobs == [] and eval_log(env) == []


def test_lease_controls_gpus_ack_and_drain(env):
    tmp, runs, _, _, _ = env
    lease, state = tmp / "lease.json", tmp / "state.json"
    add_ckpt(runs, "a-rob-s1", 50)
    w = Watcher(settings(env, gpus=(), lease_file=lease, state_file=state))

    assert w.tick() is True  # no lease yet: no work
    assert w.jobs == []

    atomic_write_json(lease, {"gen": 3, "gpus": [], "drain": False})
    w.tick()
    st = json.loads(state.read_text())
    assert st["ack_gen"] == 3 and st["pid"] == os.getpid() and w.jobs == []

    atomic_write_json(lease, {"gen": 4, "gpus": [5], "drain": False})
    w.tick()
    assert json.loads(state.read_text())["busy"] == {"5": f"a-rob-s1/{step_name(50)}"}

    atomic_write_json(lease, {"gen": 5, "gpus": [], "drain": True})
    drive(w, until=lambda: not w.jobs)
    assert w.tick() is False  # drained and idle -> exit
    assert json.loads(state.read_text())["ack_gen"] == 5


def test_watcher_cli_runs_standalone(env, tmp_path):
    _, runs, evals, script, log = env
    add_ckpt(runs, "a-rob-s1", 50, final=True)
    cfg = {
        "paths": {"workdir": str(tmp_path), "runs": str(runs), "evals": str(evals)},
        "commands": {"eval": f"{PY} {script} {{run_id}} {{step}} {{output}} {log} 0 0"},
        "eval": {"poll_interval_s": 0.05},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    lease = tmp_path / "lease.json"
    atomic_write_json(lease, {"gen": 1, "gpus": [0], "drain": False})
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [PY, "-m", "eval.watcher", "--config", str(cfg_path), "--lease-file", str(lease)],
        cwd=repo,
    )
    try:
        end = time.time() + 15
        while not eval_done_path(evals, "a-rob-s1", 50).exists() and time.time() < end:
            time.sleep(0.05)
        assert eval_done_path(evals, "a-rob-s1", 50).exists()
        atomic_write_json(lease, {"gen": 2, "gpus": [], "drain": True})
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------- base-model evals

REPO = Path(__file__).resolve().parents[1]
V3_CMD = "python train/grpo.py --run-id {run_id} --variant v3"
V2_CMD = "python train/grpo.py --run-id {run_id} --variant v2-think"
KNOWN = {
    "2b-rob-s1": {"size": "2b", "reward": "robust", "seed": 1},
    "v2-2b-leak-s1": {"size": "2b", "reward": "leaky", "seed": 1, "train_command": V2_CMD},
    "v3-4b-rob-s1": {"size": "4b", "reward": "robust", "seed": 1, "train_command": V3_CMD},
    "v3-4b-leak-s1": {"size": "4b", "reward": "leaky", "seed": 1, "train_command": V3_CMD},
}


def base_settings(env, **kw) -> WatcherSettings:
    _, _, _, script, log = env
    cmd = f"{PY} {script} {{run_id}} {{step}} {{output}} {log} 0 0 {{variant_args}}"
    return settings(env, base_eval_command=cmd, run_meta=dict(KNOWN), **kw)


def test_base_eval_ids_match_heldout():
    from eval.heldout import parse_args, resolve_target

    for size, variant in (("2b", None), ("4b", "v3")):
        argv = ["--size", size, "--checkpoint", "base", *(["--variant", variant] if variant else [])]
        assert resolve_target(parse_args(argv)).run_id == base_eval_id(size, variant)


def test_base_evals_one_per_size_and_project_version(env):
    _, runs, evals, _, _ = env

    out = pending_base_evals(runs, evals, set(KNOWN), KNOWN)

    # v1 and v2 runs share base-2b (thinking off, v1 tasks); the v3 runs get v3-base-4b, run with --variant v3
    assert [(c.run_id, c.size, c.variant, c.step) for c in out] == [
        ("base-2b", "2b", None, 0),
        ("v3-base-4b", "4b", "v3", 0),
    ]
    assert all(c.base and priority_key(c)[0] == -1 for c in out)


def test_base_eval_variant_comes_from_the_run_dir_first(env):
    _, runs, evals, _, _ = env
    add_ckpt(runs, "4b-rob-s9", 100)
    (runs / "4b-rob-s9" / "train_config.json").write_text(json.dumps({"variant": "v3"}))
    assert [c.run_id for c in pending_base_evals(runs, evals, {"4b-rob-s9"})] == ["v3-base-4b"]


def test_base_eval_skipped_when_output_done_or_failed_exists(env):
    _, runs, evals, _, _ = env
    by_hand = eval_output_path(evals, "base-2b", 0)  # run by hand: the output, no .done
    failed = eval_failed_path(evals, "v3-base-4b", 0)
    for path in (by_hand, failed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    assert pending_base_evals(runs, evals, set(KNOWN), KNOWN) == []

    by_hand.unlink()
    eval_done_path(evals, "base-2b", 0).write_text("{}")
    assert pending_base_evals(runs, evals, set(KNOWN), KNOWN) == []


def test_watcher_runs_base_evals_first_then_checkpoints(env):
    _, runs, evals, _, _ = env
    add_ckpt(runs, "2b-rob-s1", 50)
    add_ckpt(runs, "2b-rob-s1", 100, final=True)
    w = Watcher(base_settings(env, base_run_ids={"v3-4b-rob-s1"}))

    drive(w, until=lambda: len(eval_log(env)) == 4 and not w.jobs)

    assert [(r["run"], r["step"]) for r in eval_log(env)] == [
        ("base-2b", 0),  # for the run with a checkpoint
        ("v3-base-4b", 0),  # for a run in the node's queue, before it starts
        ("2b-rob-s1", 100),
        ("2b-rob-s1", 50),
    ]
    for rid in ("base-2b", "v3-base-4b"):
        assert eval_done_path(evals, rid, 0).exists()
        assert eval_output_path(evals, rid, 0).exists()
    assert w.pending() == []


def test_no_base_evals_without_eval_base_template(env):
    _, runs, _, _, _ = env
    add_ckpt(runs, "2b-rob-s1", 50)
    w = Watcher(settings(env, run_meta=dict(KNOWN), base_run_ids=set(KNOWN)))
    assert [c.run_id for c in w.pending()] == ["2b-rob-s1"]


def test_repo_eval_base_command_renders_the_heldout_base_eval(env):
    from eval.heldout import parse_args, resolve_target

    _, _, evals, _, _ = env
    cfg = yaml.safe_load((REPO / "configs" / "allocation.yaml").read_text())
    w = Watcher(WatcherSettings.from_config(cfg, "node1", evals_root=evals))

    v1 = Candidate("base-2b", 0, Path("base"), False, base=True, size="2b")
    v3 = Candidate("v3-base-4b", 0, Path("base"), False, base=True, size="4b", variant="v3")
    out = eval_output_path(evals, "v3-base-4b", 0)
    assert w.command(v3, (6,)) == (
        f"uv run python eval/heldout.py --size 4b --checkpoint base --variant v3 --output {out} --gpus 6"
    )
    assert "--variant" not in w.command(v1, (7,))
    for cand in (v1, v3):
        argv = shlex.split(w.command(cand, (6,)))
        assert argv[:4] == ["uv", "run", "python", "eval/heldout.py"]
        args = parse_args(argv[4:])
        assert resolve_target(args).run_id == cand.run_id
        assert args.output == eval_output_path(evals, cand.run_id, 0)


def test_repo_config_schedules_each_nodes_base_evals(tmp_path):
    cfg = yaml.safe_load((REPO / "configs" / "allocation.yaml").read_text())
    expected = {"node1": {"base-2b", "base-0.8b", "v3-base-4b", "v3-base-2b"}, "node2": {"base-0.8b", "base-4b"}}
    for node, ids in expected.items():
        s = WatcherSettings.from_config(cfg, node, runs_root=tmp_path / "runs", evals_root=tmp_path / "evals")
        assert s.base_eval_command
        got = pending_base_evals(s.runs_root, s.evals_root, s.base_scope(), s.run_meta)
        assert {c.run_id for c in got} == ids, node
