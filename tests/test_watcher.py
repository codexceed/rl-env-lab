import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from eval.watcher import Candidate, Watcher, WatcherSettings, pending_checkpoints, priority_key
from train.checkpoints import (
    atomic_write_json,
    eval_done_path,
    eval_failed_path,
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
    _, runs, evals, _, _ = env
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
