from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from train.checkpoints import eval_done_path, read_status, run_dir, step_name, transferred_path
from train.coordinator import ConfigError, Coordinator, build_plan, load_config, resolve_paths

PY = sys.executable
REPO = Path(__file__).resolve().parents[1]

# Fake trainer. Behaviour per run comes from a JSON file, so one template serves every run.
FAKE_TRAIN = r"""
import json, os, subprocess, sys, time
run_id, rdir, beh_path, log = sys.argv[1:5]
resume = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == "--resume-from" else None
beh = json.load(open(beh_path)).get(run_id, {})
def note(ev):
    with open(log, "a") as fh:
        fh.write(json.dumps({"ev": ev, "run": run_id, "t": time.time(), "resume": resume,
                             "gpus": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
note("start")
if beh.get("child_pid_file"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open(beh["child_pid_file"], "w").write(str(child.pid))
for n in beh.get("recovery", []):
    d = os.path.join(rdir, "recovery", f"checkpoint-{n}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "COMPLETE"), "w").close()
time.sleep(beh.get("sleep", 0))
if beh.get("crash_first") and resume is None:
    note("crash")
    sys.exit(3)
steps = beh.get("evals", [])
for s in steps:
    d = os.path.join(rdir, "eval", f"step-{s:05d}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "config.json"), "w").write("{}")
    open(os.path.join(d, "model.safetensors"), "wb").write(b"0" * 8)
    with open(os.path.join(d, "COMPLETE"), "w") as fh:
        json.dump({"step": s, "final": s == max(steps)}, fh)
time.sleep(beh.get("post_sleep", 0))
note("end")
sys.exit(beh.get("exit", 0))
"""

FAKE_EVAL = r"""
import json, os, sys, time
run_id, step, output, log, sleep_s = sys.argv[1:6]
t0 = time.time()
time.sleep(float(sleep_s))
open(output, "w").write("{}\n")
with open(log, "a") as fh:
    fh.write(json.dumps({"ev": "eval", "run": run_id, "step": int(step), "t0": t0,
                         "t1": time.time(), "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}) + "\n")
"""


class Env:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.runs = tmp / "runs"
        self.evals = tmp / "evals"
        self.log = tmp / "events.jsonl"
        self.beh = tmp / "behaviour.json"
        self.beh.write_text("{}")
        (tmp / "fake_train.py").write_text(FAKE_TRAIN)
        (tmp / "fake_eval.py").write_text(FAKE_EVAL)
        self.shutdown_marker = tmp / "SHUTDOWN_RAN"

    def behave(self, **runs):
        self.beh.write_text(json.dumps(runs))

    def config(
        self,
        groups,
        *,
        gpus=(0, 1),
        eval_gpus=(),
        watcher=False,
        eval_sleep=0.0,
        deadline=None,
        shutdown_cmd=None,
        transfer_cmd=None,
        timeout=30,
        run_timeouts=None,
        transfer_grace=1.0,
        lend_idle=False,
    ):
        run_ids = [r for g in groups for r in g["queue"]]
        runs = {}
        for rid in run_ids:
            size, reward, seed = rid.split("-")
            runs[rid] = {"size": size, "reward": reward, "seed": int(seed[1:])}
            if run_timeouts and rid in run_timeouts:
                runs[rid]["timeout_s"] = run_timeouts[rid]
        return {
            "paths": {"workdir": str(self.tmp), "runs": str(self.runs), "evals": str(self.evals)},
            "coordinator": {
                "poll_interval_s": 0.02,
                "kill_grace_s": 0.5,
                "max_retries": 1,
                "default_run_timeout_s": timeout,
                "node_hard_deadline": deadline,
                "deadline_eval_grace_s": 1.0,
                "deadline_transfer_grace_s": transfer_grace,
                "shutdown_cmd": shutdown_cmd,
            },
            "commands": {
                "train": f"{PY} {self.tmp / 'fake_train.py'} {{run_id}} {{run_dir}} {self.beh} {self.log}",
                "resume_flag": "--resume-from {resume_dir}",
                "eval": f"{PY} {self.tmp / 'fake_eval.py'} {{run_id}} {{step}} {{output}} {self.log} {eval_sleep}",
            },
            "transfer": {"command": transfer_cmd, "timeout_s": 30, "retries": 0},
            "checkpoints": {"pruning_enabled": False, "keep_last": 2},
            "eval": {
                "enabled": watcher,
                "poll_interval_s": 0.05,
                "lend_idle_gpus": lend_idle,
                "watch_scope": "all",
            },
            "sizes": {},
            "runs": runs,
            "nodes": {"n": {"gpus": list(gpus), "eval_gpus": list(eval_gpus), "groups": groups}},
        }

    def events(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def starts(self) -> list[dict]:
        return [e for e in self.events() if e["ev"] == "start"]

    def coord_events(self, node="n") -> list[dict]:
        path = self.runs / "_coordinator" / f"{node}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def status(self, rid) -> dict:
        st = read_status(run_dir(self.runs, rid))
        assert st is not None
        return st

    @property
    def idle_marker(self) -> Path:
        return self.runs / "_coordinator" / "n.idle"


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def touch_cmd(path: Path) -> str:
    return f"touch {shlex.quote(str(path))}"


def run_to_end(coord: Coordinator, timeout: float = 20.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        result = coord.tick()
        if result:
            return result
        time.sleep(0.02)
    coord._stop_watcher()
    raise AssertionError("coordinator did not finish")


def tick_until(coord: Coordinator, cond, timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        assert coord.tick() is None
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def group(name, gpus, queue, after=()):
    return {"name": name, "gpus": list(gpus), "queue": list(queue), "after": list(after)}


# ---------------------------------------------------------------- config


def test_repo_allocation_config_matches_plan():
    cfg = resolve_paths(load_config(REPO / "configs" / "allocation.yaml"), REPO)
    n1, n2 = build_plan(cfg, "node1"), build_plan(cfg, "node2")
    ids = set(n1.runs) | set(n2.runs)
    expected = {f"{size}-{r}-s{s}" for size in ("2b", "0.8b", "4b") for r in ("rob", "leak") for s in (1, 2, 3)}
    assert ids == expected and not set(n1.runs) & set(n2.runs)
    assert {r for r in n1.runs if r.startswith("4b")} == {"4b-rob-s2", "4b-leak-s2"}
    four_b = next(g for g in n1.groups if g.name == "n1-4b")
    assert four_b.queue == ("4b-rob-s2", "4b-leak-s2")  # robust then leaky, sequential
    spec = n1.runs["4b-rob-s2"]
    assert spec.gpus == (0, 1, 2, 3) and spec.vllm_server_gpus == 0  # colocate-4train
    assert n1.eval_gpus == (6, 7)
    assert n1.settings.shutdown_cmd is None  # release is opt-in
    assert n1.settings.deadline_ts == datetime(2026, 10, 16, 6, 0).timestamp()
    assert n1.settings.pruning_enabled is False


def test_config_rejects_unknown_node_and_bad_template(env):
    cfg = env.config([group("g", [0], ["2b-rob-s1"])])
    with pytest.raises(ConfigError):
        build_plan(resolve_paths(cfg), "nope")
    cfg["commands"]["train"] = "train {nonexistent}"
    with pytest.raises(ConfigError):
        build_plan(resolve_paths(cfg), "n")


# ---------------------------------------------------------------- scheduling


def test_scheduling_order_on_gpu_groups(env):
    env.behave(**{"2b-rob-s1": {"sleep": 0.3}, "2b-rob-s2": {"sleep": 0.1},
                  "2b-leak-s1": {"sleep": 0.7}, "4b-rob-s1": {}})  # fmt: skip
    cfg = env.config(
        [
            group("g0", [0], ["2b-rob-s1", "2b-rob-s2"]),
            group("g1", [1], ["2b-leak-s1"]),
            group("big", [0, 1], ["4b-rob-s1"], after=["g0", "g1"]),
        ]
    )
    result = run_to_end(Coordinator(cfg, "n"))

    starts = {e["run"]: e for e in env.starts()}
    ends = {e["run"]: e for e in env.events() if e["ev"] == "end"}
    assert result == "idle"
    assert {starts["2b-rob-s1"]["gpus"], starts["2b-leak-s1"]["gpus"]} == {"0", "1"}
    assert starts["2b-rob-s2"]["t"] >= ends["2b-rob-s1"]["t"]  # queue order on g0
    assert starts["4b-rob-s1"]["t"] >= max(ends["2b-rob-s2"]["t"], ends["2b-leak-s1"]["t"])
    assert starts["4b-rob-s1"]["gpus"] == "0,1"
    for rid in starts:
        st = env.status(rid)
        assert st["status"] == "finished" and st["exit_code"] == 0 and st["attempts"] == 1


def test_timeout_kills_process_tree_and_node_stays_up(env):
    child_pid_file = env.tmp / "child.pid"
    env.behave(**{"2b-rob-s1": {"sleep": 30, "child_pid_file": str(child_pid_file)}})
    cfg = env.config([group("g0", [0], ["2b-rob-s1", "2b-rob-s2"])], run_timeouts={"2b-rob-s1": 0.5})
    result = run_to_end(Coordinator(cfg, "n"))

    st = env.status("2b-rob-s1")
    assert st["status"] == "timeout" and st["reason"] == "run_timeout"
    child = int(child_pid_file.read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("grandchild survived the timeout")
    assert env.status("2b-rob-s2")["status"] == "finished"  # node kept going
    assert result == "idle"


def test_crash_resumes_from_latest_recovery_checkpoint(env):
    env.behave(
        **{
            "2b-rob-s1": {"crash_first": True, "recovery": [10, 20]},
            "2b-leak-s1": {"exit": 4},
        }
    )
    cfg = env.config([group("g0", [0], ["2b-rob-s1"]), group("g1", [1], ["2b-leak-s1"])])
    run_to_end(Coordinator(cfg, "n"))

    rob = [e for e in env.starts() if e["run"] == "2b-rob-s1"]
    assert rob[0]["resume"] is None
    assert rob[1]["resume"].endswith("recovery/checkpoint-20")
    st = env.status("2b-rob-s1")
    assert st["status"] == "finished" and st["attempts"] == 2
    leak = env.status("2b-leak-s1")
    assert leak["status"] == "failed" and "no recovery checkpoint" in leak["reason"]


# ---------------------------------------------------------------- gate + release


def test_gate_blocks_until_eval_and_transfer_then_opt_in_shutdown(env):
    release = env.tmp / "release"
    sync = f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.05; done"
    env.behave(**{"2b-rob-s1": {"evals": [50, 60]}})
    cfg = env.config(
        [group("g0", [0], ["2b-rob-s1"])],
        transfer_cmd=sync,
        shutdown_cmd=touch_cmd(env.shutdown_marker),
    )
    coord = Coordinator(cfg, "n", allow_shutdown=True)

    tick_until(coord, lambda: coord.runs["2b-rob-s1"].terminal)
    for _ in range(5):
        assert coord.tick() is None
    blockers = coord.gate_blockers()
    assert any("eval(s) pending" in b for b in blockers)
    assert not coord.transfers  # transfer waits for evals
    assert not env.idle_marker.exists() and not env.shutdown_marker.exists()

    for step in (50, 60):
        p = eval_done_path(env.evals, "2b-rob-s1", step)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    tick_until(coord, lambda: bool(coord.transfers))
    for _ in range(5):
        assert coord.tick() is None
    assert coord.gate_blockers() == ["2b-rob-s1: not transferred"]
    assert not env.idle_marker.exists() and not env.shutdown_marker.exists()

    release.write_text("")
    assert run_to_end(coord) == "shutdown"
    assert env.shutdown_marker.exists()
    assert env.idle_marker.exists()
    rec = json.loads(transferred_path(run_dir(env.runs, "2b-rob-s1")).read_text())
    assert rec["eval_steps_done"] == [50, 60]


def test_default_mode_writes_idle_marker_and_never_shuts_down(env):
    env.behave(**{"2b-rob-s1": {}})
    cfg = env.config([group("g0", [0], ["2b-rob-s1"])], shutdown_cmd=touch_cmd(env.shutdown_marker))
    assert run_to_end(Coordinator(cfg, "n")) == "idle"
    assert json.loads(env.idle_marker.read_text())["message"] == "node idle, safe to release"
    assert "node_idle" in [e["event"] for e in env.coord_events()]
    assert not env.shutdown_marker.exists()


def test_allow_shutdown_without_command_never_shuts_down(env):
    env.behave(**{"2b-rob-s1": {}})
    cfg = env.config([group("g0", [0], ["2b-rob-s1"])], shutdown_cmd=None)
    coord = Coordinator(cfg, "n", allow_shutdown=True)
    assert coord.shutdown_opt_in is False
    assert run_to_end(coord) == "idle"


def _deadline_clock(seconds_before: float):
    """Clock pinned to the real config deadline (Fri 16 Oct 06:00), starting just before it."""
    deadline = datetime(2026, 10, 16, 6, 0).timestamp()
    shift = deadline - seconds_before - time.time()
    return (lambda: time.time() + shift), "2026-10-16T06:00:00"


def test_default_mode_never_shuts_down_even_at_deadline(env):
    clock, deadline = _deadline_clock(0.5)
    env.behave(**{"2b-rob-s1": {"sleep": 30}})
    cfg = env.config(
        [group("g0", [0], ["2b-rob-s1", "2b-rob-s2"])],
        deadline=deadline,
        shutdown_cmd=touch_cmd(env.shutdown_marker),
    )
    result = run_to_end(Coordinator(cfg, "n", clock=clock))

    assert result == "idle"
    assert not env.shutdown_marker.exists()
    st = env.status("2b-rob-s1")
    assert st["status"] == "timeout" and st["reason"] == "node_deadline"
    skipped = env.status("2b-rob-s2")
    assert skipped["status"] == "timeout" and skipped["reason"] == "not_started_before_deadline"
    assert [e["run"] for e in env.starts()] == ["2b-rob-s1"]
    events = [e["event"] for e in env.coord_events()]
    assert "deadline_reached" in events and "shutdown_requested" not in events


def test_opt_in_shuts_down_at_deadline_after_grace(env):
    clock, deadline = _deadline_clock(0.5)
    env.behave(**{"2b-rob-s1": {"sleep": 30}})
    stuck = "sleep 60"
    cfg = env.config(
        [group("g0", [0], ["2b-rob-s1"])],
        deadline=deadline,
        shutdown_cmd=touch_cmd(env.shutdown_marker),
        transfer_cmd=stuck,
        transfer_grace=1.0,
    )
    coord = Coordinator(cfg, "n", clock=clock, allow_shutdown=True)
    assert run_to_end(coord) == "shutdown"
    assert env.shutdown_marker.exists()
    assert not env.idle_marker.exists()  # released by the deadline, not the gate
    events = [e["event"] for e in env.coord_events()]
    assert events.index("deadline_reached") < events.index("shutdown_executed")
    assert "transfer_abandoned" in events
    assert env.status("2b-rob-s1")["reason"] == "node_deadline"


def test_dry_run_never_executes_shutdown(env):
    env.behave(**{"2b-rob-s1": {}})
    cfg = env.config([group("g0", [0], ["2b-rob-s1"])], shutdown_cmd=touch_cmd(env.shutdown_marker))
    coord = Coordinator(cfg, "n", dry_run=True, allow_shutdown=True)
    assert coord._shutdown_cmd is None
    assert run_to_end(coord) == "shutdown-skipped"
    assert not env.shutdown_marker.exists()
    with pytest.raises(RuntimeError):
        coord._execute_shutdown(touch_cmd(env.shutdown_marker))
    assert not env.shutdown_marker.exists()


# ---------------------------------------------------------------- restart


def test_restart_is_idempotent(env):
    host = socket.gethostname()
    groups = [
        group("g0", [0], ["2b-rob-s1"]),
        group("g1", [1], ["2b-rob-s2"]),
        group("g2", [2], ["2b-rob-s3"]),
        group("g3", [3], ["2b-leak-s1"]),
    ]
    cfg = env.config(groups, gpus=(0, 1, 2, 3))
    env.behave()

    def write(rid, **status):
        d = run_dir(env.runs, rid)
        (d / "logs").mkdir(parents=True, exist_ok=True)
        (d / "status.json").write_text(json.dumps({"run_id": rid, "attempts": 1, **status}))
        return d

    # finished before the restart
    write("2b-rob-s1", status="finished", exit_code=0, started_ts=time.time())
    # was running; exited 0 while the coordinator was down
    d = write("2b-rob-s2", status="running", pid=999999, host=host, started_ts=time.time())
    (d / "logs" / "attempt-1.exit").write_text("0")
    # still running: must be adopted, not relaunched
    d = write("2b-rob-s3", status="running", host=host, started_ts=time.time())
    exit_file = d / "logs" / "attempt-1.exit"
    # Background it so it is not our child (no zombie), like a run left by a dead coordinator.
    live_pid = int(
        subprocess.run(
            [
                "/bin/sh",
                "-c",
                f"(sleep 0.7; printf 0 > {shlex.quote(str(exit_file))}) >/dev/null 2>&1 & echo $!",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    status = json.loads((d / "status.json").read_text())
    status["pid"] = live_pid
    (d / "status.json").write_text(json.dumps(status))

    assert run_to_end(Coordinator(cfg, "n")) == "idle"

    assert [e["run"] for e in env.starts()] == ["2b-leak-s1"]  # only the never-started run
    for rid in ("2b-rob-s1", "2b-rob-s2", "2b-rob-s3", "2b-leak-s1"):
        assert env.status(rid)["status"] == "finished", rid
    events = [e["event"] for e in env.coord_events()]
    assert "adopt" in events and "reconcile" in events

    # A second restart relaunches nothing and goes straight to idle.
    assert run_to_end(Coordinator(cfg, "n")) == "idle"
    assert len(env.starts()) == 1


# ---------------------------------------------------------------- watcher integration


def test_end_to_end_with_watcher_reclaims_eval_gpu_before_launch(env):
    env.behave(**{"2b-rob-s1": {"evals": [50, 100, 120], "post_sleep": 0.5}, "4b-rob-s1": {}})
    cfg = env.config(
        [
            group("g0", [0], ["2b-rob-s1"]),
            group("big", [0, 1], ["4b-rob-s1"], after=["g0"]),
        ],
        eval_gpus=(1,),
        watcher=True,
        eval_sleep=0.4,
    )
    result = run_to_end(Coordinator(cfg, "n"), timeout=30)

    assert result == "idle"
    evals = [e for e in env.events() if e["ev"] == "eval"]
    assert {(e["run"], e["step"]) for e in evals} == {("2b-rob-s1", s) for s in (50, 100, 120)}
    big = next(e for e in env.starts() if e["run"] == "4b-rob-s1")
    on_gpu1 = [e for e in evals if e["gpus"] == "1" and e["t0"] < big["t"]]
    assert on_gpu1, "expected an eval in flight on GPU 1 while the 2B run was still going"
    assert all(e["t1"] <= big["t"] for e in on_gpu1)  # GPU 1 freed by eval before training
    for s in (50, 100, 120):
        assert eval_done_path(env.evals, "2b-rob-s1", s).exists(), step_name(s)
    assert env.idle_marker.exists()


def test_cli_print_plan(tmp_path):
    out = subprocess.run(
        [PY, str(REPO / "train" / "coordinator.py"), "--node", "node1", "--print-plan"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "4b-rob-s2" in out and "--allow-shutdown" in out
    assert yaml.safe_load((REPO / "configs" / "allocation.yaml").read_text())["version"] == 1
