"""Per-node run coordinator (PROGRESSION.md §1.4, §3.1-3.3, P4).

Usage::

    python train/coordinator.py --node node1 --config configs/allocation.yaml
        [--dry-run] [--allow-shutdown] [--print-plan] [--poll-interval S] [--skip-preflight]

Each node runs its own coordinator. It:

* **Runs the node preflight first** (`train/preflight.py`, also ``make preflight``): GPU stack imports,
  ``ninja``, GPU count, data against its manifest, model exports, disk. On any failure it launches
  nothing and exits 1; ``--skip-preflight`` overrides. ``--print-plan`` never runs it.

* **Launches runs** from the node's GPU groups (`configs/allocation.yaml`).
  A group's queue runs in order. The head run starts when every group in its
  ``after`` list is drained and all its GPUs are free (no training run, no
  eval). Each run gets ``CUDA_VISIBLE_DEVICES`` and its own process group, so a
  timeout kills the whole tree.
* **Enforces per-run timeouts.** A timed-out run ends with status ``timeout``.
  The node stays up.
* **Recovers crashes.** A run that exits non-zero is relaunched with the
  resume flag pointing at its newest complete recovery checkpoint, up to
  ``coordinator.max_retries`` times (default 1). With no recovery checkpoint
  it is marked ``failed``.
* **Records status** in ``<runs>/<run_id>/status.json`` and appends events to
  ``<runs>/_coordinator/<node>.jsonl``.
* **Supervises the eval watcher** (`eval/watcher.py`). It leases eval GPUs to
  the watcher through ``<runs>/_coordinator/<node>.eval_gpus.json``: the
  node's ``eval_gpus`` minus GPUs that training needs now, plus (with
  ``eval.lend_idle_gpus``) any GPU no remaining run will use. That is the
  "eval falls behind" fallback.
* **Transfers outputs.** Once a run has ended and all its eval checkpoints are
  evaluated, it optionally prunes recovery checkpoints (`train/checkpoints.py`)
  and runs ``transfer.command``. Success writes ``<runs>/<run_id>/TRANSFERRED``.
* **Restarts idempotently.** On start it re-reads every ``status.json``.
  Finished, failed and timed-out runs are never relaunched. A ``running`` run
  whose process is still alive is adopted. One that died while the
  coordinator was down is settled from its exit-code file.

Node release (owner requirement: Studio machines stay attached by default):

* **Gate.** All of the node's runs are finished, failed or timed out, AND
  every ``COMPLETE`` eval checkpoint of those runs has a ``.done`` eval, AND
  every run is ``TRANSFERRED``.
* **Default: report only.** When the gate passes, the coordinator writes
  ``<runs>/_coordinator/<node>.idle`` ("node idle, safe to release"), logs a
  ``node_idle`` event, drains the watcher and exits. It never runs a command.
* **Shutdown is opt-in.** A shutdown command runs only with ``--allow-shutdown``
  AND a configured ``coordinator.shutdown_cmd``. Without both, never. With
  ``--dry-run`` it is never executed, even when both are set.
* **Hard node deadline** (``coordinator.node_hard_deadline``). This is the
  training cutoff. Running training runs are killed (status ``timeout``) and
  queued runs never start. By default the node is NOT released: eval and
  transfers continue until the gate passes, as above. With opt-in shutdown,
  the coordinator waits up to ``deadline_eval_grace_s`` for in-flight evals,
  then up to ``deadline_transfer_grace_s`` for transfers, then shuts down
  whether or not the gate passed.
* Individual runs and the watcher never stop the machine.

Artifact layout: see `train/checkpoints.py` and `train/LAYOUT.md`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `train` / `eval` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib

import yaml

from train.checkpoints import (
    DEFAULT_REQUIRED_EVAL_FILES,
    TERMINAL_STATUSES,
    atomic_write_json,
    coordinator_dir,
    eval_done_path,
    eval_failed_path,
    latest_recovery_checkpoint,
    list_eval_checkpoints,
    prune_recovery_checkpoints,
    read_json,
    read_status,
    run_dir,
    transferred_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- config


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_paths(cfg: dict, base: Path | None = None) -> dict:
    """Copy of ``cfg`` with ``paths.workdir/runs/evals`` made absolute."""
    cfg = copy.deepcopy(cfg)
    base = Path(base or Path.cwd()).resolve()
    paths = cfg.setdefault("paths", {})
    workdir = Path(paths.get("workdir", "."))
    workdir = workdir if workdir.is_absolute() else (base / workdir).resolve()
    paths["workdir"] = str(workdir)
    for key, default in (("runs", "runs"), ("evals", "evals")):
        p = Path(paths.get(key, default))
        paths[key] = str(p if p.is_absolute() else (workdir / p).resolve())
    return cfg


def parse_deadline(value: Any) -> float | None:
    """ISO datetime -> epoch seconds. Naive values are machine-local time."""
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    return datetime.fromisoformat(str(value)).timestamp()


def render(template: str, values: dict[str, Any]) -> str:
    """Fill ``{placeholders}``; values are shell-quoted."""
    return template.format_map({k: shlex.quote(str(v)) for k, v in values.items()})


def _csv(gpus: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(g) for g in gpus)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    size: str
    reward: str
    seed: int
    gpus: tuple[int, ...]
    group: str
    timeout_s: float
    train_command: str
    vllm_server_gpus: int = 0


@dataclass(frozen=True)
class Group:
    name: str
    gpus: tuple[int, ...]
    queue: tuple[str, ...]
    after: tuple[str, ...] = ()


@dataclass
class Settings:
    workdir: Path
    runs_root: Path
    evals_root: Path
    poll_interval_s: float = 15.0
    kill_grace_s: float = 60.0
    max_retries: int = 1
    deadline_ts: float | None = None
    deadline_eval_grace_s: float = 1800.0
    deadline_transfer_grace_s: float = 1800.0
    shutdown_cmd: str | None = None
    resume_flag: str = "--resume-from {resume_dir}"
    transfer_cmd: str | None = None
    transfer_timeout_s: float = 3600.0
    transfer_retries: int = 2
    transfer_max_concurrent: int = 2
    pruning_enabled: bool = False
    keep_last: int = 2
    required_eval_files: tuple[str, ...] = DEFAULT_REQUIRED_EVAL_FILES
    watcher_enabled: bool = True
    watcher_poll_interval_s: float = 30.0
    watcher_max_restarts: int = 5
    lend_idle_gpus: bool = True


@dataclass
class NodePlan:
    name: str
    gpus: tuple[int, ...]
    eval_gpus: tuple[int, ...]
    groups: list[Group]
    runs: dict[str, RunSpec]
    settings: Settings


def train_values(spec: RunSpec, runs_root: Path, resume_dir: str | Path = "") -> dict[str, Any]:
    v = spec.vllm_server_gpus
    return {
        "run_id": spec.run_id,
        "size": spec.size,
        "reward": spec.reward,
        "seed": spec.seed,
        "gpus": _csv(spec.gpus),
        "num_gpus": len(spec.gpus),
        "vllm_gpus": _csv(spec.gpus[:v]),
        "train_gpus": _csv(spec.gpus[v:]),
        "num_vllm_gpus": v,
        "num_train_gpus": len(spec.gpus) - v,
        "run_dir": run_dir(runs_root, spec.run_id),
        "resume_dir": resume_dir,
    }


def train_command(plan: NodePlan, spec: RunSpec, resume_dir: Path | None = None) -> str:
    values = train_values(spec, plan.settings.runs_root, resume_dir or "")
    cmd = render(spec.train_command, values)
    if resume_dir is not None:
        cmd = f"{cmd} {render(plan.settings.resume_flag, values)}"
    return cmd


def build_plan(cfg: dict, node: str) -> NodePlan:
    """Validate ``cfg`` (already path-resolved) and build ``node``'s plan."""
    nodes = cfg.get("nodes") or {}
    if node not in nodes:
        raise ConfigError(f"unknown node {node!r}; known: {sorted(nodes)}")
    runs_cfg = cfg.get("runs") or {}
    sizes = cfg.get("sizes") or {}
    commands = cfg.get("commands") or {}
    co = cfg.get("coordinator") or {}
    tr = cfg.get("transfer") or {}
    ck = cfg.get("checkpoints") or {}
    ev = cfg.get("eval") or {}
    paths = cfg["paths"]

    # A run ID belongs to exactly one group on exactly one node.
    owner: dict[str, str] = {}
    for nname, ncfg in nodes.items():
        for g in ncfg.get("groups", []):
            for rid in g.get("queue", []):
                if rid in owner:
                    raise ConfigError(f"run {rid} queued twice ({owner[rid]}, {nname}/{g['name']})")
                if rid not in runs_cfg:
                    raise ConfigError(f"run {rid} in {nname}/{g['name']} is not in `runs`")
                owner[rid] = f"{nname}/{g['name']}"

    ncfg = nodes[node]
    node_gpus = tuple(int(g) for g in ncfg.get("gpus", range(8)))
    eval_gpus = tuple(int(g) for g in ncfg.get("eval_gpus", []))
    if not set(eval_gpus) <= set(node_gpus):
        raise ConfigError(f"{node}: eval_gpus {eval_gpus} not within node gpus {node_gpus}")

    settings = Settings(
        workdir=Path(paths["workdir"]),
        runs_root=Path(paths["runs"]),
        evals_root=Path(paths["evals"]),
        poll_interval_s=float(co.get("poll_interval_s", 15)),
        kill_grace_s=float(co.get("kill_grace_s", 60)),
        max_retries=int(co.get("max_retries", 1)),
        deadline_ts=parse_deadline(co.get("node_hard_deadline")),
        deadline_eval_grace_s=float(co.get("deadline_eval_grace_s", 1800)),
        deadline_transfer_grace_s=float(co.get("deadline_transfer_grace_s", 1800)),
        shutdown_cmd=co.get("shutdown_cmd") or None,
        resume_flag=commands.get("resume_flag", "--resume-from {resume_dir}"),
        transfer_cmd=tr.get("command") or None,
        transfer_timeout_s=float(tr.get("timeout_s", 3600)),
        transfer_retries=int(tr.get("retries", 2)),
        transfer_max_concurrent=int(tr.get("max_concurrent", 2)),
        pruning_enabled=bool(ck.get("pruning_enabled", False)),
        keep_last=int(ck.get("keep_last", 2)),
        required_eval_files=tuple(ck.get("required_eval_files", DEFAULT_REQUIRED_EVAL_FILES)),
        watcher_enabled=bool(ev.get("enabled", True)),
        watcher_poll_interval_s=float(ev.get("poll_interval_s", 30)),
        watcher_max_restarts=int(ev.get("max_restarts", 5)),
        lend_idle_gpus=bool(ev.get("lend_idle_gpus", True)),
    )

    groups: list[Group] = []
    runs: dict[str, RunSpec] = {}
    names = {g["name"] for g in ncfg.get("groups", [])}
    for g in ncfg.get("groups", []):
        gpus = tuple(int(x) for x in g["gpus"])
        if not set(gpus) <= set(node_gpus):
            raise ConfigError(f"{node}/{g['name']}: gpus {gpus} not within node gpus")
        after = tuple(g.get("after", []))
        if not set(after) <= names:
            raise ConfigError(f"{node}/{g['name']}: unknown groups in after: {after}")
        group = Group(g["name"], gpus, tuple(g.get("queue", [])), after)
        groups.append(group)
        for rid in group.queue:
            rc = runs_cfg[rid]
            size = str(rc["size"])
            sc = sizes.get(size, {})
            want = sc.get("gpus_per_run")
            if want is not None and int(want) != len(gpus):
                raise ConfigError(f"{rid}: size {size} needs {want} GPUs, group has {len(gpus)}")
            vllm = int((sc.get("layout") or {}).get("vllm_server_gpus", 0))
            if vllm >= len(gpus) and vllm > 0:
                raise ConfigError(f"{rid}: vllm_server_gpus={vllm} leaves no training GPU")
            timeout = rc.get("timeout_s", sc.get("run_timeout_s", co.get("default_run_timeout_s")))
            if timeout is None:
                raise ConfigError(f"{rid}: no timeout (runs/sizes/coordinator.default_run_timeout_s)")
            template = rc.get("train_command") or sc.get("train_command") or commands.get("train")
            if not template:
                raise ConfigError(f"{rid}: no train command template")
            runs[rid] = RunSpec(
                run_id=rid,
                size=size,
                reward=str(rc["reward"]),
                seed=int(rc["seed"]),
                gpus=gpus,
                group=group.name,
                timeout_s=float(timeout),
                train_command=template,
                vllm_server_gpus=vllm,
            )
    plan = NodePlan(node, node_gpus, eval_gpus, groups, runs, settings)
    for spec in runs.values():  # fail fast on bad placeholders
        try:
            train_command(plan, spec, Path("/resume"))
        except (KeyError, IndexError, ValueError) as exc:
            raise ConfigError(f"{spec.run_id}: bad train/resume template: {exc!r}") from exc
    return plan


def describe_plan(plan: NodePlan) -> str:
    s = plan.settings
    lines = [f"node {plan.name}: gpus {list(plan.gpus)}, eval gpus {list(plan.eval_gpus)}"]
    if s.deadline_ts is not None:
        lines.append(f"hard deadline: {datetime.fromtimestamp(s.deadline_ts).astimezone()}")
    lines.append(f"shutdown_cmd: {s.shutdown_cmd!r} (runs only with --allow-shutdown)")
    for g in plan.groups:
        after = f" after {list(g.after)}" if g.after else ""
        lines.append(f"  group {g.name} gpus {list(g.gpus)}{after}")
        for rid in g.queue:
            spec = plan.runs[rid]
            lines.append(f"    {rid} (timeout {spec.timeout_s:.0f}s): {train_command(plan, spec)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- process helpers


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def killpg(pid: int | None, sig: int) -> None:
    if not pid:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- state


@dataclass
class _Run:
    spec: RunSpec
    status: dict | None = None
    proc: subprocess.Popen | None = None
    pid: int | None = None
    log: Any = None
    term_sent: float | None = None
    kill_reason: str | None = None

    @property
    def state(self) -> str:
        return (self.status or {}).get("status", "pending")

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATUSES

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def attempts(self) -> int:
        return int((self.status or {}).get("attempts", 0))


@dataclass
class _Transfer:
    run_id: str
    proc: subprocess.Popen
    started: float
    steps: list[int]
    log: Any
    term_sent: float | None = None


@dataclass
class _Watcher:
    proc: subprocess.Popen | None = None
    restarts: int = 0
    given_up: bool = False
    log: Any = None


# ---------------------------------------------------------------- coordinator


class Coordinator:
    def __init__(
        self,
        config: dict,
        node: str,
        *,
        dry_run: bool = False,
        allow_shutdown: bool = False,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.cfg = resolve_paths(config, base_dir)
        self.plan = build_plan(self.cfg, node)
        self.s = self.plan.settings
        self.node = node
        self.dry_run = dry_run
        self.clock = clock
        self.sleep = sleep
        self.poll_interval = self.s.poll_interval_s if poll_interval is None else poll_interval
        # Shutdown needs BOTH the flag and a configured command. In dry-run the
        # command is never even kept, so there is nothing to execute.
        self.shutdown_opt_in = bool(allow_shutdown and self.s.shutdown_cmd)
        self._shutdown_cmd: str | None = None if dry_run else self.s.shutdown_cmd
        self.host = socket.gethostname()

        self.cdir = coordinator_dir(self.s.runs_root)
        self.event_log = self.cdir / f"{node}.jsonl"
        self.idle_marker = self.cdir / f"{node}.idle"
        self.lease_file = self.cdir / f"{node}.eval_gpus.json"
        self.watcher_state_file = self.cdir / f"{node}.watcher.json"
        self.config_snapshot = self.cdir / f"{node}.config.json"

        self.runs: dict[str, _Run] = {rid: _Run(spec) for rid, spec in self.plan.runs.items()}
        self.transfers: dict[str, _Transfer] = {}
        self.transfer_failures: dict[str, int] = {}
        self.pruned: set[str] = set()
        self.watcher = _Watcher()
        self.lease_gen = 0
        self.lease: tuple[tuple[int, ...], bool] | None = None
        self.phase = "normal"  # normal | draining | deadline_evals | deadline_transfers
        self.phase_deadline = 0.0
        self.deadline_hit = False
        self.result: str | None = None
        self._last_blockers: list[str] | None = None
        self._started = False

    # ---- events / status

    def event(self, name: str, **fields: Any) -> None:
        now = self.clock()
        rec = {"ts": round(now, 3), "iso": _iso(now), "node": self.node, "event": name, **fields}
        line = json.dumps(rec, default=str)
        self.cdir.mkdir(parents=True, exist_ok=True)
        with open(self.event_log, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def _write_status(self, run: _Run, **updates: Any) -> None:
        now = self.clock()
        status = dict(run.status or {})
        status.update(updates)
        status["updated_at"] = _iso(now)
        run.status = status
        atomic_write_json(run_dir(self.s.runs_root, run.spec.run_id) / "status.json", status)

    def _exit_file(self, run: _Run, attempt: int) -> Path:
        return run_dir(self.s.runs_root, run.spec.run_id) / "logs" / f"attempt-{attempt}.exit"

    # ---- startup

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.cdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.config_snapshot, self.cfg)
        if self.idle_marker.exists():
            self.idle_marker.unlink()  # stale "safe to release" must not survive a restart
            self.event("idle_cleared")
        lease = read_json(self.lease_file, default={}) or {}
        self.lease_gen = int(lease.get("gen", 0))
        self.event(
            "coordinator_start",
            pid=os.getpid(),
            dry_run=self.dry_run,
            shutdown_opt_in=self.shutdown_opt_in,
            deadline=_iso(self.s.deadline_ts) if self.s.deadline_ts else None,
        )
        for run in self.runs.values():
            run.status = read_status(run_dir(self.s.runs_root, run.spec.run_id))
            if not run.running:
                continue
            status = run.status or {}
            pid = int(status.get("pid") or 0)
            if status.get("host") == self.host and pid_alive(pid):
                run.pid = pid
                self.event("adopt", run_id=run.spec.run_id, pid=pid)
            else:
                rc = self._read_exit(run)
                self.event("reconcile", run_id=run.spec.run_id, exit_code=rc)
                self._on_exit(run, rc)
        self._kill_stale_watcher()
        if self.s.watcher_enabled:
            self._update_lease()
            self._start_watcher()

    def _read_exit(self, run: _Run) -> int | None:
        try:
            return int(self._exit_file(run, run.attempts).read_text().strip())
        except (OSError, ValueError):
            return None

    # ---- runs

    def _launch(self, run: _Run, resume: Path | None = None) -> None:
        spec = run.spec
        rdir = run_dir(self.s.runs_root, spec.run_id)
        (rdir / "logs").mkdir(parents=True, exist_ok=True)
        attempt = run.attempts + 1
        cmd = train_command(self.plan, spec, resume)
        exit_file = self._exit_file(run, attempt)
        script = f"{cmd}\n__rc=$?\nprintf '%s' \"$__rc\" > {shlex.quote(str(exit_file))}\nexit $__rc\n"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=_csv(spec.gpus),
            ENVLAB_RUN_ID=spec.run_id,
            ENVLAB_RUN_DIR=str(rdir),
        )
        run.log = open(rdir / "logs" / f"attempt-{attempt}.log", "ab")  # noqa: SIM115 - handed to a child process; closed when it exits
        run.proc = subprocess.Popen(
            ["/bin/sh", "-c", script],
            cwd=self.s.workdir,
            env=env,
            stdout=run.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        run.pid = run.proc.pid
        run.term_sent = None
        run.kill_reason = None
        now = self.clock()
        first = run.status or {}
        self._write_status(
            run,
            run_id=spec.run_id,
            node=self.node,
            size=spec.size,
            reward=spec.reward,
            seed=spec.seed,
            gpus=list(spec.gpus),
            group=spec.group,
            status="running",
            started_at=first.get("started_at") or _iso(now),
            started_ts=first.get("started_ts") or now,
            ended_at=None,
            exit_code=None,
            reason=None,
            attempts=attempt,
            pid=run.pid,
            host=self.host,
            resume_from=str(resume) if resume else None,
            command=cmd,
        )
        self.event(
            "relaunch" if resume else "launch",
            run_id=spec.run_id,
            gpus=list(spec.gpus),
            attempt=attempt,
            pid=run.pid,
            resume_from=str(resume) if resume else None,
        )

    def _poll(self, run: _Run) -> int | None:
        if run.proc is not None:
            return run.proc.poll()
        if pid_alive(run.pid):  # adopted from a previous coordinator
            return None
        rc = self._read_exit(run)
        return -1 if rc is None else rc

    def _kill(self, run: _Run, reason: str, now: float) -> None:
        run.kill_reason = reason
        run.term_sent = now
        killpg(run.pid, signal.SIGTERM)
        self.event("kill", run_id=run.spec.run_id, reason=reason, pid=run.pid)

    def _end(self, run: _Run, status: str, rc: int | None, reason: str | None) -> None:
        if run.log is not None:
            run.log.close()
            run.log = None
        run.proc = None
        run.pid = None
        self._write_status(run, status=status, exit_code=rc, reason=reason, ended_at=_iso(self.clock()))
        self.event("run_end", run_id=run.spec.run_id, status=status, exit_code=rc, reason=reason)

    def _on_exit(self, run: _Run, rc: int | None) -> None:
        if run.kill_reason:
            self._end(run, "timeout", rc, run.kill_reason)
        elif rc == 0:
            self._end(run, "finished", rc, None)
        else:
            rdir = run_dir(self.s.runs_root, run.spec.run_id)
            latest = latest_recovery_checkpoint(rdir)
            retries_used = max(0, run.attempts - 1)
            if latest and retries_used < self.s.max_retries and not self._past_deadline():
                if run.log is not None:
                    run.log.close()
                    run.log = None
                self.event("crash", run_id=run.spec.run_id, exit_code=rc, resume=str(latest.path))
                self._launch(run, resume=latest.path)
            else:
                why = "no recovery checkpoint" if not latest else "retries exhausted"
                if latest and self._past_deadline():
                    why = "past node deadline"
                self._end(run, "failed", rc, f"exit {rc}; {why}")

    def _supervise_runs(self, now: float) -> None:
        for run in self.runs.values():
            if not run.running:
                continue
            rc = self._poll(run)
            if rc is None:
                if run.term_sent is not None:
                    if now - run.term_sent >= self.s.kill_grace_s:
                        killpg(run.pid, signal.SIGKILL)
                elif self._past_deadline(now):
                    self._kill(run, "node_deadline", now)
                elif now >= float((run.status or {}).get("started_ts", now)) + run.spec.timeout_s:
                    self._kill(run, "run_timeout", now)
                continue
            self._on_exit(run, rc)

    def _past_deadline(self, now: float | None = None) -> bool:
        d = self.s.deadline_ts
        return d is not None and (self.clock() if now is None else now) >= d

    def _group_drained(self, name: str) -> bool:
        group = next(g for g in self.plan.groups if g.name == name)
        return all(self.runs[r].terminal for r in group.queue)

    def ready_runs(self) -> list[_Run]:
        """Head-of-queue runs whose ``after`` groups are drained (not yet running)."""
        out = []
        for g in self.plan.groups:
            if not all(self._group_drained(a) for a in g.after):
                continue
            head = next((self.runs[r] for r in g.queue if not self.runs[r].terminal), None)
            if head is not None and not head.running:
                out.append(head)
        return out

    def _launch_ready(self) -> None:
        used = {g for r in self.runs.values() if r.running for g in r.spec.gpus}
        leased = set(self.lease[0]) if self.lease else set()
        for run in self.ready_runs():
            gpus = set(run.spec.gpus)
            if gpus & used or gpus & leased or not self._watcher_released(gpus):
                continue
            self._launch(run)
            used |= gpus

    # ---- eval GPU lease + watcher

    def _desired_lease(self) -> tuple[tuple[int, ...], bool]:
        if self.phase != "normal":
            return (), True
        running = {g for r in self.runs.values() if r.running for g in r.spec.gpus}
        ready = {g for r in self.ready_runs() for g in r.spec.gpus}
        pending = {g for r in self.runs.values() if not r.terminal for g in r.spec.gpus}
        lease = set(self.plan.eval_gpus) - running - ready
        if self.s.lend_idle_gpus:
            lease |= set(self.plan.gpus) - pending
        return tuple(sorted(lease)), False

    def _update_lease(self) -> None:
        if not self.s.watcher_enabled:
            return
        desired = self._desired_lease()
        if desired == self.lease:
            return
        self.lease_gen += 1
        self.lease = desired
        atomic_write_json(
            self.lease_file,
            {"gen": self.lease_gen, "gpus": list(desired[0]), "drain": desired[1]},
        )
        self.event("eval_lease", gen=self.lease_gen, gpus=list(desired[0]), drain=desired[1])

    def _watcher_alive(self) -> bool:
        return self.watcher.proc is not None and self.watcher.proc.poll() is None

    def _watcher_released(self, gpus: set[int]) -> bool:
        """True once the watcher acked the current lease and holds none of ``gpus``."""
        if not self.s.watcher_enabled or self.watcher.given_up:
            return True
        if not self._watcher_alive():
            return False
        proc = self.watcher.proc
        if proc is None:
            return False
        st = read_json(self.watcher_state_file, default={}) or {}
        if st.get("pid") != proc.pid or int(st.get("ack_gen", -1)) < self.lease_gen:
            return False
        busy = {int(g) for g in (st.get("busy") or {})}
        return not (busy & gpus)

    def _kill_stale_watcher(self) -> None:
        st = read_json(self.watcher_state_file, default={}) or {}
        pid = int(st.get("pid") or 0)
        if st.get("host") == self.host and pid_alive(pid) and pid != os.getpid():
            self.event("watcher_stale_kill", pid=pid)
            killpg(pid, signal.SIGTERM)
            end = time.monotonic() + self.s.kill_grace_s
            while pid_alive(pid) and time.monotonic() < end:
                time.sleep(0.05)
            killpg(pid, signal.SIGKILL)

    def _start_watcher(self) -> None:
        if self.watcher.log is None:
            self.watcher.log = open(self.cdir / f"{self.node}.watcher.log", "ab")  # noqa: SIM115 - handed to a child process; closed when it exits
        cmd = [
            sys.executable,
            "-m",
            "eval.watcher",
            "--config",
            str(self.config_snapshot),
            "--node",
            self.node,
            "--lease-file",
            str(self.lease_file),
            "--state-file",
            str(self.watcher_state_file),
            "--poll-interval",
            str(self.s.watcher_poll_interval_s),
        ]
        pythonpath = os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)
        self.watcher.proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=dict(os.environ, PYTHONPATH=pythonpath),
            stdout=self.watcher.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.event("watcher_start", pid=self.watcher.proc.pid, restarts=self.watcher.restarts)

    def _supervise_watcher(self) -> None:
        w = self.watcher
        if not self.s.watcher_enabled or w.given_up or self._watcher_alive():
            return
        if w.proc is not None:
            if self.lease and self.lease[1]:
                return  # drained on request: expected exit
            w.restarts += 1
            self.event("watcher_died", exit_code=w.proc.returncode, restarts=w.restarts)
            w.proc = None
            if w.restarts > self.s.watcher_max_restarts:
                w.given_up = True
                self.event("watcher_given_up")
                return
        if self.phase == "normal":
            self._start_watcher()

    def _watcher_finished(self) -> bool:
        return not self.s.watcher_enabled or self.watcher.given_up or not self._watcher_alive()

    def _stop_watcher(self) -> None:
        proc = self.watcher.proc
        if proc is not None and proc.poll() is None:
            killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=self.s.kill_grace_s)
            except subprocess.TimeoutExpired:
                killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            self.event("watcher_stopped", pid=proc.pid)
        if self.watcher.log is not None:
            self.watcher.log.close()
            self.watcher.log = None

    def _set_drain(self) -> None:
        self._update_lease()  # phase != normal -> empty lease with drain: true

    # ---- evals + transfers

    def _eval_state(self, run_id: str) -> tuple[list[int], list[int], list[int]]:
        """(done, failed, pending) eval steps of a run."""
        done, failed, pending = [], [], []
        for ck in list_eval_checkpoints(run_dir(self.s.runs_root, run_id)):
            if eval_done_path(self.s.evals_root, run_id, ck.step).exists():
                done.append(ck.step)
            elif eval_failed_path(self.s.evals_root, run_id, ck.step).exists():
                failed.append(ck.step)
            else:
                pending.append(ck.step)
        return done, failed, pending

    def needs_transfer(self, run_id: str) -> bool:
        if self.runs[run_id].attempts == 0:
            return False  # never ran: nothing to transfer
        rec = read_json(transferred_path(run_dir(self.s.runs_root, run_id)), default=None)
        if not isinstance(rec, dict):
            return True
        done, _, _ = self._eval_state(run_id)
        return not set(done) <= set(rec.get("eval_steps_done", []))

    def _start_transfers(self, force: bool) -> None:
        limit = self.s.transfer_retries + 1
        for rid, run in self.runs.items():
            if not run.terminal or rid in self.transfers or not self.needs_transfer(rid):
                continue
            if self.transfer_failures.get(rid, 0) >= limit:
                continue
            done, _, pending = self._eval_state(rid)
            if pending and not force:
                continue
            if len(self.transfers) >= self.s.transfer_max_concurrent:
                return
            self._maybe_prune(rid)
            rdir = run_dir(self.s.runs_root, rid)
            if self.s.transfer_cmd is None:
                atomic_write_json(
                    transferred_path(rdir),
                    {"ts": _iso(self.clock()), "eval_steps_done": done, "noop": True},
                )
                self.event("transferred", run_id=rid, noop=True, eval_steps_done=done)
                continue
            spec = run.spec
            values = {
                **train_values(spec, self.s.runs_root),
                "eval_dir": self.s.evals_root / rid,
                "runs_root": self.s.runs_root,
                "evals_root": self.s.evals_root,
            }
            cmd = render(self.s.transfer_cmd, values)
            log = open(rdir / "logs" / "transfer.log", "ab")  # noqa: SIM115 - handed to a child process; closed when it exits
            proc = subprocess.Popen(
                ["/bin/sh", "-c", cmd],
                cwd=self.s.workdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.transfers[rid] = _Transfer(rid, proc, self.clock(), done, log)
            self.event("transfer_start", run_id=rid, pid=proc.pid, eval_steps_done=done)

    def _supervise_transfers(self, now: float) -> None:
        for rid, t in list(self.transfers.items()):
            rc = t.proc.poll()
            if rc is None:
                if t.term_sent is not None:
                    if now - t.term_sent >= self.s.kill_grace_s:
                        killpg(t.proc.pid, signal.SIGKILL)
                elif now - t.started > self.s.transfer_timeout_s:
                    t.term_sent = now
                    killpg(t.proc.pid, signal.SIGTERM)
                    self.event("transfer_timeout", run_id=rid)
                continue
            t.log.close()
            del self.transfers[rid]
            if rc == 0:
                atomic_write_json(
                    transferred_path(run_dir(self.s.runs_root, rid)),
                    {"ts": _iso(now), "eval_steps_done": t.steps, "noop": False},
                )
                self.event("transferred", run_id=rid, eval_steps_done=t.steps)
            else:
                self.transfer_failures[rid] = self.transfer_failures.get(rid, 0) + 1
                self.event(
                    "transfer_failed",
                    run_id=rid,
                    exit_code=rc,
                    failures=self.transfer_failures[rid],
                )

    def _kill_transfers(self) -> None:
        for rid, t in list(self.transfers.items()):
            killpg(t.proc.pid, signal.SIGKILL)
            t.proc.wait()
            t.log.close()
            self.event("transfer_abandoned", run_id=rid)
        self.transfers.clear()

    def _maybe_prune(self, rid: str) -> None:
        if rid in self.pruned or not self.s.pruning_enabled:
            return
        self.pruned.add(rid)
        res = prune_recovery_checkpoints(
            run_dir(self.s.runs_root, rid),
            self.s.keep_last,
            enabled=self.s.pruning_enabled,
            required_files=self.s.required_eval_files,
        )
        self.event(
            "prune",
            run_id=rid,
            pruned=[p.name for p in res.pruned],
            kept=[p.name for p in res.kept],
            refused=res.refused,
        )

    # ---- gate + deadline + release

    def gate_blockers(self) -> list[str]:
        """Why the node is not yet idle. Empty list = gate passes."""
        out = []
        for rid, run in self.runs.items():
            if not run.terminal:
                out.append(f"{rid}: {run.state}")
                continue
            if run.attempts == 0:
                continue
            _, failed, pending = self._eval_state(rid)
            if pending:
                out.append(f"{rid}: {len(pending)} eval(s) pending")
            if failed:
                out.append(f"{rid}: eval failed for steps {failed}")
            if self.needs_transfer(rid):
                out.append(f"{rid}: not transferred")
        return out

    def _on_deadline(self, now: float) -> None:
        self.deadline_hit = True
        self.event("deadline_reached", shutdown_opt_in=self.shutdown_opt_in)
        for run in self.runs.values():
            if run.running and run.term_sent is None:
                self._kill(run, "node_deadline", now)
            elif run.status is None:
                self._write_status(
                    run,
                    run_id=run.spec.run_id,
                    node=self.node,
                    status="timeout",
                    reason="not_started_before_deadline",
                    attempts=0,
                    exit_code=None,
                    ended_at=_iso(now),
                )
                self.event("skipped_at_deadline", run_id=run.spec.run_id)
        if self.shutdown_opt_in:
            self.phase = "deadline_evals"
            self.phase_deadline = now + self.s.deadline_eval_grace_s
            self._set_drain()

    def _release(self, reason: str) -> str:
        """End of the coordinator's job. Shuts down only on explicit opt-in."""
        if not self.shutdown_opt_in:
            self.result = "idle"
            self.event("coordinator_exit", result=self.result, reason=reason)
            return self.result
        self.event("shutdown_requested", reason=reason, command=self.s.shutdown_cmd)
        if self.dry_run or self._shutdown_cmd is None:
            self.result = "shutdown-skipped"
            self.event("shutdown_skipped", reason="dry-run")
            return self.result
        self.result = "shutdown"
        self._execute_shutdown(self._shutdown_cmd)
        return self.result

    def _execute_shutdown(self, cmd: str) -> None:
        if self.dry_run or not self.shutdown_opt_in:  # unreachable by construction
            raise RuntimeError("refusing to run the shutdown command")
        proc = subprocess.run(["/bin/sh", "-c", cmd], cwd=self.s.workdir, timeout=600)
        self.event("shutdown_executed", exit_code=proc.returncode)

    def tick(self) -> str | None:
        """One poll. Returns a result string once the coordinator is done."""
        if not self._started:
            self.start()
        if self.result:
            return self.result
        now = self.clock()
        if self._past_deadline(now) and not self.deadline_hit:
            self._on_deadline(now)
        self._supervise_runs(now)
        if self.phase == "normal":
            self._update_lease()
            if not self._past_deadline(now):
                self._launch_ready()
        self._supervise_watcher()
        self._supervise_transfers(now)

        if self.phase == "normal":
            self._start_transfers(force=False)
            blockers = self.gate_blockers()
            if blockers:
                if blockers != self._last_blockers:
                    self.event("gate_blocked", count=len(blockers), blockers=blockers[:20])
                    self._last_blockers = blockers
                return None
            atomic_write_json(
                self.idle_marker,
                {"node": self.node, "ts": _iso(now), "message": "node idle, safe to release"},
            )
            self.event("node_idle", marker=str(self.idle_marker))
            self.phase = "draining"
            self.phase_deadline = now + self.s.deadline_eval_grace_s
            self._set_drain()

        if self.phase == "draining":
            if self._watcher_finished() or now >= self.phase_deadline:
                self._stop_watcher()
                return self._release("gate passed")
            return None

        if self.phase == "deadline_evals":
            if self._watcher_finished() or now >= self.phase_deadline:
                self._stop_watcher()
                self.phase = "deadline_transfers"
                self.phase_deadline = now + self.s.deadline_transfer_grace_s
                self.transfer_failures.clear()
            else:
                return None

        if self.phase == "deadline_transfers":
            self._start_transfers(force=True)
            limit = self.s.transfer_retries + 1
            outstanding = [
                rid
                for rid, run in self.runs.items()
                if run.attempts > 0
                and (not run.terminal or self.needs_transfer(rid))
                and self.transfer_failures.get(rid, 0) < limit
            ]
            if (not self.transfers and not outstanding) or now >= self.phase_deadline:
                self._kill_transfers()
                return self._release("node hard deadline")
        return None

    def run(self) -> str:
        self.start()
        while True:
            result = self.tick()
            if result:
                return result
            self.sleep(self.poll_interval)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-node run coordinator (see module docstring).")
    ap.add_argument("--node", required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/allocation.yaml"))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="run everything as configured but never execute the shutdown command",
    )
    ap.add_argument(
        "--allow-shutdown",
        action="store_true",
        help="opt in to running coordinator.shutdown_cmd after the gate or at the deadline",
    )
    ap.add_argument("--print-plan", action="store_true", help="print the resolved plan and exit")
    ap.add_argument("--poll-interval", type=float, default=None)
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="launch without the node preflight (GPU stack, ninja, GPUs, data, exports, disk; train/preflight.py)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.print_plan:
        print(describe_plan(build_plan(resolve_paths(cfg), args.node)))
        return 0
    if not args.skip_preflight:
        from train import preflight  # imports this module; lazy so --print-plan stays light

        if preflight.main(["--node", args.node, "--config", str(args.config)]) != 0:
            print(
                f"coordinator: preflight failed on {args.node}; nothing launched. Fix the items above "
                "(`make preflight NODE=...` re-checks), or pass --skip-preflight to launch anyway.",
                file=sys.stderr,
            )
            return 1
    coord = Coordinator(
        cfg,
        args.node,
        dry_run=args.dry_run,
        allow_shutdown=args.allow_shutdown,
        poll_interval=args.poll_interval,
    )
    result = coord.run()
    print(f"coordinator finished: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
