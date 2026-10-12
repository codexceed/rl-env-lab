"""Eval watcher: evaluates eval checkpoints as they land.

Usage::

    python -m eval.watcher --config configs/allocation.yaml --node node1 [--gpus 6,7]

Normally the coordinator starts and supervises it. It can also run on its own.

What it does, every poll:

1. Scan ``<runs>/*/eval/step-*/COMPLETE`` (layout in `train/checkpoints.py` and
   `train/LAYOUT.md`). A step is pending until ``<evals>/<run_id>/step-NNNNN.done``
   or ``.failed`` exists.
2. Order pending steps by priority, which is the "eval falls behind" fallback:
   final checkpoints first, then every 100 steps, then the rest. Ties go to the
   lower step, then run ID.
3. Fill free GPU slots (``eval.gpus_per_eval`` GPUs each) with the
   ``commands.eval`` template. It sets ``CUDA_VISIBLE_DEVICES`` and runs each
   eval in its own process group with a timeout.
4. On exit 0 with the output file present, write ``step-NNNNN.done``. On
   failure, retry up to ``eval.retries`` times, then write ``.failed``.

GPUs: with ``--lease-file`` the GPU set comes from the coordinator's lease
(``<runs>/_coordinator/<node>.eval_gpus.json``: ``{"gen", "gpus", "drain"}``).
The watcher acks each lease generation in its state file, so the coordinator
knows when revoked GPUs are actually free. With ``drain: true`` it stops taking
new work and exits once in-flight evals finish. Without a lease file it uses
``--gpus`` (or the node's ``eval_gpus``).

Several watchers can share one runs root (for example on shared storage). A
step is claimed by creating ``step-NNNNN.claim`` exclusively. Claims are
heartbeated. A claim whose owner died, or whose heartbeat is older than
``eval.stale_claim_s``, is taken over.

The watcher never deletes anything. Locks are released or broken by renaming.
It never shuts a machine down.

Eval command placeholders: ``{run_id} {size} {reward} {seed} {gpus} {num_gpus}
{step} {ckpt_dir} {output} {run_dir} {eval_dir}``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib

import yaml

from train.checkpoints import (
    atomic_write_json,
    eval_done_path,
    eval_failed_path,
    eval_output_path,
    eval_run_dir,
    list_eval_checkpoints,
    read_json,
    read_status,
    run_dir,
    step_name,
)

# ---------------------------------------------------------------- priority


@dataclass(frozen=True)
class Candidate:
    run_id: str
    step: int
    ckpt_dir: Path
    final: bool

    @property
    def label(self) -> str:
        return f"{self.run_id}/{step_name(self.step)}"


def priority_key(c: Candidate) -> tuple[int, int, str]:
    """Final first, then every 100 steps, then the rest; then step, run ID."""
    tier = 0 if c.final else (1 if c.step % 100 == 0 else 2)
    return (tier, c.step, c.run_id)


def pending_checkpoints(runs_root: Path, evals_root: Path, run_ids: set[str] | None = None) -> list[Candidate]:
    """Complete eval checkpoints with no ``.done``/``.failed``, in priority order."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    out = []
    for rdir in runs_root.iterdir():
        if not rdir.is_dir() or rdir.name.startswith(("_", ".")):
            continue
        if run_ids is not None and rdir.name not in run_ids:
            continue
        for ck in list_eval_checkpoints(rdir):
            if eval_done_path(evals_root, ck.run_id, ck.step).exists():
                continue
            if eval_failed_path(evals_root, ck.run_id, ck.step).exists():
                continue
            out.append(Candidate(ck.run_id, ck.step, ck.path, ck.final))
    return sorted(out, key=priority_key)


# ---------------------------------------------------------------- helpers


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def killpg(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def render(template: str, values: dict[str, Any]) -> str:
    """Fill ``{placeholders}``; values are shell-quoted."""
    return template.format_map({k: shlex.quote(str(v)) for k, v in values.items()})


_REWARD_ALIASES = {"rob": "robust", "leak": "leaky"}


def run_metadata(run_id: str, runs_root: Path, known: dict[str, dict]) -> dict[str, Any]:
    """size/reward/seed for a run: config first, then status.json, then the ID."""
    if run_id in known:
        meta = known[run_id]
        return {k: meta.get(k, "") for k in ("size", "reward", "seed")}
    status = read_status(run_dir(runs_root, run_id)) or {}
    if all(k in status for k in ("size", "reward", "seed")):
        return {k: status[k] for k in ("size", "reward", "seed")}
    parts = run_id.split("-")
    if len(parts) >= 3 and parts[2].startswith("s"):
        return {
            "size": parts[0],
            "reward": _REWARD_ALIASES.get(parts[1], parts[1]),
            "seed": parts[2][1:],
        }
    return {"size": "", "reward": "", "seed": ""}


# ---------------------------------------------------------------- settings


@dataclass
class WatcherSettings:
    runs_root: Path
    evals_root: Path
    eval_command: str
    workdir: Path = field(default_factory=Path.cwd)
    gpus_per_eval: int = 1
    timeout_s: float = 5400.0
    retries: int = 1
    stale_claim_s: float = 900.0
    poll_interval_s: float = 30.0
    kill_grace_s: float = 30.0
    static_gpus: list[int] = field(default_factory=list)
    lease_file: Path | None = None
    state_file: Path | None = None
    run_ids: set[str] | None = None
    run_meta: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, node: str | None = None, **overrides: Any) -> WatcherSettings:
        paths = cfg.get("paths", {})
        ev = cfg.get("eval", {})
        workdir = Path(paths.get("workdir", ".")).resolve()

        def rel(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else workdir / q

        node_cfg = (cfg.get("nodes") or {}).get(node, {}) if node else {}
        run_ids = None
        if node and ev.get("watch_scope", "all") == "node":
            run_ids = {r for g in node_cfg.get("groups", []) for r in g.get("queue", [])}
        settings = cls(
            runs_root=rel(paths.get("runs", "runs")),
            evals_root=rel(paths.get("evals", "evals")),
            eval_command=(cfg.get("commands") or {})["eval"],
            workdir=workdir,
            gpus_per_eval=int(ev.get("gpus_per_eval", 1)),
            timeout_s=float(ev.get("timeout_s", 5400)),
            retries=int(ev.get("retries", 1)),
            stale_claim_s=float(ev.get("stale_claim_s", 900)),
            poll_interval_s=float(ev.get("poll_interval_s", 30)),
            kill_grace_s=float((cfg.get("coordinator") or {}).get("kill_grace_s", 30)),
            static_gpus=list(node_cfg.get("eval_gpus", [])),
            run_ids=run_ids,
            run_meta=dict(cfg.get("runs") or {}),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(settings, k, v)
        return settings


# ---------------------------------------------------------------- watcher


@dataclass
class _Job:
    cand: Candidate
    gpus: tuple[int, ...]
    proc: subprocess.Popen
    started: float
    claim: Path
    attempt: int
    log: Any
    term_sent: float | None = None
    timed_out: bool = False


class Watcher:
    def __init__(
        self,
        settings: WatcherSettings,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.s = settings
        self.clock = clock
        self.sleep = sleep
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.token = uuid.uuid4().hex
        self.jobs: list[_Job] = []
        self.stopping = False
        self.ack_gen = -1
        # Fail fast on a bad template rather than on the first checkpoint.
        render(self.s.eval_command, self._values(Candidate("x-y-s0", 0, Path("."), False), (0,)))

    # ---- lease / state

    def read_lease(self) -> tuple[int, list[int], bool]:
        if self.s.lease_file is None:
            return 0, list(self.s.static_gpus), False
        data = read_json(self.s.lease_file, default=None)
        if not isinstance(data, dict):
            return self.ack_gen, [], False  # no lease yet: take no new work
        return (
            int(data.get("gen", 0)),
            [int(g) for g in data.get("gpus", [])],
            bool(data.get("drain", False)),
        )

    def busy_gpus(self) -> dict[int, str]:
        return {g: j.cand.label for j in self.jobs for g in j.gpus}

    def write_state(self, draining: bool) -> None:
        if self.s.state_file is None:
            return
        atomic_write_json(
            self.s.state_file,
            {
                "pid": self.pid,
                "host": self.host,
                "token": self.token,
                "ack_gen": self.ack_gen,
                "busy": {str(g): label for g, label in self.busy_gpus().items()},
                "inflight": len(self.jobs),
                "draining": draining,
                "ts": self.clock(),
            },
        )

    # ---- claims

    def _claim_path(self, c: Candidate) -> Path:
        return eval_run_dir(self.s.evals_root, c.run_id) / f"{step_name(c.step)}.claim"

    def _claim_is_stale(self, path: Path) -> bool:
        info = read_json(path, default={}) or {}
        if info.get("token") == self.token:
            return False
        if info.get("host") == self.host and not pid_alive(int(info.get("pid", -1))):
            return True
        try:
            age = self.clock() - path.stat().st_mtime
        except OSError:
            return False
        return age > self.s.stale_claim_s

    def try_claim(self, c: Candidate) -> Path | None:
        path = self._claim_path(c)
        path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if not self._claim_is_stale(path):
                    return None
                stale = path.with_name(f"{path.name}.stale-{int(self.clock() * 1000)}-{self.pid}")
                try:
                    os.rename(path, stale)  # atomic; only one breaker wins
                except OSError:
                    return None
                continue
            with os.fdopen(fd, "w") as fh:
                json.dump(
                    {
                        "host": self.host,
                        "pid": self.pid,
                        "token": self.token,
                        "ts": self.clock(),
                        "label": c.label,
                    },
                    fh,
                )
            return path
        return None

    def _release(self, claim: Path, attempt: int) -> None:
        with contextlib.suppress(OSError):
            os.rename(claim, claim.with_name(f"{claim.name}.released-{attempt}-{self.token[:8]}"))

    def _heartbeat(self) -> None:
        for job in self.jobs:
            with contextlib.suppress(OSError):
                os.utime(job.claim)

    # ---- launching

    def _values(self, c: Candidate, gpus: tuple[int, ...]) -> dict[str, Any]:
        return {
            "run_id": c.run_id,
            **run_metadata(c.run_id, self.s.runs_root, self.s.run_meta),
            "gpus": ",".join(str(g) for g in gpus),
            "num_gpus": len(gpus),
            "step": c.step,
            "ckpt_dir": c.ckpt_dir,
            "output": eval_output_path(self.s.evals_root, c.run_id, c.step),
            "run_dir": run_dir(self.s.runs_root, c.run_id),
            "eval_dir": eval_run_dir(self.s.evals_root, c.run_id),
        }

    def _attempts_so_far(self, c: Candidate) -> int:
        d = eval_run_dir(self.s.evals_root, c.run_id)
        return len(list(d.glob(f"{step_name(c.step)}.fail-*"))) if d.is_dir() else 0

    def _launch(self, c: Candidate, gpus: tuple[int, ...], claim: Path) -> None:
        attempt = self._attempts_so_far(c) + 1
        log_dir = eval_run_dir(self.s.evals_root, c.run_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / f"{step_name(c.step)}.attempt-{attempt}.log", "ab")  # noqa: SIM115 - handed to a child process; closed when it exits
        cmd = render(self.s.eval_command, self._values(c, gpus))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
        proc = subprocess.Popen(
            ["/bin/sh", "-c", cmd],
            cwd=self.s.workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.jobs.append(_Job(c, gpus, proc, self.clock(), claim, attempt, log))
        self._log("eval_start", label=c.label, gpus=list(gpus), attempt=attempt, pid=proc.pid)

    # ---- reaping

    def _finish(self, job: _Job, rc: int) -> None:
        job.log.close()
        c = job.cand
        out = eval_output_path(self.s.evals_root, c.run_id, c.step)
        duration = self.clock() - job.started
        if rc == 0 and not job.timed_out and out.exists():
            atomic_write_json(
                eval_done_path(self.s.evals_root, c.run_id, c.step),
                {
                    "run_id": c.run_id,
                    "step": c.step,
                    "final": c.final,
                    "output": str(out),
                    "duration_s": round(duration, 3),
                    "host": self.host,
                    "ts": self.clock(),
                },
            )
            self._log("eval_done", label=c.label, duration_s=round(duration, 3))
            self._release(job.claim, job.attempt)
            return
        reason = "timeout" if job.timed_out else ("no output" if rc == 0 else f"exit {rc}")
        d = eval_run_dir(self.s.evals_root, c.run_id)
        atomic_write_json(
            d / f"{step_name(c.step)}.fail-{job.attempt}",
            {"reason": reason, "rc": rc, "host": self.host, "ts": self.clock()},
        )
        if job.attempt > self.s.retries:
            atomic_write_json(
                eval_failed_path(self.s.evals_root, c.run_id, c.step),
                {"reason": reason, "attempts": job.attempt, "ts": self.clock()},
            )
            self._log("eval_failed", label=c.label, reason=reason, attempts=job.attempt)
        else:
            self._log("eval_retry", label=c.label, reason=reason, attempt=job.attempt)
        self._release(job.claim, job.attempt)

    def _reap(self) -> None:
        now = self.clock()
        still = []
        for job in self.jobs:
            rc = job.proc.poll()
            if rc is not None:
                self._finish(job, rc)
                continue
            if job.term_sent is not None:
                if now - job.term_sent >= self.s.kill_grace_s:
                    killpg(job.proc.pid, signal.SIGKILL)
            elif now - job.started > self.s.timeout_s:
                job.timed_out = True
                job.term_sent = now
                killpg(job.proc.pid, signal.SIGTERM)
            still.append(job)
        self.jobs = still

    # ---- main loop

    def tick(self) -> bool:
        """One poll. Returns False when the watcher should exit (drained)."""
        self._reap()
        self._heartbeat()
        gen, lease_gpus, drain = self.read_lease()
        if not drain and not self.stopping:
            busy = self.busy_gpus()
            free = [g for g in sorted(lease_gpus) if g not in busy]
            k = max(1, self.s.gpus_per_eval)
            slots = [tuple(free[i : i + k]) for i in range(0, len(free) - k + 1, k)]
            if slots:
                for cand in pending_checkpoints(self.s.runs_root, self.s.evals_root, self.s.run_ids):
                    if not slots:
                        break
                    if any(j.cand == cand for j in self.jobs):
                        continue
                    claim = self.try_claim(cand)
                    if claim is not None:
                        self._launch(cand, slots.pop(0), claim)
        self.ack_gen = gen
        self.write_state(draining=drain)
        return not (drain and not self.jobs)

    def stop(self) -> None:
        """Kill in-flight evals (their claims go stale and are re-taken later)."""
        for job in self.jobs:
            killpg(job.proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.s.kill_grace_s
        for job in self.jobs:
            try:
                job.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                killpg(job.proc.pid, signal.SIGKILL)
                job.proc.wait()
            job.log.close()
        self.jobs = []

    def run(self) -> int:
        def _on_term(signum: int, _frame: Any) -> None:
            self.stopping = True

        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)
        self._log("watcher_start", pid=self.pid, gpus_per_eval=self.s.gpus_per_eval)
        try:
            while not self.stopping:
                if not self.tick():
                    self._log("watcher_drained")
                    return 0
                self.sleep(self.s.poll_interval_s)
        finally:
            self.stop()
            self.write_state(draining=True)
        self._log("watcher_stopped")
        return 0

    def _log(self, event: str, **fields: Any) -> None:
        rec = {"ts": round(self.clock(), 3), "event": event, **fields}
        print(json.dumps(rec, default=str), flush=True)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--node", default=None, help="node name in the config (eval GPUs, scope)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPUs (when no lease file)")
    ap.add_argument("--lease-file", type=Path, default=None)
    ap.add_argument("--state-file", type=Path, default=None)
    ap.add_argument("--poll-interval", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    static = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    settings = WatcherSettings.from_config(
        cfg,
        args.node,
        static_gpus=static,
        lease_file=args.lease_file,
        state_file=args.state_file,
        poll_interval_s=args.poll_interval,
    )
    return Watcher(settings).run()


if __name__ == "__main__":
    sys.exit(main())
