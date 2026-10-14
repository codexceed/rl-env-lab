"""Eval watcher: evaluates eval checkpoints as they land.

Usage::

    python -m eval.watcher --config configs/allocation.yaml --node node1 [--gpus 6,7]

Normally the coordinator starts and supervises it. It can also run on its own.

What it does, every poll:

1. Scan ``<runs>/*/eval/step-*/COMPLETE`` (layout in `train/checkpoints.py` and
   `train/LAYOUT.md`). A step is pending until ``<evals>/<run_id>/step-NNNNN.done``
   or ``.failed`` exists.
   With ``commands.eval_base`` configured, the **base-model held-out evals** are
   pending too: one per (size, project version) among the watched runs (the
   node's queue plus every run with an eval checkpoint), until
   ``<evals>/<base_id>/step-00000.jsonl``, ``.done`` or ``.failed`` exists. The
   ids follow `eval/heldout.py` ``--checkpoint base``: ``base-<size>`` for v1 and
   v2 runs (as `eval/compare.py` and `eval/sametype_batch.py` read them), and
   ``v3-base-<size>`` with ``--variant v3`` for v3 runs (`eval/v3_report.py`).
2. Order pending steps by priority, which is the "eval falls behind" fallback:
   base-model evals first, then final checkpoints, then every 100 steps, then
   the rest. Ties go to the lower step, then run ID.
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
{step} {ckpt_dir} {output} {run_dir} {eval_dir}``. The base-eval template
(``commands.eval_base``) has the same ones (``reward`` is ``base``, ``seed`` empty,
``step`` 0, ``ckpt_dir`` ``base``) plus ``{variant}`` and ``{variant_args}``
(``--variant v3``, or nothing for a v1/v2 base). Without ``commands.eval_base``
no base eval is scheduled, as before.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    COMPLETE,
    atomic_write_json,
    eval_done_path,
    eval_failed_path,
    eval_output_path,
    eval_run_dir,
    list_eval_checkpoints,
    read_json,
    read_marker,
    read_status,
    run_dir,
    step_name,
)
from train.run_manifest import read_run_manifest

# ---------------------------------------------------------------- priority


@dataclass(frozen=True)
class Candidate:
    run_id: str
    step: int
    ckpt_dir: Path
    final: bool
    base: bool = False  # the base model's held-out eval (``commands.eval_base``)
    size: str = ""  # base evals only
    variant: str | None = None  # base evals only: `--variant` (v3), or None for ``base-<size>``

    @property
    def label(self) -> str:
        return f"{self.run_id}/{step_name(self.step)}"


def priority_key(c: Candidate) -> tuple[int, int, str]:
    """Base-model evals first, then final, then every 100 steps, then the rest; then step, run ID."""
    tier = -1 if c.base else 0 if c.final else (1 if c.step % 100 == 0 else 2)
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


# ---------------------------------------------------------------- base-model evals

# Project versions whose base model has its own held-out eval, run with ``--variant`` (ADR 0007: v3 changes the
# task set). Every other run (v1, and v2-think on the v1 tasks) is compared with the plain ``base-<size>`` eval.
BASE_VARIANTS = ("v3",)
_VARIANT_FLAG_RE = re.compile(r"--variant[= ]+(\S+)")


def base_eval_id(size: str, variant: str | None) -> str:
    """The run id `eval/heldout.py` gives ``--checkpoint base [--variant V]``: ``base-<size>``, ``<V>-base-<size>``."""
    return f"{variant}-base-{size}" if variant else f"base-{size}"


def run_variant(run_id: str, runs_root: Path, known: dict[str, dict]) -> str | None:
    """A run's ``--variant``: its run dir (run_manifest.json, train_config.json, COMPLETE), else its configured
    ``train_command``, else a ``<variant>-`` run id prefix for the `BASE_VARIANTS`; None for a v1 run."""
    rdir = run_dir(runs_root, run_id)
    sources: list[Any] = [read_run_manifest(rdir), read_json(rdir / "train_config.json")]
    sources += [read_marker(ck.path / COMPLETE) for ck in list_eval_checkpoints(rdir)[:1]]
    for src in sources:
        if isinstance(src, dict) and src.get("variant"):
            return str(src["variant"])
    if m := _VARIANT_FLAG_RE.search(str((known.get(run_id) or {}).get("train_command") or "")):
        return m[1]
    return next((v for v in BASE_VARIANTS if run_id.startswith(f"{v}-")), None)


def pending_base_evals(
    runs_root: Path, evals_root: Path, run_ids: set[str], known: dict[str, dict] | None = None
) -> list[Candidate]:
    """The base-model held-out evals ``run_ids`` need, one per (size, base variant), that have no output,
    ``.done`` or ``.failed`` yet (a base eval run by hand counts as done), in priority order."""
    known = known or {}
    wanted: dict[str, Candidate] = {}
    for rid in sorted(run_ids):
        size = str(run_metadata(rid, runs_root, known).get("size") or "")
        if not size:
            continue
        variant = run_variant(rid, runs_root, known)
        variant = variant if variant in BASE_VARIANTS else None
        bid = base_eval_id(size, variant)
        wanted.setdefault(bid, Candidate(bid, 0, Path("base"), False, base=True, size=size, variant=variant))
    out = [
        c
        for c in wanted.values()
        if not any(
            p.exists()
            for p in (
                eval_output_path(evals_root, c.run_id, 0),
                eval_done_path(evals_root, c.run_id, 0),
                eval_failed_path(evals_root, c.run_id, 0),
            )
        )
    ]
    return sorted(out, key=priority_key)


def runs_with_checkpoints(runs_root: Path, run_ids: set[str] | None = None) -> set[str]:
    """Run ids under ``runs_root`` (within ``run_ids`` when given) with at least one complete eval checkpoint."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return set()
    return {
        rdir.name
        for rdir in runs_root.iterdir()
        if rdir.is_dir()
        and not rdir.name.startswith(("_", "."))
        and (run_ids is None or rdir.name in run_ids)
        and list_eval_checkpoints(rdir)
    }


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


RAW_PLACEHOLDERS = frozenset({"variant_args"})  # already shell-quoted argument lists, possibly empty


def render(template: str, values: dict[str, Any]) -> str:
    """Fill ``{placeholders}``; values are shell-quoted (`RAW_PLACEHOLDERS` are inserted as they are)."""
    return template.format_map({k: str(v) if k in RAW_PLACEHOLDERS else shlex.quote(str(v)) for k, v in values.items()})


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
    base_eval_command: str | None = None  # commands.eval_base; None = no base-model evals
    base_run_ids: set[str] = field(default_factory=set)  # runs whose base evals to schedule before they start

    def base_scope(self) -> set[str]:
        """Runs whose base-model evals this watcher schedules: the node's queue and every watched run with an
        eval checkpoint."""
        return set(self.base_run_ids) | runs_with_checkpoints(self.runs_root, self.run_ids)

    @classmethod
    def from_config(cls, cfg: dict, node: str | None = None, **overrides: Any) -> WatcherSettings:
        paths = cfg.get("paths", {})
        ev = cfg.get("eval", {})
        workdir = Path(paths.get("workdir", ".")).resolve()

        def rel(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else workdir / q

        node_cfg = (cfg.get("nodes") or {}).get(node, {}) if node else {}
        queued = {r for g in node_cfg.get("groups", []) for r in g.get("queue", [])}
        run_ids = queued if node and ev.get("watch_scope", "all") == "node" else None
        commands = cfg.get("commands") or {}
        settings = cls(
            runs_root=rel(paths.get("runs", "runs")),
            evals_root=rel(paths.get("evals", "evals")),
            eval_command=commands["eval"],
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
            base_eval_command=commands.get("eval_base") or None,
            base_run_ids=queued,
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
        if self.s.base_eval_command:
            for variant in (None, *BASE_VARIANTS):
                bid = base_eval_id("2b", variant)
                self.command(Candidate(bid, 0, Path("base"), False, base=True, size="2b", variant=variant), (0,))

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

    def command(self, c: Candidate, gpus: tuple[int, ...]) -> str:
        """The shell command for one candidate: ``commands.eval_base`` for a base eval, else ``commands.eval``."""
        if c.base:
            if not self.s.base_eval_command:
                raise ValueError(f"{c.label}: base eval without commands.eval_base")
            return render(self.s.base_eval_command, self._values(c, gpus))
        return render(self.s.eval_command, self._values(c, gpus))

    def pending(self) -> list[Candidate]:
        """Everything still to evaluate, in priority order (base-model evals only with commands.eval_base)."""
        out = pending_checkpoints(self.s.runs_root, self.s.evals_root, self.s.run_ids)
        if self.s.base_eval_command:
            base = pending_base_evals(self.s.runs_root, self.s.evals_root, self.s.base_scope(), self.s.run_meta)
            out = sorted([*base, *out], key=priority_key)
        return out

    def _values(self, c: Candidate, gpus: tuple[int, ...]) -> dict[str, Any]:
        if c.base:
            return {
                "run_id": c.run_id,
                "size": c.size,
                "reward": "base",
                "seed": "",
                "gpus": ",".join(str(g) for g in gpus),
                "num_gpus": len(gpus),
                "step": 0,
                "ckpt_dir": "base",
                "output": eval_output_path(self.s.evals_root, c.run_id, 0),
                "run_dir": run_dir(self.s.runs_root, c.run_id),
                "eval_dir": eval_run_dir(self.s.evals_root, c.run_id),
                "variant": c.variant or "",
                "variant_args": shlex.join(["--variant", c.variant]) if c.variant else "",
            }
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
        cmd = self.command(c, gpus)
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
                for cand in self.pending():
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
        self._log(
            "watcher_start",
            pid=self.pid,
            gpus_per_eval=self.s.gpus_per_eval,
            base_evals=bool(self.s.base_eval_command),
        )
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
