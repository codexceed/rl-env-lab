"""Exploratory same-type eval of every saved eval checkpoint (ADR 0005 item 2).

    uv run python eval/sametype_batch.py --gpus 4,5,6,7                 # `make sametype-evals GPUS=4,5,6,7`
    uv run python eval/sametype_batch.py --gpus 6 --runs '2b-*' --dry-run
    uv run python eval/sametype_batch.py --gpus 0,1,2,3 --final-only           # one job per run: its last checkpoint

**Jobs.** Every ``runs/<id>/eval/step-NNNNN/`` with ``COMPLETE`` (the trainer writes it last), plus
``base-<size>`` (step 0) for each size among those runs. By default only run-matrix ids, v2 and v3 included
(``[v<k>-]<size>-<rob|leak>-s<seed>...``), are taken; ``--runs`` (fnmatch globs, comma-separated)
selects others, e.g. ``--runs 'dry-*'``. A v2 checkpoint is evaluated with its own variant settings
(`eval/heldout.py` reads them from the run dir). A job whose output already exists is skipped, so the driver can be
re-run at any time to pick up new checkpoints.

**v3 runs** (ADR 0007; variant ``v3`` in the run dir, or a ``v3-`` run id) read their sametype split from
``--v3-data-dir`` (default ``data/v3``) instead of ``--data-dir``, and bring ``v3-base-<size>`` (the base model
with ``--variant v3``: thinking on, cap 1,024) for each of their sizes instead of ``base-<size>``. Their outputs
are ``evals/v3-<id>/sametype/step-NNNNN.jsonl``. v1 and v2 jobs and their commands are unchanged.

**Metadata** (size, reward, seed) comes from ``runs/<id>/run_manifest.json`` (`train/run_manifest.py`),
else the ``COMPLETE`` marker, else ``status.json``, else the run id.

**Execution.** One `eval/heldout.py --mode sametype` subprocess per GPU in ``--gpus``, each with
``CUDA_VISIBLE_DEVICES`` set to that one GPU and in its own process group; no other GPU is ever
touched, so pick GPUs that training and the eval watcher are not using. Each job has a timeout
(default 1800 s; its process group is killed on expiry, or when the driver itself is stopped).
Output goes to ``evals/<run>/sametype/step-NNNNN.jsonl`` (+ ``.traces.jsonl``), a subdirectory the
verdict code never reads; each job logs to ``evals/<run>/sametype/logs/step-NNNNN.log``. Nothing
is ever deleted.

**Progress.** ``\\nPROGRESS done/total`` after each job, then a JSON summary line and
``PROGRESS_EXIT <code>`` (0 when every job succeeded), as lightning-jobs ``progress.py`` expects.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):  # run as a script: make `eval` and `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envlab.tasks.v3 import DATA_DIR as V3_DATA_DIR
from eval.heldout import sametype_output_path
from train.checkpoints import COMPLETE, list_eval_checkpoints, read_json, read_marker, read_status, step_name
from train.grpo import SIZES
from train.run_manifest import read_run_manifest
from train.venv_bin import ensure_venv_bin_on_path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELDOUT = REPO_ROOT / "eval" / "heldout.py"
DEFAULT_TIMEOUT_S = 1800.0
KILL_GRACE_S = 30.0
POLL_S = 5.0
RUN_ID_RE = re.compile(r"^(?:v\d+-)?(?P<size>\d+(?:\.\d+)?b)-(?P<reward>rob|leak)-s(?P<seed>\d+)")
REWARD_ALIASES = {"rob": "robust", "leak": "leaky"}
V3_VARIANT = "v3"  # configs/variants/v3.yaml (ADR 0007)


@dataclass(frozen=True)
class Job:
    run_id: str
    size: str
    reward: str  # robust | leaky | base
    seed: int | None
    step: int
    checkpoint: Path | None  # None: the base model
    output: Path
    v3: bool = False  # ADR 0007: v3 data dir; a v3 base job runs with --variant v3

    @property
    def label(self) -> str:
        return f"{self.run_id}@{self.step}"

    def argv(self, gpu: int, *, python: str = sys.executable, extra: Sequence[str] = ()) -> list[str]:
        """The `eval/heldout.py` command for this job on one GPU."""
        cmd = [python, "-u", str(HELDOUT), "--size", self.size, "--mode", "sametype"]
        if self.checkpoint is None:
            cmd += ["--checkpoint", "base"]
            if self.v3:
                cmd += ["--variant", V3_VARIANT]
        else:
            seed = "" if self.seed is None else str(self.seed)
            cmd += ["--run-id", self.run_id, "--reward", self.reward, "--seed", seed]
            cmd += ["--checkpoint", str(self.checkpoint), "--step", str(self.step)]
        return [*cmd, "--output", str(self.output), "--gpus", str(gpu), *extra]


# ---------------------------------------------------------------- planning


def run_metadata(rdir: Path) -> dict[str, Any] | None:
    """size / reward / seed of a run: run_manifest.json, the COMPLETE marker, status.json, the id."""
    sources: list[dict[str, Any]] = [read_run_manifest(rdir) or {}]
    for ck in list_eval_checkpoints(rdir)[:1]:
        sources.append(read_marker(ck.path / COMPLETE))
    sources.append(read_status(rdir) or {})
    if m := RUN_ID_RE.match(rdir.name):
        sources.append({"size": m["size"], "reward": REWARD_ALIASES[m["reward"]], "seed": int(m["seed"])})
    for src in sources:
        size, reward = src.get("size"), REWARD_ALIASES.get(str(src.get("reward")), src.get("reward"))
        if size in SIZES and reward in ("robust", "leaky") and "seed" in src:
            seed = src["seed"]
            return {"size": size, "reward": reward, "seed": None if seed in (None, "") else int(seed)}
    return None


def is_v3_run(rdir: Path) -> bool:
    """ADR 0007: the run trained with ``--variant v3`` (run_manifest.json, train_config.json, COMPLETE) or has a
    ``v3-`` run id."""
    sources: list[Any] = [read_run_manifest(rdir), read_json(rdir / "train_config.json")]
    sources += [read_marker(ck.path / COMPLETE) for ck in list_eval_checkpoints(rdir)[:1]]
    for src in sources:
        if isinstance(src, dict) and src.get("variant"):
            return src["variant"] == V3_VARIANT
    return rdir.name.startswith(f"{V3_VARIANT}-")


def _selected(run_id: str, patterns: Sequence[str] | None) -> bool:
    if patterns:
        return any(fnmatch.fnmatchcase(run_id, p) for p in patterns)
    return RUN_ID_RE.match(run_id) is not None


def plan_jobs(
    runs_root: Path, evals_root: Path, patterns: Sequence[str] | None = None, *, final_only: bool = False
) -> tuple[list[Job], list[Job], list[str]]:
    """(jobs to run, jobs whose output already exists, notes); bases first, then by run id and step.

    With ``final_only``, each run contributes only its last complete eval checkpoint (its final step once
    training has ended).
    """
    runs_root, evals_root = Path(runs_root), Path(evals_root)
    wanted: list[Job] = []
    notes: list[str] = []
    sizes: set[str] = set()
    v3_sizes: set[str] = set()
    rdirs = sorted(p for p in runs_root.iterdir() if p.is_dir()) if runs_root.is_dir() else []
    for rdir in rdirs:
        if rdir.name.startswith(("_", ".")) or not _selected(rdir.name, patterns):
            continue
        checkpoints = list_eval_checkpoints(rdir)
        if final_only:
            checkpoints = checkpoints[-1:]
        if not checkpoints:
            continue
        meta = run_metadata(rdir)
        if meta is None:
            notes.append(f"{rdir.name}: no size/reward/seed in run_manifest.json, COMPLETE, status.json or the id")
            continue
        v3 = is_v3_run(rdir)
        (v3_sizes if v3 else sizes).add(meta["size"])
        for ck in checkpoints:
            out = sametype_output_path(evals_root, rdir.name, ck.step)
            wanted.append(Job(rdir.name, meta["size"], meta["reward"], meta["seed"], ck.step, ck.path, out, v3))
    bases = [
        Job(f"base-{s}", s, "base", None, 0, None, sametype_output_path(evals_root, f"base-{s}", 0))
        for s in sorted(sizes, key=SIZES.index)
    ]
    for s in sorted(v3_sizes, key=SIZES.index):
        rid = f"{V3_VARIANT}-base-{s}"
        bases.append(Job(rid, s, "base", None, 0, None, sametype_output_path(evals_root, rid, 0), True))
    jobs = [*bases, *wanted]
    return [j for j in jobs if not j.output.exists()], [j for j in jobs if j.output.exists()], notes


# ---------------------------------------------------------------- execution


class Proc(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass
class Result:
    job: Job
    gpu: int
    rc: int | None
    timed_out: bool
    seconds: float

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out and self.job.output.exists()


def log_path(job: Job) -> Path:
    return job.output.parent / "logs" / f"{step_name(job.step)}.log"


def launch_subprocess(job: Job, gpu: int, extra: Sequence[str] = ()) -> subprocess.Popen:
    """Start one eval on ``gpu`` only, in its own process group, logging to `log_path`."""
    log = log_path(job)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    env.pop("UV_LIGHTNING_VIRTUALENV_ROOT", None)
    ensure_venv_bin_on_path(env)  # as `uv run` does: vLLM's sampler JIT-builds with `ninja` from PATH
    with open(log, "ab") as fh:  # the child keeps its own copy of the descriptor
        return subprocess.Popen(
            job.argv(gpu, extra=extra),
            cwd=REPO_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def stop_group(proc: Proc, grace_s: float = KILL_GRACE_S) -> int:
    """SIGTERM the job's process group, then SIGKILL after ``grace_s``; returns the exit code."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)
        try:
            return proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            continue
    return proc.wait()


def run_jobs(
    jobs: Iterable[Job],
    gpus: Sequence[int],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    launch: Callable[[Job, int], Proc] = launch_subprocess,
    stop: Callable[[Proc], int] = stop_group,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = POLL_S,
    emit: Callable[[str], None] = print,
) -> list[Result]:
    """Run ``jobs``, at most one per GPU at a time; every job is reaped (or killed) before returning."""
    pending = deque(jobs)
    total = len(pending)
    running: dict[int, tuple[Job, Proc, float]] = {}
    results: list[Result] = []
    try:
        while pending or running:
            for gpu in gpus:
                if gpu not in running and pending:
                    job = pending.popleft()
                    running[gpu] = (job, launch(job, gpu), clock())
                    emit(f"[sametype] start {job.label} on GPU {gpu}")
            finished = False
            for gpu, (job, proc, started) in list(running.items()):
                rc = proc.poll()
                timed_out = rc is None and clock() - started > timeout_s
                if timed_out:
                    rc = stop(proc)
                if rc is None:
                    continue
                del running[gpu]
                finished = True
                res = Result(job, gpu, rc, timed_out, round(clock() - started, 1))
                results.append(res)
                state = "ok" if res.ok else ("timeout" if timed_out else f"failed (exit {rc}, see {log_path(job)})")
                emit(f"[sametype] {job.label} on GPU {gpu}: {state} after {res.seconds:.0f} s")
                emit(f"\nPROGRESS {len(results)}/{total}")
            if not finished and (pending or running):
                sleep(poll_s)
    finally:
        for _job, proc, _ in running.values():  # interrupted: never leave an eval holding a GPU
            stop(proc)
    return results


def summary(results: Sequence[Result], skipped: Sequence[Job]) -> dict[str, Any]:
    return {
        "exploratory": "ADR 0005",
        "total": len(results),
        "ok": sum(r.ok for r in results),
        "failed": [r.job.label for r in results if not r.ok and not r.timed_out],
        "timeout": [r.job.label for r in results if r.timed_out],
        "skipped_existing": len(skipped),
    }


# ---------------------------------------------------------------- CLI


def parse_gpus(text: str) -> list[int]:
    gpus = [int(g) for g in str(text).replace(" ", "").split(",") if g != ""]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise argparse.ArgumentTypeError(f"--gpus needs distinct GPU indices, e.g. 4,5,6,7; got {text!r}")
    return gpus


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exploratory same-type eval of every saved eval checkpoint (ADR 0005)")
    p.add_argument("--gpus", required=True, type=parse_gpus, help="physical GPU indices, one eval per GPU")
    p.add_argument("--runs", help="comma-separated fnmatch globs over run ids (default: run-matrix ids)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per job")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--evals-root", type=Path, default=Path("evals"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--v3-data-dir", type=Path, default=Path(V3_DATA_DIR), help="sametype split of v3 runs (ADR 0007)")
    p.add_argument("--config", type=Path, help="passed to eval/heldout.py")
    p.add_argument("--final-only", action="store_true", help="only each run's last complete eval checkpoint")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The evals run with the repo root as cwd: hand them absolute paths.
    args.runs_root, args.evals_root, args.data_dir, args.v3_data_dir = (
        p.resolve() for p in (args.runs_root, args.evals_root, args.data_dir, args.v3_data_dir)
    )
    patterns = [p for p in (args.runs or "").split(",") if p] or None
    jobs, skipped, notes = plan_jobs(args.runs_root, args.evals_root, patterns, final_only=args.final_only)
    for note in notes:
        print(f"[sametype] skip {note}", flush=True)
    print(f"[sametype] {len(jobs)} to run, {len(skipped)} already done, GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"  {job.label:<22} -> {job.output}")
        return 0
    for v3, hint in ((False, "make data-sametype"), (True, "make data-v3")):
        data_dir = args.v3_data_dir if v3 else args.data_dir
        if any(j.v3 == v3 for j in jobs) and not (data_dir / "sametype.jsonl").is_file():
            print(f"[sametype] {data_dir / 'sametype.jsonl'} not found: run `{hint}`", file=sys.stderr)
            print("PROGRESS_EXIT 2", flush=True)
            return 2

    def extra_for(job: Job) -> list[str]:
        """``--evals-root`` and the job's ``--data-dir`` (v3: ``--v3-data-dir``), then ``--config``."""
        extra = ["--evals-root", str(args.evals_root), "--data-dir", str(args.v3_data_dir if job.v3 else args.data_dir)]
        return [*extra, "--config", str(args.config.resolve())] if args.config else extra

    def emit(line: str) -> None:
        print(line, flush=True)

    # `kill <driver>` must still stop the evals it started: SystemExit lets run_jobs clean up.
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        results = run_jobs(
            jobs,
            args.gpus,
            timeout_s=args.timeout,
            launch=lambda job, gpu: launch_subprocess(job, gpu, extra_for(job)),
            emit=emit,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    report = summary(results, skipped)
    rc = 0 if report["ok"] == report["total"] else 1
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"PROGRESS_EXIT {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
