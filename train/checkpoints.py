"""Artifact layout contract and recovery-checkpoint retention.

This module is the single definition of where run artifacts live. The trainer
(`train/grpo.py`), the eval watcher (`eval/watcher.py`), the eval script
(`eval/heldout.py`) and the coordinator (`train/coordinator.py`) all follow it.
The same contract is documented for humans in `train/LAYOUT.md`.

Layout (``<runs>`` and ``<evals>`` are `paths.runs` / `paths.evals` in
`configs/allocation.yaml`)::

    <runs>/<run_id>/
        status.json                      coordinator: running|finished|failed|timeout,
                                         start/end, exit code, attempts
        eval/step-<NNNNN>/               trainer: eval weights, every 50 steps + final step
            ...weights...                (config.json, *.safetensors, tokenizer files)
            COMPLETE                     trainer: written LAST; JSON {"step", "final", "files"}
        recovery/checkpoint-<N>/         trainer: full recovery state (model, optimizer,
            ...state...                  scheduler, RNG, trainer state)
            COMPLETE                     trainer: written LAST
        recovery/.tmp-checkpoint-<N>/    trainer (async recovery): checkpoint-<N>/ being written,
                                         plus one .rank-<K>.done per rank; never a checkpoint
        logs/attempt-<k>.log|.exit       coordinator: stdout/stderr and exit code per attempt
        TRANSFERRED                      coordinator: written after the sync command succeeds
    <runs>/_coordinator/
        <node>.jsonl                     coordinator event log (append-only)
        <node>.idle                      coordinator: gate passed, node safe to release
        <node>.eval_gpus.json            coordinator -> watcher GPU lease
        <node>.watcher.json              watcher -> coordinator state / lease ack
    <evals>/<run_id>/
        step-<NNNNN>.jsonl               eval script: per-sample results
        step-<NNNNN>.done                watcher: written after a successful eval
        step-<NNNNN>.failed              watcher: retries exhausted (blocks the gate)
        step-<NNNNN>.claim               watcher: lock so two watchers never share a step
        logs/                            watcher: eval logs

Rules:

* ``NNNNN`` is the step zero-padded to 5 digits (``step-00050``).
* A directory without ``COMPLETE`` is in progress (or a partial write) and is
  ignored by every consumer.
* ``COMPLETE`` content is a JSON object. ``final: true`` marks the final eval
  step. ``files`` optionally lists the relative file names that must exist.
  An empty ``COMPLETE`` is accepted and read as ``{}`` (non-final).
* Nothing under ``eval/`` or ``<evals>/`` is ever deleted. Only complete
  recovery checkpoints are pruned, and only after the run's final eval step
  passes `check_final_integrity`, and only with ``pruning_enabled``.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPLETE = "COMPLETE"
STATUS_FILE = "status.json"
TRANSFERRED = "TRANSFERRED"
COORDINATOR_DIRNAME = "_coordinator"
TERMINAL_STATUSES = frozenset({"finished", "failed", "timeout"})
DEFAULT_REQUIRED_EVAL_FILES = ("config.json", "*.safetensors")
TRACES_SUFFIX = ".traces.jsonl"  # per-sample traces next to each eval records file

_STEP_RE = re.compile(r"^step-(\d+)$")
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
RECOVERY_TMP_PREFIX = ".tmp-checkpoint-"  # async recovery staging dirs; never match _CKPT_RE
_TMP_RE = re.compile(r"^\.tmp-checkpoint-(\d+)$")


# ---------------------------------------------------------------- paths


def step_name(step: int) -> str:
    return f"step-{step:05d}"


def run_dir(runs_root: Path, run_id: str) -> Path:
    return Path(runs_root) / run_id


def eval_step_dir(rdir: Path, step: int) -> Path:
    return Path(rdir) / "eval" / step_name(step)


def recovery_root(rdir: Path) -> Path:
    return Path(rdir) / "recovery"


def recovery_dir(rdir: Path, number: int) -> Path:
    return recovery_root(rdir) / f"checkpoint-{number}"


def recovery_tmp_dir(rdir: Path, number: int) -> Path:
    """Staging dir of an asynchronous recovery checkpoint (`train/async_recovery.py`)."""
    return recovery_root(rdir) / f"{RECOVERY_TMP_PREFIX}{number}"


def status_path(rdir: Path) -> Path:
    return Path(rdir) / STATUS_FILE


def transferred_path(rdir: Path) -> Path:
    return Path(rdir) / TRANSFERRED


def coordinator_dir(runs_root: Path) -> Path:
    return Path(runs_root) / COORDINATOR_DIRNAME


def eval_run_dir(evals_root: Path, run_id: str) -> Path:
    return Path(evals_root) / run_id


def eval_output_path(evals_root: Path, run_id: str, step: int) -> Path:
    return eval_run_dir(evals_root, run_id) / f"{step_name(step)}.jsonl"


def eval_traces_path(records_path: Path) -> Path:
    """The trace file next to an eval records file: ``step-NNNNN.jsonl`` -> ``step-NNNNN.traces.jsonl``."""
    records_path = Path(records_path)
    stem = records_path.name.removesuffix(".jsonl")
    return records_path.with_name(f"{stem}{TRACES_SUFFIX}")


def eval_done_path(evals_root: Path, run_id: str, step: int) -> Path:
    return eval_run_dir(evals_root, run_id) / f"{step_name(step)}.done"


def eval_failed_path(evals_root: Path, run_id: str, step: int) -> Path:
    return eval_run_dir(evals_root, run_id) / f"{step_name(step)}.failed"


# ---------------------------------------------------------------- small I/O


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_marker(path: Path) -> dict[str, Any]:
    """Read a marker file. Empty or non-JSON content reads as ``{}``."""
    data = read_json(path, default={})
    return data if isinstance(data, dict) else {}


def read_status(rdir: Path) -> dict[str, Any] | None:
    data = read_json(status_path(rdir))
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- eval checkpoints


@dataclass(frozen=True)
class EvalCheckpoint:
    run_id: str
    step: int
    path: Path
    final: bool
    files: tuple[str, ...] = ()


def list_eval_checkpoints(rdir: Path) -> list[EvalCheckpoint]:
    """Complete eval checkpoints of a run, sorted by step."""
    rdir = Path(rdir)
    root = rdir / "eval"
    if not root.is_dir():
        return []
    out = []
    for child in root.iterdir():
        m = _STEP_RE.match(child.name)
        if not m or not child.is_dir() or not (child / COMPLETE).is_file():
            continue
        marker = read_marker(child / COMPLETE)
        files = marker.get("files") or ()
        out.append(
            EvalCheckpoint(
                run_id=rdir.name,
                step=int(m.group(1)),
                path=child,
                final=bool(marker.get("final", False)),
                files=tuple(str(f) for f in files),
            )
        )
    return sorted(out, key=lambda c: c.step)


def final_eval_checkpoint(rdir: Path) -> EvalCheckpoint | None:
    finals = [c for c in list_eval_checkpoints(rdir) if c.final]
    return finals[-1] if finals else None


# ---------------------------------------------------------------- recovery checkpoints


@dataclass(frozen=True)
class RecoveryCheckpoint:
    number: int
    path: Path
    complete: bool


def list_recovery_checkpoints(rdir: Path) -> list[RecoveryCheckpoint]:
    """All recovery checkpoint dirs (complete or not), sorted by number."""
    root = recovery_root(rdir)
    if not root.is_dir():
        return []
    out = []
    for child in root.iterdir():
        m = _CKPT_RE.match(child.name)
        if m and child.is_dir() and not child.is_symlink():
            out.append(RecoveryCheckpoint(int(m.group(1)), child, (child / COMPLETE).is_file()))
    return sorted(out, key=lambda c: c.number)


def list_recovery_tmp_dirs(rdir: Path) -> list[tuple[int, Path]]:
    """Async-recovery staging dirs ``recovery/.tmp-checkpoint-<N>`` as (N, path), sorted by N."""
    root = recovery_root(rdir)
    if not root.is_dir():
        return []
    out = []
    for child in root.iterdir():
        m = _TMP_RE.match(child.name)
        if m and child.is_dir() and not child.is_symlink():
            out.append((int(m.group(1)), child))
    return sorted(out)


def latest_recovery_checkpoint(rdir: Path) -> RecoveryCheckpoint | None:
    complete = [c for c in list_recovery_checkpoints(rdir) if c.complete]
    return complete[-1] if complete else None


# ---------------------------------------------------------------- integrity + pruning


@dataclass
class IntegrityResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    checkpoint: EvalCheckpoint | None = None


def _non_empty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def check_final_integrity(
    rdir: Path, required_files: tuple[str, ...] | list[str] = DEFAULT_REQUIRED_EVAL_FILES
) -> IntegrityResult:
    """Final eval step exists with COMPLETE and its expected files are present.

    Expected files are the ``files`` listed in COMPLETE (each must exist and be
    non-empty) plus every glob in ``required_files`` (each must match at least
    one non-empty file in the step dir).
    """
    final = final_eval_checkpoint(rdir)
    if final is None:
        return IntegrityResult(False, ["no complete final eval step (COMPLETE with final: true)"])
    reasons = []
    base = final.path.resolve()
    for rel in final.files:
        target = (final.path / rel).resolve()
        if base not in target.parents:
            reasons.append(f"listed file escapes step dir: {rel}")
        elif not _non_empty_file(target):
            reasons.append(f"listed file missing or empty: {rel}")
    names = [p.name for p in final.path.iterdir() if _non_empty_file(p)]
    for pattern in required_files:
        if not fnmatch.filter(names, pattern):
            reasons.append(f"no non-empty file matches {pattern!r}")
    return IntegrityResult(not reasons, reasons, final)


@dataclass
class PruneResult:
    pruned: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)
    refused: str | None = None


def prune_recovery_checkpoints(
    rdir: Path,
    keep: int = 2,
    *,
    enabled: bool = False,
    required_files: tuple[str, ...] | list[str] = DEFAULT_REQUIRED_EVAL_FILES,
) -> PruneResult:
    """Delete all but the newest ``keep`` complete recovery checkpoints.

    Refuses (deletes nothing) unless ``enabled`` is True and the run's final
    artifacts pass `check_final_integrity`. Incomplete checkpoint dirs and
    everything under ``eval/`` are never touched.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    rdir = Path(rdir)
    if not enabled:
        return PruneResult(refused="pruning disabled (enable only after resume is verified)")
    integrity = check_final_integrity(rdir, required_files)
    if not integrity.ok:
        return PruneResult(refused="integrity check failed: " + "; ".join(integrity.reasons))
    complete = [c for c in list_recovery_checkpoints(rdir) if c.complete]
    doomed, kept = complete[:-keep], complete[-keep:]
    root = recovery_root(rdir).resolve()
    result = PruneResult(kept=[c.path for c in kept])
    for ckpt in doomed:
        path = ckpt.path
        # Belt and braces: only real checkpoint-<N> dirs directly under recovery/.
        if path.is_symlink() or path.resolve().parent != root or not _CKPT_RE.match(path.name):
            continue
        shutil.rmtree(path)
        result.pruned.append(path)
    return result
