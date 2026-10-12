"""Trainer callbacks for `train/grpo.py`: per-step metrics, eval weights, recovery markers, pruning.

No torch or transformers import, so CI can drive the callback with a fake trainer loop. The
callback is duck-typed against `transformers.TrainerCallback` (the `CallbackHandler` calls
``getattr(callback, event)(args, state, control, **kwargs)``); `train/grpo.py` mixes it into a real
`TrainerCallback` subclass so every other event is a no-op.

Artifacts (`train/LAYOUT.md`, paths from `train/checkpoints.py`):

* **Eval weights** ``<run>/eval/step-NNNNN/`` every ``eval_every`` steps and at the final step
  (``state.global_step >= state.max_steps``), written in ``on_step_end`` (before the Trainer logs
  and saves). Every rank calls the injected ``save_eval_weights`` (ZeRO-3 gathers collectively);
  then a barrier; then the main rank writes ``COMPLETE`` **last**:
  ``{"step", "final", "files"}``. A partial dir (no ``COMPLETE``) from a crash is replaced; a
  complete one is never touched.
* **Recovery checkpoints** ``<run>/recovery/checkpoint-N/`` are written by the Trainer itself
  (``output_dir=<run>/recovery``, ``save_steps``). ``on_save`` fires after the Trainer has written
  everything; after a barrier the main rank writes ``COMPLETE`` last. With
  ``checkpoints.async_recovery`` (``async_recovery=True`` here) a background writer
  (`train/async_recovery.py`) writes ``COMPLETE`` and prunes instead, and ``on_save`` does nothing.
* **Rolling prune** (`rolling_prune`), only with ``pruning_enabled``: after a new ``COMPLETE``,
  delete complete recovery checkpoints older than the newest ``keep_last`` complete ones. Incomplete
  dirs and ``eval/`` are never touched.
* **Training log** ``<run>/train_metrics.jsonl`` (`TrainMetricsWriter`), one row per optimizer step,
  append-only, main rank only.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from train.checkpoints import (
    COMPLETE,
    atomic_write_json,
    eval_step_dir,
    list_recovery_checkpoints,
    list_recovery_tmp_dirs,
    recovery_dir,
    recovery_root,
)

TRAIN_METRICS = "train_metrics.jsonl"

# TRL 1.15 GRPOTrainer log keys -> train_metrics.jsonl keys. `reward` is TRL's weighted reward:
# weight 1 on the run's own grader and 0 on the other (`train.rewards.reward_setup`), i.e. the mean
# over all generations under the run's own reward, before advantage normalisation.
METRIC_KEYS: dict[str, str] = {
    "reward": "reward",
    "reward_std": "reward_std",
    "rewards/robust_reward/mean": "robust_reward_mean",
    "rewards/leaky_reward/mean": "leaky_reward_mean",
    "frac_reward_zero_std": "dead_group_share",
    "entropy": "entropy",
    "completions/mean_length": "completion_length",
    "completions/clipped_ratio": "completion_clipped_ratio",
    "loss": "loss",
    "grad_norm": "grad_norm",
    "learning_rate": "learning_rate",
    "step_time": "trl_step_time_s",
    # Sampler (vLLM) vs trainer disagreement on the sampled tokens. Spikes preceded every toy collapse.
    "sampling/sampling_logp_difference/mean": "sampler_logp_diff_mean",
    "sampling/sampling_logp_difference/max": "sampler_logp_diff_max",
    "sampling/importance_sampling_ratio/min": "is_ratio_min",
}


# ---------------------------------------------------------------- train_metrics.jsonl


def metrics_row(step: int, logs: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One train_metrics.jsonl row from a TRL log dict. Unknown keys are dropped; NaN becomes null."""
    row: dict[str, Any] = {"step": int(step)}
    for src, dst in METRIC_KEYS.items():
        value = logs.get(src)
        if value is None:
            continue
        f = float(value)
        row[dst] = f if math.isfinite(f) else None
    if extra:
        row.update(extra)
    return row


class TrainMetricsWriter:
    """Append-only writer for ``train_metrics.jsonl`` (eval/compare.py: the last line for a step wins)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, row: Mapping[str, Any]) -> None:
        if "step" not in row or "reward" not in row:
            raise ValueError(f"a train_metrics row needs step and reward: {dict(row)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")
            f.flush()


# ---------------------------------------------------------------- pruning


def rolling_prune(rdir: Path, keep: int = 2, *, enabled: bool = False) -> list[Path]:
    """Delete complete recovery checkpoints older than the newest ``keep`` complete ones.

    Runs during training, so unlike `train.checkpoints.prune_recovery_checkpoints` it does not wait
    for the final eval step. An older checkpoint is deleted only once ``keep`` newer ones carry
    ``COMPLETE``. Incomplete checkpoint dirs and everything under ``eval/`` are never touched.
    Returns the deleted paths (none unless ``enabled``).
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    if not enabled:
        return []
    complete = [c for c in list_recovery_checkpoints(rdir) if c.complete]
    root = recovery_root(rdir).resolve()
    pruned = []
    for ckpt in complete[:-keep]:
        path = ckpt.path
        # Only real checkpoint-<N> dirs directly under recovery/.
        if path.is_symlink() or path.resolve().parent != root:
            continue
        shutil.rmtree(path)
        pruned.append(path)
    return pruned


def remove_incomplete_recovery(rdir: Path, keep: Path | None = None) -> list[Path]:
    """Remove recovery dirs without ``COMPLETE`` (partial writes from a crashed attempt).

    Also removes every async-recovery staging dir ``recovery/.tmp-checkpoint-N`` (a write that never
    finished). Call it only when no writer of this run is alive, i.e. at the start of a worker.
    """
    removed = []
    for ckpt in list_recovery_checkpoints(rdir):
        if ckpt.complete or (keep is not None and ckpt.path.resolve() == Path(keep).resolve()):
            continue
        shutil.rmtree(ckpt.path)
        removed.append(ckpt.path)
    for _, tmp in list_recovery_tmp_dirs(rdir):
        shutil.rmtree(tmp)
        removed.append(tmp)
    return removed


def list_files(directory: Path) -> list[str]:
    """Relative names of the non-hidden files under `directory` (the ``files`` list of ``COMPLETE``)."""
    directory = Path(directory)
    return sorted(
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )


# ---------------------------------------------------------------- the callback


class ArtifactCallback:
    """Eval weights, recovery ``COMPLETE`` markers, rolling prune and train_metrics.jsonl.

    Injected hooks keep this testable without a GPU:

    * ``save_eval_weights(path)``: write config, weights and tokenizer into ``path``. Called on every
      rank (ZeRO-3 needs all ranks to gather); only the main rank writes files.
    * ``barrier()``: wait for every rank (no-op in a single process).
    * ``is_main``: whether this process writes markers and metrics.
    * ``write_marker(path, obj)``: defaults to an atomic JSON write.
    * ``async_recovery``: recovery checkpoints are finished by `train.async_recovery.AsyncRecovery`
      (``COMPLETE``, prune), so ``on_save`` is a no-op.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        eval_every: int,
        keep_last: int = 2,
        pruning_enabled: bool = False,
        save_eval_weights: Callable[[Path], None] | None = None,
        barrier: Callable[[], None] = lambda: None,
        is_main: bool = True,
        marker_extra: Mapping[str, Any] | None = None,
        write_marker: Callable[[Path, Any], None] = atomic_write_json,
        clock: Callable[[], float] = time.perf_counter,
        async_recovery: bool = False,
    ) -> None:
        if eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        self.run_dir = Path(run_dir)
        self.eval_every = eval_every
        self.keep_last = keep_last
        self.pruning_enabled = pruning_enabled
        self.save_eval_weights = save_eval_weights
        self.barrier = barrier
        self.is_main = is_main
        self.marker_extra = dict(marker_extra or {})
        self.write_marker = write_marker
        self.clock = clock
        self.async_recovery = async_recovery
        self.metrics = TrainMetricsWriter(self.run_dir / TRAIN_METRICS)
        self.extra_metrics: Callable[[], Mapping[str, Any]] | None = None
        self._t0: float | None = None
        self._step_time: float | None = None
        self.eval_steps_written: list[int] = []
        self.recovery_marked: list[int] = []
        self.pruned: list[Path] = []

    # -- TrainerCallback events ------------------------------------------------------------

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._t0 = self.clock()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._t0 is not None:
            self._step_time = self.clock() - self._t0
        step = int(state.global_step)
        final = bool(state.max_steps) and step >= int(state.max_steps)
        if step % self.eval_every == 0 or final:
            self.write_eval_step(step, final=final)
        if self.is_main:  # live progress bar (lightning-jobs progress.py)
            print(f"\nPROGRESS {step}/{int(state.max_steps)}", flush=True)  # own line: tqdm leaves the cursor mid-line

    def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **kwargs: Any):
        logs = logs or {}
        if "reward" not in logs or not self.is_main:  # skip the end-of-training summary log
            return
        extra: dict[str, Any] = {"step_time_s": self._step_time}
        if self.extra_metrics is not None:
            extra.update(self.extra_metrics())
        self.metrics.append(metrics_row(state.global_step, logs, extra))

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.async_recovery:  # the background writer marks and prunes once the write is durable
            return
        step = int(state.global_step)
        ckpt = recovery_dir(self.run_dir, step)
        self.barrier()  # every rank has finished writing its shards
        if self.is_main:
            if not ckpt.is_dir():
                raise RuntimeError(f"Trainer saved no recovery checkpoint at {ckpt}; check output_dir")
            self.write_marker(ckpt / COMPLETE, {"step": step, **self.marker_extra})
            self.recovery_marked.append(step)
            self.pruned.extend(rolling_prune(self.run_dir, self.keep_last, enabled=self.pruning_enabled))
        self.barrier()

    # -- eval weights ----------------------------------------------------------------------

    def write_eval_step(self, step: int, *, final: bool) -> Path:
        out = eval_step_dir(self.run_dir, step)
        skip = (out / COMPLETE).is_file()  # same on every rank: shared filesystem, checked before any write
        self.barrier()
        if skip:
            return out
        if self.is_main and out.exists():
            shutil.rmtree(out)  # partial write from a crashed attempt
        self.barrier()
        if self.save_eval_weights is None:
            raise RuntimeError("ArtifactCallback.save_eval_weights is not bound")
        self.save_eval_weights(out)
        self.barrier()
        if self.is_main:
            files = list_files(out)
            if not files:
                raise RuntimeError(f"eval weights for step {step} wrote no files to {out}")
            self.write_marker(out / COMPLETE, {"step": step, "final": final, "files": files, **self.marker_extra})
            self.eval_steps_written.append(step)
        self.barrier()
        return out
