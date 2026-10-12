"""Asynchronous recovery checkpoints (``checkpoints.async_recovery`` in `configs/grpo.yaml`).

A synchronous recovery checkpoint stops training while ~22 GB (2B) or ~55 GB (4B ZeRO-3) go to a
shared network disk. Here the Trainer's own ``_save_checkpoint`` still runs on every rank at the
save step, so every file, name and collective is exactly the stock one (HF `Trainer`, TRL
`GRPOTrainer`, DeepSpeed ``engine.save_checkpoint``), but the two heavy writers are deferred:

1. **Decide** (main thread, every rank): if a previous write is still in flight on *any* rank (one
   ``all_reduce`` of a local busy flag), skip this checkpoint everywhere and log ``recovery_skipped``.
2. **Snapshot** (main thread): run ``_save_checkpoint`` into ``recovery/.tmp-checkpoint-N/`` with
   `DeferredWrites` active. ``torch.save(obj, path)`` and safetensors ``save_file(tensors, path)``
   copy their tensors into reusable CPU buffers (pinned when CUDA is available; storage sharing
   kept) and queue the write. Small files (config, tokenizer, trainer state, DeepSpeed ``latest``)
   are written as usual. Training resumes as soon as the copies land.
3. **Write** (one background thread per rank, never a collective): write the queued files, fsync,
   then ``.tmp-checkpoint-N/.rank-K.done``. The main rank's thread waits for every rank's done file,
   fsyncs, renames ``.tmp-checkpoint-N/checkpoint-N`` to ``recovery/checkpoint-N`` (atomic), writes
   ``COMPLETE`` (same content as the synchronous path), runs the rolling prune and removes the
   staging dir. A rank failure (``.rank-K.failed``) or a timeout abandons that checkpoint: no
   ``COMPLETE``, an ``.abandoned`` file instead; the staging dir is removed before the next
   snapshot or at the next start. The other ranks' threads stay alive until the main rank has
   settled (``COMPLETE`` written, or ``.abandoned``), so "busy" means the same on every rank and
   the ranks finish their final join together.

At most one snapshot is in flight per rank. Events (``recovery_snapshot``, ``recovery_written``,
``recovery_skipped``, ``recovery_failed``, with seconds and bytes) go to ``recovery_events.jsonl``
(main rank) and stdout. `train/grpo.py` joins the writer before the process exits.

`AsyncRecovery` (the coordinator) is torch-free so CI tests it with fakes; torch is imported lazily
by `DeferredWrites`, `PinnedPool`, `Snapshotter`, `any_rank_busy` and the Trainer mixin.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from train.callbacks import rolling_prune
from train.checkpoints import (
    COMPLETE,
    atomic_write_json,
    list_recovery_tmp_dirs,
    recovery_dir,
    recovery_root,
    recovery_tmp_dir,
)

EVENTS_FILE = "recovery_events.jsonl"
DONE_TIMEOUT_S = 3600.0  # main rank: wait this long for every rank's done file
JOIN_TIMEOUT_S = 3 * 3600.0  # end of training: wait this long for the in-flight write
CRASH_JOIN_TIMEOUT_S = 600.0  # training raised: give the in-flight write a chance to finish
_print_flush = functools.partial(print, flush=True)


def done_path(tmp: Path, rank: int) -> Path:
    return Path(tmp) / f".rank-{rank}.done"


def failed_path(tmp: Path, rank: int) -> Path:
    return Path(tmp) / f".rank-{rank}.failed"


def abandoned_path(tmp: Path) -> Path:
    return Path(tmp) / ".abandoned"


def fsync_path(path: Path) -> None:
    """fsync a file or a directory (best effort for directories: some filesystems refuse)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        if not Path(path).is_dir():
            raise
    finally:
        os.close(fd)


def fsync_tree(root: Path) -> None:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            fsync_path(Path(dirpath) / name)
        fsync_path(Path(dirpath))


@dataclass
class DeferredWrite:
    """One file whose bytes the background thread writes: ``write()`` creates ``path``."""

    path: Path
    write: Callable[[], None]
    nbytes: int = 0


class AsyncRecovery:
    """Per-rank coordinator of asynchronous recovery checkpoints (see the module docstring).

    ``save(step, snapshot)`` is called on every rank at the save step from the training thread.
    ``snapshot(tmp)`` must run the stock checkpoint code into ``tmp`` (so the checkpoint lands in
    ``tmp/checkpoint-<step>``) and return the deferred writes. Injected hooks keep this testable:
    ``any_busy(local)`` is the collective OR of the per-rank busy flags (identity in one process).
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        rank: int = 0,
        world_size: int = 1,
        is_main: bool = True,
        keep_last: int = 2,
        pruning_enabled: bool = False,
        marker_extra: Mapping[str, Any] | None = None,
        any_busy: Callable[[bool], bool] = lambda local: local,
        write_marker: Callable[[Path, Any], None] = atomic_write_json,
        done_timeout: float = DONE_TIMEOUT_S,
        poll: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
        log: Callable[[str], None] = _print_flush,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"bad rank {rank} for world size {world_size}")
        self.run_dir = Path(run_dir)
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main
        self.keep_last = keep_last
        self.pruning_enabled = pruning_enabled
        self.marker_extra = dict(marker_extra or {})
        self.any_busy = any_busy
        self.write_marker = write_marker
        self.done_timeout = done_timeout
        self.poll = poll
        self.clock = clock
        self.log = log
        self.events_path = self.run_dir / EVENTS_FILE
        self._events_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.started: list[int] = []
        self.skipped: list[int] = []
        self.completed: list[int] = []
        self.failed: list[int] = []
        self.pruned: list[Path] = []

    # -- main thread ---------------------------------------------------------------------

    def busy(self) -> bool:
        """Whether this rank's previous write is still in flight."""
        return self._thread is not None and self._thread.is_alive()

    def save(self, step: int, snapshot: Callable[[Path], list[DeferredWrite]]) -> bool:
        """Snapshot and start the background write of recovery checkpoint ``step``, or skip it.

        Args:
            step: The global step being checkpointed.
            snapshot: Runs the stock checkpoint code into the staging dir it is given and returns the
                deferred writes.

        Returns:
            True if a write was started, False if the checkpoint was skipped on every rank.
        """
        if self.any_busy(self.busy()):  # collective: the same answer on every rank
            self.skipped.append(step)
            if self.is_main:
                self._event({"event": "recovery_skipped", "step": step})
            return False
        if self.is_main:
            self.remove_staging(keep_step=step)  # nothing is in flight on any rank
        tmp = recovery_tmp_dir(self.run_dir, step)
        (tmp / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
        t0 = self.clock()
        writes = snapshot(tmp)
        snapshot_s = self.clock() - t0
        nbytes = sum(w.nbytes for w in writes)
        self.started.append(step)
        if self.is_main:
            self._event({"event": "recovery_snapshot", "step": step, "snapshot_s": snapshot_s, "bytes": nbytes})
        self._thread = threading.Thread(
            target=self._finish, args=(step, tmp, writes, snapshot_s), name=f"recovery-{step}", daemon=True
        )
        self._thread.start()
        return True

    def join(self, timeout: float | None = JOIN_TIMEOUT_S) -> bool:
        """Wait for the in-flight write. Returns False if it is still running after ``timeout``."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        if thread.is_alive():
            self.log(f"[recovery] rank {self.rank}: write still in flight after {timeout}s; giving up on it")
            return False
        return True

    # -- background thread ---------------------------------------------------------------

    def _finish(self, step: int, tmp: Path, writes: list[DeferredWrite], snapshot_s: float) -> None:
        t0 = self.clock()
        try:
            nbytes = 0
            for w in writes:
                w.write()
                fsync_path(w.path)
                nbytes += w.nbytes
            write_s = self.clock() - t0
            mine = {"rank": self.rank, "step": step, "snapshot_s": snapshot_s, "write_s": write_s, "bytes": nbytes}
            atomic_write_json(done_path(tmp, self.rank), mine)
            fsync_path(done_path(tmp, self.rank))
        except Exception as exc:
            self.failed.append(step)
            reason = f"rank {self.rank} write failed: {type(exc).__name__}: {exc}"
            with contextlib.suppress(OSError):
                atomic_write_json(failed_path(tmp, self.rank), {"rank": self.rank, "step": step, "reason": reason})
            self._fail(step, reason, snapshot_s, tmp)
            return
        self.log(f"[recovery] rank {self.rank} step {step}: snapshot {snapshot_s:.1f}s, write {write_s:.1f}s")
        if not self.is_main:
            self._wait_for_main(step, tmp)
            return
        try:
            self._complete(step, tmp, mine, t0)
        except Exception as exc:
            self.failed.append(step)
            self._fail(step, f"finishing failed: {type(exc).__name__}: {exc}", snapshot_s, tmp)

    def _complete(self, step: int, tmp: Path, mine: dict[str, Any], t0: float) -> None:
        ranks = self._wait_for_ranks(step, tmp)
        if ranks is None:
            return
        wait_s = self.clock() - t0 - mine["write_s"]
        staged = tmp / f"checkpoint-{step}"
        fsync_tree(staged)
        final = recovery_dir(self.run_dir, step)
        if final.exists():  # left by an earlier attempt that reached this step; replaced like the sync path does
            shutil.rmtree(final)
        os.rename(staged, final)
        fsync_path(recovery_root(self.run_dir))
        self.write_marker(final / COMPLETE, {"step": step, **self.marker_extra})
        self.completed.append(step)
        pruned = rolling_prune(self.run_dir, self.keep_last, enabled=self.pruning_enabled)
        self.pruned.extend(pruned)
        shutil.rmtree(tmp, ignore_errors=True)
        self._event(
            {
                "event": "recovery_written",
                "step": step,
                "snapshot_s": mine["snapshot_s"],
                "write_s": mine["write_s"],
                "max_rank_write_s": max(float(r.get("write_s") or 0.0) for r in ranks),
                "wait_s": wait_s,
                "total_s": self.clock() - t0,
                "bytes": sum(int(r.get("bytes") or 0) for r in ranks),
                "pruned": [p.name for p in pruned],
            }
        )

    def _wait_for_ranks(self, step: int, tmp: Path) -> list[dict[str, Any]] | None:
        """Every rank's done-file content, or None (logged) if a rank failed or the wait timed out."""
        deadline = self.clock() + self.done_timeout
        while True:
            if all(done_path(tmp, r).is_file() for r in range(self.world_size)):
                return [_read_json(done_path(tmp, r)) for r in range(self.world_size)]
            bad = [r for r in range(self.world_size) if failed_path(tmp, r).is_file()]
            if bad:
                self.failed.append(step)
                self._fail(step, f"rank(s) {bad} failed to write", None, tmp)
                return None
            if self.clock() > deadline:
                missing = [r for r in range(self.world_size) if not done_path(tmp, r).is_file()]
                self.failed.append(step)
                self._fail(step, f"timed out after {self.done_timeout}s waiting for rank(s) {missing}", None, tmp)
                return None
            time.sleep(self.poll)

    def _wait_for_main(self, step: int, tmp: Path) -> None:
        """Non-main ranks: stay busy until the main rank has completed or abandoned this checkpoint."""
        deadline = self.clock() + self.done_timeout + 60.0
        complete = recovery_dir(self.run_dir, step) / COMPLETE
        while tmp.exists() and not abandoned_path(tmp).is_file() and not complete.is_file():
            if self.clock() > deadline:
                self.log(f"[recovery] rank {self.rank} step {step}: main rank has not settled; not waiting longer")
                return
            time.sleep(self.poll)

    # -- helpers ---------------------------------------------------------------------------

    def remove_staging(self, keep_step: int | None = None) -> list[Path]:
        """Remove staging dirs of abandoned checkpoints. Only when no writer is alive on any rank."""
        removed = []
        for number, path in list_recovery_tmp_dirs(self.run_dir):
            if number != keep_step:
                shutil.rmtree(path, ignore_errors=True)
                self.log(f"[recovery] removed abandoned staging dir {path}")
                removed.append(path)
        return removed

    def _fail(self, step: int, reason: str, snapshot_s: float | None, tmp: Path) -> None:
        self.log(f"[recovery] step {step}: checkpoint abandoned (no {COMPLETE}): {reason}")
        if not self.is_main:
            return
        try:
            if tmp.is_dir():
                atomic_write_json(abandoned_path(tmp), {"step": step, "reason": reason})
        except OSError:
            pass
        self._event({"event": "recovery_failed", "step": step, "reason": reason, "snapshot_s": snapshot_s})

    def _event(self, row: dict[str, Any]) -> None:
        row = {**row, "ts": time.time()}
        self.log(f"[recovery] {json.dumps(row, sort_keys=True)}")
        with self._events_lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- snapshot (torch, imported lazily)


class PinnedPool:
    """Reusable CPU byte buffers for snapshots, pinned when CUDA is available.

    At most one snapshot is in flight per rank, so `reset` before each snapshot frees every buffer;
    the same storages come back every checkpoint, so after the first one nothing is allocated.
    """

    def __init__(self, pin: bool | None = None) -> None:
        self.pin = pin
        self._free: dict[int, list[Any]] = {}
        self._used: list[tuple[int, Any]] = []
        self.allocated_bytes = 0

    def reset(self) -> None:
        for nbytes, buf in self._used:
            self._free.setdefault(nbytes, []).append(buf)
        self._used = []

    def take(self, nbytes: int) -> Any:
        free = self._free.get(nbytes)
        buf = free.pop() if free else self._alloc(nbytes)
        self._used.append((nbytes, buf))
        return buf

    def _alloc(self, nbytes: int) -> Any:
        import torch

        if self.pin is None:
            self.pin = torch.cuda.is_available()
        if self.pin and nbytes:
            try:
                buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True).untyped_storage()
                self.allocated_bytes += nbytes
                return buf
            except RuntimeError as exc:  # pinned memory exhausted or refused: pageable from now on
                print(f"[recovery] pinned allocation failed ({exc}); using pageable memory", flush=True)
                self.pin = False
        self.allocated_bytes += nbytes
        return torch.empty(nbytes, dtype=torch.uint8).untyped_storage()


_ATOMS = (type(None), bool, int, float, complex, str, bytes)


class Snapshotter:
    """Copies objects for a later ``torch.save``: tensors into `PinnedPool` buffers, containers
    rebuilt with the same types, other objects deep-copied (or shared if they refuse).

    Within one `copy` call storages are memoised by address, so tensors that share a storage (views,
    tied weights) still share one in the copy, and each storage is copied in full, as ``torch.save``
    writes it. The memo never outlives a call: once the caller drops its tensors, their addresses can
    be reused by new ones.
    """

    def __init__(self, pool: PinnedPool) -> None:
        self.pool = pool
        self._memo: dict[tuple[str, int, int], Any] = {}
        self.copied_bytes = 0
        self._devices: set[Any] = set()  # CUDA devices with copies in flight

    def copy(self, obj: Any) -> Any:
        """A copy of ``obj`` that later in-place updates of its tensors do not reach."""
        self._memo = {}
        try:
            return self._copy(obj)
        finally:
            self._memo = {}

    def _copy(self, obj: Any) -> Any:
        import numpy as np
        import torch

        if isinstance(obj, torch.Tensor):
            return self._tensor(obj)
        if isinstance(obj, (*_ATOMS, torch.dtype, torch.device, torch.Size)):
            return obj
        if isinstance(obj, dict):
            new = copy.copy(obj)  # keeps the type and attributes (OrderedDict._metadata, defaultdict factory)
            for key, value in obj.items():
                new[key] = self._copy(value)
            return new
        if isinstance(obj, list):
            items = [self._copy(v) for v in obj]
            if type(obj) is list:
                return items
            new = copy.copy(obj)
            new[:] = items
            return new
        if isinstance(obj, tuple):
            items = [self._copy(v) for v in obj]
            if hasattr(obj, "_fields"):  # namedtuple
                return type(obj)(*items)
            return tuple(items) if type(obj) is tuple else type(obj)(items)
        if isinstance(obj, np.ndarray):
            return obj.copy()
        try:
            return copy.deepcopy(obj)
        except Exception:  # unpicklable handles (process groups, locks): shared, as torch.save sees them now
            return obj

    def _tensor(self, t: Any) -> Any:
        import torch

        if t.device.type == "meta":  # no data; torch.save writes it as is
            return t.detach()
        if t.layout != torch.strided:
            return t.detach().to("cpu", copy=True)
        src = t.untyped_storage()
        key = (str(src.device), src.data_ptr(), src.nbytes())
        dst = self._memo.get(key)
        if dst is None:
            dst = self.pool.take(src.nbytes())
            if src.nbytes():
                dst.copy_(src, non_blocking=True)
                if src.device.type == "cuda":
                    self._devices.add(src.device)
            self._memo[key] = dst
            self.copied_bytes += src.nbytes()
        out = torch.empty((0,), dtype=t.dtype).set_(dst, t.storage_offset(), t.size(), t.stride())
        if isinstance(t, torch.nn.Parameter):
            return torch.nn.Parameter(out, requires_grad=t.requires_grad)
        return out

    def synchronize(self) -> None:
        """Wait for the device-to-host copies (pinned copies are asynchronous)."""
        if self._devices:
            import torch

            for device in self._devices:
                torch.cuda.synchronize(device)
            self._devices = set()


class DeferredWrites:
    """Context manager: while active, ``torch.save`` and safetensors ``save_file`` called with a path
    from *this* thread snapshot their arguments and queue the write in ``items`` instead.

    Patched: ``torch.save``, ``safetensors.torch.save_file`` and transformers'
    ``modeling_utils.safe_save_file`` (its own binding). Calls from other threads, or with a file
    object, go straight to the originals.
    """

    def __init__(self, pool: PinnedPool) -> None:
        self.snap = Snapshotter(pool)
        self.items: list[DeferredWrite] = []
        self._restore: list[tuple[Any, str, Any]] = []
        self._active = False

    def __enter__(self) -> DeferredWrites:
        import safetensors.torch
        import torch

        try:  # imported before patching: it binds `save_file` by name at import time
            import transformers.modeling_utils as modeling_utils
        except ImportError:  # pragma: no cover - transformers is always present on GPU nodes
            modeling_utils = None
        owner = threading.get_ident()
        orig_save = torch.save
        orig_st = safetensors.torch.save_file

        def deferred_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> None:
            if not self._active or threading.get_ident() != owner or not isinstance(f, (str, os.PathLike)):
                return orig_save(obj, f, *args, **kwargs)
            before = self.snap.copied_bytes
            copied = self.snap.copy(obj)
            self.snap.synchronize()
            write = functools.partial(orig_save, copied, os.fspath(f), *args, **kwargs)
            self.items.append(DeferredWrite(Path(f), write, self.snap.copied_bytes - before))
            return None

        def deferred_st(tensors: dict[str, Any], filename: Any, metadata: dict[str, str] | None = None) -> None:
            if not self._active or threading.get_ident() != owner:
                return orig_st(tensors, filename, metadata=metadata)
            before = self.snap.copied_bytes
            copied = self.snap.copy(dict(tensors))
            self.snap.synchronize()
            write = functools.partial(orig_st, copied, os.fspath(filename), metadata=metadata)
            self.items.append(DeferredWrite(Path(filename), write, self.snap.copied_bytes - before))
            return None

        self._patch(torch, "save", deferred_save)
        self._patch(safetensors.torch, "save_file", deferred_st)
        if modeling_utils is not None and hasattr(modeling_utils, "safe_save_file"):
            self._patch(modeling_utils, "safe_save_file", deferred_st)
        self._active = True
        return self

    def _patch(self, module: Any, name: str, value: Any) -> None:
        self._restore.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def __exit__(self, *exc: object) -> None:
        self._active = False  # a binding copied during the window (a first-time import) now passes through
        for module, name, value in reversed(self._restore):
            setattr(module, name, value)
        self._restore = []


def any_rank_busy(device: Any) -> Callable[[bool], bool]:
    """Collective OR of the per-rank busy flags (``all_reduce`` MAX); call on the training thread only."""

    def any_busy(local: bool) -> bool:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return local
        flag = torch.tensor([1 if local else 0], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    return any_busy


class AsyncRecoveryTrainerMixin:
    """`transformers.Trainer` mixin: ``_save_checkpoint`` goes through `AsyncRecovery`.

    With ``async_recovery`` unset the Trainer's own ``_save_checkpoint`` runs unchanged. Otherwise
    the stock method (TRL's model card and entropy state included) runs into the staging dir under
    `DeferredWrites`, by pointing ``_get_output_dir`` at it for the duration of the snapshot.
    """

    async_recovery: AsyncRecovery | None = None
    _recovery_output_dir: str | None = None
    _recovery_pool: PinnedPool | None = None

    def _get_output_dir(self, trial: Any) -> str:
        if self._recovery_output_dir is not None:
            return self._recovery_output_dir
        return super()._get_output_dir(trial)  # type: ignore[misc]

    def _save_checkpoint(self, model: Any, trial: Any) -> None:
        recovery = self.async_recovery
        if recovery is None:
            return super()._save_checkpoint(model, trial)  # type: ignore[misc]
        if self._recovery_pool is None:
            self._recovery_pool = PinnedPool()
        pool = self._recovery_pool

        def snapshot(tmp: Path) -> list[DeferredWrite]:
            pool.reset()  # nothing is in flight on this rank: every buffer is free
            self._recovery_output_dir = str(tmp)
            try:
                with DeferredWrites(pool) as deferred:
                    super(AsyncRecoveryTrainerMixin, self)._save_checkpoint(model, trial)  # type: ignore[misc]
            finally:
                self._recovery_output_dir = None
            return deferred.items

        recovery.save(int(self.state.global_step), snapshot)  # type: ignore[attr-defined]
        return None
