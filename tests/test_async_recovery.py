"""train/async_recovery.py without torch: ordering, skip-if-busy, crashes, pruning, final join, flag off."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from train.async_recovery import (
    EVENTS_FILE,
    AsyncRecovery,
    DeferredWrite,
    abandoned_path,
    done_path,
    failed_path,
)
from train.callbacks import ArtifactCallback, remove_incomplete_recovery
from train.checkpoints import (
    COMPLETE,
    atomic_write_json,
    latest_recovery_checkpoint,
    list_recovery_checkpoints,
    list_recovery_tmp_dirs,
    read_marker,
    recovery_dir,
    recovery_root,
    recovery_tmp_dir,
)
from train.grpo import DEFAULT_CONFIG, load_config, parse_args, run_settings
from train.run_manifest import config_identity

HF_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")  # transformers.trainer_utils.get_last_checkpoint
JOIN = 60.0  # upper bound for waits that should succeed; generous so a loaded node (load ~450) does not flake


def wait_for(cond, timeout: float = JOIN) -> bool:
    """Poll ``cond`` until true or ``timeout``; the background writer threads run on their own schedule."""
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


class Writer:
    """Fake checkpoint code for one rank: small files written now, big ones deferred.

    ``gate`` holds the deferred writes back until it is set; ``fail_on`` makes the write of that
    file raise half way.
    """

    def __init__(self, rank: int = 0, gate: threading.Event | None = None, fail_on: str | None = None):
        self.rank = rank
        self.gate = gate
        self.fail_on = fail_on
        self.calls: list[Path] = []

    def snapshot(self, tmp: Path) -> list[DeferredWrite]:
        self.calls.append(tmp)
        step = int(tmp.name.rsplit("-", 1)[1])
        ckpt = tmp / f"checkpoint-{step}"
        if self.rank == 0:
            (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        names = ["model.safetensors", "optimizer.pt"] if self.rank == 0 else []
        names.append(f"rng_state_{self.rank}.pth")
        return [DeferredWrite(ckpt / name, self._writer(ckpt / name, step), 10) for name in names]

    def _writer(self, path: Path, step: int):
        def write() -> None:
            if self.gate is not None:
                assert self.gate.wait(JOIN), "test gate never opened"
            if path.name == self.fail_on:
                path.write_bytes(b"half")
                raise OSError("disk full")
            path.write_bytes(f"{path.name}@{step}".encode())

        return write


def make(run_dir: Path, *, rank=0, world=1, any_busy=None, pruning=False, keep=2, marker=None, timeout=JOIN):
    return AsyncRecovery(
        run_dir,
        rank=rank,
        world_size=world,
        is_main=rank == 0,
        keep_last=keep,
        pruning_enabled=pruning,
        marker_extra={"run_id": "4b-rob-s1"},
        any_busy=any_busy or (lambda local: local),
        write_marker=marker or atomic_write_json,
        done_timeout=timeout,
        poll=0.01,
        log=lambda _msg: None,
    )


def events(run_dir: Path) -> list[dict]:
    path = run_dir / EVENTS_FILE
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def complete_steps(run_dir: Path) -> list[int]:
    return [c.number for c in list_recovery_checkpoints(run_dir) if c.complete]


# ---------------------------------------------------------------- happy path and layout


def test_write_lands_in_checkpoint_dir_with_complete_last(tmp_path):
    rec = make(tmp_path)
    writer = Writer()
    assert rec.save(3, writer.snapshot) is True
    assert rec.join(JOIN)
    ckpt = recovery_dir(tmp_path, 3)
    names = sorted(p.name for p in ckpt.iterdir())
    assert names == [COMPLETE, "model.safetensors", "optimizer.pt", "rng_state_0.pth", "trainer_state.json"]
    assert (ckpt / "optimizer.pt").read_bytes() == b"optimizer.pt@3"
    assert read_marker(ckpt / COMPLETE) == {"step": 3, "run_id": "4b-rob-s1"}  # same marker as the sync path
    assert writer.calls == [recovery_tmp_dir(tmp_path, 3)]
    assert list_recovery_tmp_dirs(tmp_path) == []  # staging dir and done files are gone
    rows = events(tmp_path)
    assert [r["event"] for r in rows] == ["recovery_snapshot", "recovery_written"]
    written = rows[1]
    assert written["step"] == 3 and written["bytes"] == 30
    assert {"snapshot_s", "write_s", "max_rank_write_s", "wait_s", "total_s", "ts"} <= set(written)
    assert rec.completed == [3] and rec.skipped == [] and rec.failed == []


def test_staging_dir_never_looks_like_a_checkpoint(tmp_path):
    gate = threading.Event()
    rec = make(tmp_path)
    rec.save(50, Writer(gate=gate).snapshot)
    tmp = recovery_tmp_dir(tmp_path, 50)
    assert tmp.is_dir() and (tmp / "checkpoint-50").is_dir()
    assert list_recovery_checkpoints(tmp_path) == [] and latest_recovery_checkpoint(tmp_path) is None
    # what HF `get_last_checkpoint(recovery/)` and the prune both scan: direct children named checkpoint-N
    assert not [p for p in recovery_root(tmp_path).iterdir() if HF_CHECKPOINT_RE.match(p.name)]
    gate.set()
    assert rec.join(JOIN)
    assert complete_steps(tmp_path) == [50]


# ---------------------------------------------------------------- ordering across ranks


def test_complete_only_after_every_rank_is_done(tmp_path):
    gate1 = threading.Event()
    rank0, rank1 = make(tmp_path, rank=0, world=2), make(tmp_path, rank=1, world=2)
    seen: list[set[str]] = []

    def marker(path: Path, obj) -> None:
        tmp = recovery_tmp_dir(tmp_path, 5)
        seen.append({p.name for p in tmp.iterdir()} | {p.name for p in path.parent.iterdir()})
        atomic_write_json(path, obj)

    rank0.write_marker = marker
    rank0.save(5, Writer(rank=0).snapshot)
    rank1.save(5, Writer(rank=1, gate=gate1).snapshot)
    tmp = recovery_tmp_dir(tmp_path, 5)
    for _ in range(100):  # rank 0 has finished its own files and now waits for rank 1
        if done_path(tmp, 0).is_file():
            break
        threading.Event().wait(0.01)
    assert done_path(tmp, 0).is_file() and not done_path(tmp, 1).exists()
    assert rank0.busy() and rank1.busy()
    assert complete_steps(tmp_path) == [] and not recovery_dir(tmp_path, 5).exists()
    gate1.set()
    assert rank1.join(JOIN) and rank0.join(JOIN)
    assert complete_steps(tmp_path) == [5]
    assert {".rank-0.done", ".rank-1.done", "rng_state_1.pth", "optimizer.pt"} <= seen[0]  # all before COMPLETE
    assert sorted(p.name for p in recovery_dir(tmp_path, 5).iterdir()) == [
        COMPLETE,
        "model.safetensors",
        "optimizer.pt",
        "rng_state_0.pth",
        "rng_state_1.pth",
        "trainer_state.json",
    ]
    written = events(tmp_path)[-1]
    assert written["event"] == "recovery_written" and written["bytes"] == 40  # both ranks


def test_rank_waits_for_main_before_it_stops_being_busy(tmp_path):
    gate0 = threading.Event()
    rank0, rank1 = make(tmp_path, rank=0, world=2), make(tmp_path, rank=1, world=2)
    rank0.save(5, Writer(rank=0, gate=gate0).snapshot)
    rank1.save(5, Writer(rank=1).snapshot)
    assert wait_for(done_path(recovery_tmp_dir(tmp_path, 5), 1).is_file)  # its own files are done ...
    assert not rank1.join(0.2)  # ... but main has not settled
    gate0.set()
    assert rank0.join(JOIN) and rank1.join(JOIN)


def test_main_gives_up_after_timeout_without_complete(tmp_path):
    rank0 = make(tmp_path, rank=0, world=2, timeout=0.2)
    rank0.save(5, Writer(rank=0).snapshot)  # rank 1 never writes
    assert rank0.join(JOIN)
    tmp = recovery_tmp_dir(tmp_path, 5)
    assert complete_steps(tmp_path) == [] and abandoned_path(tmp).is_file()
    failed = events(tmp_path)[-1]
    assert failed["event"] == "recovery_failed" and "timed out" in failed["reason"] and "[1]" in failed["reason"]
    assert rank0.failed == [5]


# ---------------------------------------------------------------- skip if busy


def test_skip_while_previous_write_in_flight(tmp_path):
    gate = threading.Event()
    rec = make(tmp_path)
    writer = Writer(gate=gate)
    assert rec.save(3, writer.snapshot) is True
    assert rec.save(6, writer.snapshot) is False  # step 3 still writing: no snapshot at all
    assert len(writer.calls) == 1 and not recovery_tmp_dir(tmp_path, 6).exists()
    gate.set()
    assert rec.join(JOIN)
    assert rec.save(9, writer.snapshot) is True
    assert rec.join(JOIN)
    assert complete_steps(tmp_path) == [3, 9]
    assert rec.skipped == [6]
    assert [(r["event"], r["step"]) for r in events(tmp_path) if r["event"] == "recovery_skipped"] == [
        ("recovery_skipped", 6)
    ]


def test_skip_decision_is_collective(tmp_path):
    """A rank that is idle still skips when another rank reports busy (all_reduce MAX on GPU)."""
    calls = []
    rec = make(tmp_path, any_busy=lambda local: calls.append(local) or True)
    writer = Writer()
    assert rec.save(3, writer.snapshot) is False
    assert calls == [False] and writer.calls == [] and rec.skipped == [3]
    assert not recovery_root(tmp_path).exists()


def test_non_main_rank_skips_silently(tmp_path):
    rec = make(tmp_path, rank=1, world=2, any_busy=lambda local: True)
    assert rec.save(3, Writer(rank=1).snapshot) is False
    assert events(tmp_path) == []  # only the main rank logs


# ---------------------------------------------------------------- crashes


def test_failed_write_leaves_no_complete_and_is_cleaned(tmp_path):
    rec = make(tmp_path)
    rec.save(3, Writer(fail_on="optimizer.pt").snapshot)
    assert rec.join(JOIN)
    tmp = recovery_tmp_dir(tmp_path, 3)
    assert complete_steps(tmp_path) == [] and list_recovery_checkpoints(tmp_path) == []
    assert failed_path(tmp, 0).is_file() and abandoned_path(tmp).is_file()
    assert (tmp / "checkpoint-3" / "optimizer.pt").read_bytes() == b"half"
    assert events(tmp_path)[-1]["event"] == "recovery_failed" and rec.failed == [3]
    # training goes on: the next save works and removes the abandoned staging dir first
    assert rec.save(6, Writer().snapshot) is True
    assert rec.join(JOIN)
    assert complete_steps(tmp_path) == [6] and list_recovery_tmp_dirs(tmp_path) == []


def test_process_crash_mid_write_is_ignored_then_removed(tmp_path):
    # what a killed process leaves behind: staging dir, partial files, no done files, no COMPLETE
    partial = recovery_tmp_dir(tmp_path, 100) / "checkpoint-100"
    partial.mkdir(parents=True)
    (partial / "model.safetensors").write_bytes(b"half")
    good = recovery_dir(tmp_path, 50)
    good.mkdir(parents=True)
    (good / COMPLETE).write_text("{}")
    latest = latest_recovery_checkpoint(tmp_path)
    assert latest is not None and latest.number == 50  # the coordinator resumes from 50
    assert remove_incomplete_recovery(tmp_path) == [recovery_tmp_dir(tmp_path, 100)]
    assert list_recovery_tmp_dirs(tmp_path) == [] and complete_steps(tmp_path) == [50]


def test_snapshot_error_propagates_like_a_failed_sync_save(tmp_path):
    rec = make(tmp_path)

    def broken(tmp: Path) -> list[DeferredWrite]:
        raise RuntimeError("out of host memory")

    with pytest.raises(RuntimeError, match="host memory"):
        rec.save(3, broken)
    assert not rec.busy() and complete_steps(tmp_path) == []


def test_leftover_final_dir_from_an_earlier_attempt_is_replaced(tmp_path):
    old = recovery_dir(tmp_path, 3)
    old.mkdir(parents=True)
    (old / "stale.bin").write_bytes(b"old")
    rec = make(tmp_path)
    rec.save(3, Writer().snapshot)
    assert rec.join(JOIN)
    assert not (old / "stale.bin").exists() and (old / COMPLETE).is_file()


# ---------------------------------------------------------------- pruning and the final join


def test_prune_only_after_the_new_checkpoint_is_complete(tmp_path):
    rec = make(tmp_path, pruning=True, keep=2)
    for step in (3, 6):
        rec.save(step, Writer().snapshot)
        assert rec.join(JOIN)
    gate = threading.Event()
    rec.save(9, Writer(gate=gate).snapshot)
    assert complete_steps(tmp_path) == [3, 6]  # 9 is in flight: nothing pruned yet
    gate.set()
    assert rec.join(JOIN)
    assert complete_steps(tmp_path) == [6, 9]
    assert [p.name for p in rec.pruned] == ["checkpoint-3"]
    assert events(tmp_path)[-1]["pruned"] == ["checkpoint-3"]


def test_pruning_disabled_keeps_everything(tmp_path):
    rec = make(tmp_path, pruning=False)
    for step in (3, 6, 9):
        rec.save(step, Writer().snapshot)
        assert rec.join(JOIN)
    assert complete_steps(tmp_path) == [3, 6, 9] and rec.pruned == []


def test_final_join_waits_for_the_in_flight_write(tmp_path):
    gate = threading.Event()
    rec = make(tmp_path)
    assert rec.join(JOIN)  # nothing started
    rec.save(10, Writer(gate=gate).snapshot)
    assert rec.join(0.05) is False and complete_steps(tmp_path) == []
    gate.set()
    assert rec.join(JOIN) is True
    assert complete_steps(tmp_path) == [10] and not rec.busy()


def test_bad_rank_rejected(tmp_path):
    with pytest.raises(ValueError):
        AsyncRecovery(tmp_path, rank=2, world_size=2)


# ---------------------------------------------------------------- flag and config


def test_on_save_is_a_no_op_with_async_recovery(tmp_path):
    cb = ArtifactCallback(tmp_path, eval_every=5, async_recovery=True)
    cb.on_save(None, type("S", (), {"global_step": 5, "max_steps": 10})(), None)  # no checkpoint dir: no raise
    assert cb.recovery_marked == [] and not recovery_root(tmp_path).exists()


def test_flag_off_keeps_the_synchronous_on_save(tmp_path):
    cb = ArtifactCallback(tmp_path, eval_every=5)
    assert cb.async_recovery is False
    recovery_dir(tmp_path, 5).mkdir(parents=True)
    (recovery_dir(tmp_path, 5) / "optimizer.pt").write_bytes(b"x")
    cb.on_save(None, type("S", (), {"global_step": 5, "max_steps": 10})(), None)
    assert cb.recovery_marked == [5] and (recovery_dir(tmp_path, 5) / COMPLETE).is_file()


def test_config_enables_async_recovery_and_it_is_not_identity():
    cfg = load_config(DEFAULT_CONFIG, "4b")
    assert cfg["checkpoints"]["async_recovery"] is True
    args = parse_args(["--run-id", "x", "--size", "4b", "--reward", "robust", "--seed", "1", "--output-dir", "r"])
    settings = run_settings(cfg, args)
    assert settings["checkpoints"]["async_recovery"] is True  # resolved into train_config.json
    sync = {**settings, "checkpoints": {**settings["checkpoints"], "async_recovery": False}}
    legacy = {**settings, "checkpoints": {k: v for k, v in settings["checkpoints"].items() if k != "async_recovery"}}
    assert config_identity(settings) == config_identity(sync) == config_identity(legacy)
    changed = {**settings, "checkpoints": {**settings["checkpoints"], "recovery_every": 25}}
    assert config_identity(changed) != config_identity(settings)
