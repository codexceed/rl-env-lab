"""train/callbacks.py under a fake Trainer loop: COMPLETE-last, metrics log, gated pruning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from train.callbacks import (
    TRAIN_METRICS,
    ArtifactCallback,
    TrainMetricsWriter,
    list_files,
    metrics_row,
    remove_incomplete_recovery,
    rolling_prune,
)
from train.checkpoints import (
    COMPLETE,
    atomic_write_json,
    check_final_integrity,
    eval_step_dir,
    final_eval_checkpoint,
    list_eval_checkpoints,
    list_recovery_checkpoints,
    read_marker,
    recovery_dir,
)


@dataclass
class State:
    global_step: int = 0
    max_steps: int = 0


def trl_logs(step: int) -> dict:
    """A TRL 1.15 GRPOTrainer log dict for one step (robust run: weight 1 on robust)."""
    return {
        "loss": 0.01,
        "grad_norm": 0.5,
        "learning_rate": 3e-6,
        "reward": 0.25 + step / 100,
        "reward_std": 0.4,
        "rewards/robust_reward/mean": 0.25 + step / 100,
        "rewards/leaky_reward/mean": 0.5 + step / 100,
        "frac_reward_zero_std": 0.375,
        "entropy": 0.9,
        "completions/mean_length": 61.0,
        "completions/clipped_ratio": 0.0,
        "step_time": 12.0,
        "num_tokens": 1234.0,  # unknown keys are dropped
    }


class FakeTrainer:
    """Mimics the order of events in `transformers.Trainer` (5.17) for one rank.

    Per optimizer step: on_step_begin -> on_step_end -> on_log -> (if save step) _save_checkpoint
    writes recovery/checkpoint-N, then on_save.
    """

    def __init__(self, run_dir: Path, callback: ArtifactCallback, *, max_steps: int, save_steps: int):
        self.run_dir = run_dir
        self.cb = callback
        self.state = State(0, max_steps)
        self.save_steps = save_steps
        self.events: list[str] = []

    def save_eval_weights(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"weights")
        (path / "tokenizer.json").write_text("{}")
        self.events.append(f"eval-weights:{path.name}")

    def _save_checkpoint(self) -> None:
        d = recovery_dir(self.run_dir, self.state.global_step)
        d.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            (d / name).write_bytes(b"x")
        self.events.append(f"recovery:{d.name}")

    def train(self, start: int = 0) -> None:
        self.state.global_step = start
        while self.state.global_step < self.state.max_steps:
            self.cb.on_step_begin(None, self.state, None)
            self.state.global_step += 1
            self.cb.on_step_end(None, self.state, None)
            self.cb.on_log(None, self.state, None, logs=trl_logs(self.state.global_step))
            final = self.state.global_step >= self.state.max_steps
            if self.state.global_step % self.save_steps == 0 or final:
                self._save_checkpoint()
                self.cb.on_save(None, self.state, None)
        self.cb.on_log(None, self.state, None, logs={"train_runtime": 1.0})  # end-of-training summary


def make(run_dir: Path, *, max_steps=7, every=3, pruning=False, keep=2, is_main=True, clock=None):
    events: list[str] = []

    def marker(path: Path, obj) -> None:
        # COMPLETE must come after every other file of its dir
        siblings = [p.name for p in path.parent.iterdir() if p.name != COMPLETE and not p.name.startswith(".")]
        assert siblings, f"COMPLETE written into an empty dir {path.parent}"
        if path.parent.parent.name == "eval":
            assert {"config.json", "model.safetensors"} <= set(siblings)
        events.append(f"complete:{path.parent.name}")
        atomic_write_json(path, obj)

    ticks = iter(range(10_000))
    cb = ArtifactCallback(
        run_dir,
        eval_every=every,
        keep_last=keep,
        pruning_enabled=pruning,
        is_main=is_main,
        marker_extra={"run_id": "2b-rob-s1"},
        write_marker=marker,
        clock=clock or (lambda: float(next(ticks)) * 2.0),
    )
    trainer = FakeTrainer(run_dir, cb, max_steps=max_steps, save_steps=every)
    cb.save_eval_weights = trainer.save_eval_weights
    trainer.events = events
    return trainer, cb, events


# ---------------------------------------------------------------- COMPLETE last


def test_eval_and_recovery_artifacts_with_complete_last(tmp_path):
    trainer, _, events = make(tmp_path, max_steps=7, every=3)
    trainer.train()
    assert [c.step for c in list_eval_checkpoints(tmp_path)] == [3, 6, 7]
    assert [c.final for c in list_eval_checkpoints(tmp_path)] == [False, False, True]
    assert [c.number for c in list_recovery_checkpoints(tmp_path) if c.complete] == [3, 6, 7]
    # every COMPLETE right after its own dir's files
    for name in ("step-00003", "step-00006", "step-00007"):
        assert events.index(f"complete:{name}") == events.index(f"eval-weights:{name}") + 1
    for name in ("checkpoint-3", "checkpoint-6", "checkpoint-7"):
        assert events.index(f"complete:{name}") == events.index(f"recovery:{name}") + 1
    marker = read_marker(eval_step_dir(tmp_path, 7) / COMPLETE)
    assert marker["step"] == 7 and marker["final"] is True and marker["run_id"] == "2b-rob-s1"
    assert marker["files"] == ["config.json", "model.safetensors", "tokenizer.json"]
    assert check_final_integrity(tmp_path).ok


def test_final_step_on_a_multiple_is_marked_final_once(tmp_path):
    trainer, _, _ = make(tmp_path, max_steps=6, every=3)
    trainer.train()
    assert [(c.step, c.final) for c in list_eval_checkpoints(tmp_path)] == [(3, False), (6, True)]


def test_crash_during_eval_save_leaves_no_complete_and_retry_replaces_partial(tmp_path):
    trainer, cb, _ = make(tmp_path, max_steps=7, every=3)

    def crash(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"half")
        raise RuntimeError("node died")

    real = cb.save_eval_weights
    cb.save_eval_weights = lambda p: crash(p) if p.name == "step-00006" else real(p)  # type: ignore[misc]
    with pytest.raises(RuntimeError):
        trainer.train()
    assert [c.step for c in list_eval_checkpoints(tmp_path)] == [3]  # the partial step-6 is invisible
    assert eval_step_dir(tmp_path, 6).exists() and not (eval_step_dir(tmp_path, 6) / COMPLETE).exists()

    # resume from recovery checkpoint-3 (the coordinator's relaunch)
    trainer2, _, _ = make(tmp_path, max_steps=7, every=3)
    trainer2.train(start=3)
    assert [(c.step, c.final) for c in list_eval_checkpoints(tmp_path)] == [(3, False), (6, False), (7, True)]
    assert (eval_step_dir(tmp_path, 6) / "config.json").exists()


def test_complete_eval_step_is_never_rewritten(tmp_path):
    trainer, cb, _ = make(tmp_path, max_steps=3, every=3)
    trainer.train()
    before = (eval_step_dir(tmp_path, 3) / COMPLETE).read_text()
    calls = []
    cb.save_eval_weights = calls.append  # type: ignore[assignment]
    cb.write_eval_step(3, final=True)
    assert calls == [] and (eval_step_dir(tmp_path, 3) / COMPLETE).read_text() == before


def test_non_main_rank_saves_weights_but_writes_no_markers(tmp_path):
    trainer, cb, _ = make(tmp_path, max_steps=3, every=3, is_main=False)
    saved = []
    cb.save_eval_weights = saved.append  # type: ignore[assignment]
    trainer.train()
    assert saved == [eval_step_dir(tmp_path, 3)]  # ZeRO-3 gathers need every rank
    assert not list(tmp_path.rglob(COMPLETE))
    assert not (tmp_path / TRAIN_METRICS).exists()


def test_on_save_without_trainer_checkpoint_raises(tmp_path):
    _, cb, _ = make(tmp_path)
    with pytest.raises(RuntimeError, match="no recovery checkpoint"):
        cb.on_save(None, State(5, 10), None)


def test_eval_weights_must_write_files(tmp_path):
    _, cb, _ = make(tmp_path)
    cb.save_eval_weights = lambda p: p.mkdir(parents=True)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="no files"):
        cb.write_eval_step(3, final=False)


# ---------------------------------------------------------------- train_metrics.jsonl


def test_train_metrics_one_row_per_step(tmp_path):
    trainer, cb, _ = make(tmp_path, max_steps=4, every=2)
    cb.extra_metrics = lambda: {"grader_errors": 0}
    trainer.train()
    rows = [json.loads(line) for line in (tmp_path / TRAIN_METRICS).read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    r = rows[0]
    assert r["reward"] == pytest.approx(0.26)
    assert r["robust_reward_mean"] == pytest.approx(0.26) and r["leaky_reward_mean"] == pytest.approx(0.51)
    assert r["dead_group_share"] == 0.375 and r["entropy"] == 0.9 and r["completion_length"] == 61.0
    assert r["step_time_s"] == pytest.approx(2.0) and r["grader_errors"] == 0
    assert "num_tokens" not in r


def test_train_metrics_append_only_across_resume(tmp_path):
    trainer, _, _ = make(tmp_path, max_steps=4, every=2)
    trainer.train()
    trainer2, _, _ = make(tmp_path, max_steps=4, every=2)
    trainer2.train(start=2)  # re-logs steps 3 and 4
    steps = [json.loads(line)["step"] for line in (tmp_path / TRAIN_METRICS).read_text().splitlines()]
    assert steps == [1, 2, 3, 4, 3, 4]


def test_metrics_row_mapping_and_nan():
    row = metrics_row(5, {"reward": float("nan"), "entropy": 1.5, "frac_reward_zero_std": 1.0}, {"x": 1})
    assert row == {"step": 5, "reward": None, "entropy": 1.5, "dead_group_share": 1.0, "x": 1}
    assert not any(isinstance(v, float) and math.isnan(v) for v in row.values())


def test_writer_requires_step_and_reward(tmp_path):
    writer = TrainMetricsWriter(tmp_path / "m.jsonl")
    with pytest.raises(ValueError):
        writer.append({"step": 1})
    writer.append({"step": 1, "reward": 0.5})
    writer.append({"step": 2, "reward": 0.75})
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 2


# ---------------------------------------------------------------- pruning


def _ckpt(run: Path, n: int, complete: bool = True) -> Path:
    d = recovery_dir(run, n)
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    if complete:
        (d / COMPLETE).write_text(json.dumps({"step": n}))
    return d


def test_pruning_disabled_by_default_keeps_everything(tmp_path):
    trainer, cb, _ = make(tmp_path, max_steps=12, every=3)
    trainer.train()
    assert [c.number for c in list_recovery_checkpoints(tmp_path)] == [3, 6, 9, 12]
    assert cb.pruned == []


def test_pruning_enabled_keeps_last_two_complete(tmp_path):
    trainer, cb, _ = make(tmp_path, max_steps=12, every=3, pruning=True)
    trainer.train()
    assert [c.number for c in list_recovery_checkpoints(tmp_path)] == [9, 12]
    assert [p.name for p in cb.pruned] == ["checkpoint-3", "checkpoint-6"]
    assert [c.step for c in list_eval_checkpoints(tmp_path)] == [3, 6, 9, 12]  # eval/ untouched
    assert final_eval_checkpoint(tmp_path) is not None


def test_rolling_prune_waits_for_complete_newer(tmp_path):
    _ckpt(tmp_path, 50)
    _ckpt(tmp_path, 100)
    _ckpt(tmp_path, 150, complete=False)  # being written: does not count, is not touched
    assert rolling_prune(tmp_path, keep=2, enabled=True) == []
    (recovery_dir(tmp_path, 150) / COMPLETE).write_text("{}")
    assert [p.name for p in rolling_prune(tmp_path, keep=2, enabled=True)] == ["checkpoint-50"]
    assert [c.number for c in list_recovery_checkpoints(tmp_path)] == [100, 150]
    with pytest.raises(ValueError):
        rolling_prune(tmp_path, keep=0, enabled=True)


def test_rolling_prune_gated_by_flag(tmp_path):
    for n in (1, 2, 3, 4):
        _ckpt(tmp_path, n)
    assert rolling_prune(tmp_path, keep=2, enabled=False) == []
    assert len(list_recovery_checkpoints(tmp_path)) == 4


def test_remove_incomplete_recovery(tmp_path):
    _ckpt(tmp_path, 50)
    partial = _ckpt(tmp_path, 100, complete=False)
    keep = _ckpt(tmp_path, 150, complete=False)
    assert remove_incomplete_recovery(tmp_path, keep=keep) == [partial]
    assert [c.number for c in list_recovery_checkpoints(tmp_path)] == [50, 150]


def test_list_files_skips_hidden(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / ".tmp").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x")
    assert list_files(tmp_path) == ["a.json", "sub/b.bin"]
