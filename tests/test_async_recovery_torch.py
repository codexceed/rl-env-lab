"""train/async_recovery.py with real torch and transformers (skipped where the GPU stack is absent, as in CI).

Covers what the torch-free tests cannot: the snapshot is a true copy (later updates do not leak into
the files), storage sharing survives, and a real `transformers.Trainer` writes the same checkpoint
asynchronously as synchronously and resumes from it to the same weights.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from train.async_recovery import (  # noqa: E402 - after the importorskip guards
    AsyncRecovery,
    DeferredWrites,
    PinnedPool,
    Snapshotter,
)
from train.checkpoints import COMPLETE, list_recovery_checkpoints, recovery_dir  # noqa: E402


def _step(model, opt) -> None:
    loss = model(torch.randn(4, 3)).pow(2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()


def test_deferred_files_hold_the_state_at_snapshot_time(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    _step(model, opt)
    expected_w = model.weight.detach().clone()
    expected_m = opt.state[model.weight]["exp_avg"].clone()
    original_save = torch.save
    with DeferredWrites(PinnedPool()) as deferred:
        torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
        safetensors_torch.save_file({"weight": model.weight}, str(tmp_path / "model.safetensors"), metadata={"a": "b"})
    assert torch.save is original_save  # restored on exit
    assert [w.path.name for w in deferred.items] == ["optimizer.pt", "model.safetensors"]
    assert not (tmp_path / "optimizer.pt").exists()  # nothing written yet
    _step(model, opt)  # training moves on before the background write
    for item in deferred.items:
        item.write()
    saved = torch.load(tmp_path / "optimizer.pt", weights_only=True)
    assert torch.equal(saved["state"][0]["exp_avg"], expected_m)
    assert not torch.equal(saved["state"][0]["exp_avg"], opt.state[model.weight]["exp_avg"])
    assert torch.equal(safetensors_torch.load_file(tmp_path / "model.safetensors")["weight"], expected_w)
    assert deferred.items[0].nbytes > 0


def test_snapshot_keeps_storage_sharing_and_container_types():
    base = torch.arange(12.0)
    obj = {"a": base[:6].view(2, 3), "b": base[6:], "t": (1, "x", base), "p": torch.nn.Parameter(torch.ones(2))}
    snap = Snapshotter(PinnedPool())
    copied = snap.copy(obj)
    base.zero_()
    assert torch.equal(copied["a"], torch.arange(6.0).view(2, 3)) and torch.equal(copied["b"], torch.arange(6.0, 12.0))
    assert copied["a"].untyped_storage().data_ptr() == copied["t"][2].untyped_storage().data_ptr()
    assert isinstance(copied["p"], torch.nn.Parameter) and copied["t"][:2] == (1, "x")
    assert snap.copied_bytes == 12 * 4 + 2 * 4  # each storage once


def test_freed_and_reallocated_tensors_are_not_confused(tmp_path):
    """A tensor freed after one save may come back at the same address in the next (e.g. RNG state)."""
    with DeferredWrites(PinnedPool()) as deferred:
        for i in range(5):
            torch.save({"rng": torch.full((1264,), float(i))}, tmp_path / f"f{i}.pt")
    for item in deferred.items:
        item.write()
    for i in range(5):
        assert torch.equal(torch.load(tmp_path / f"f{i}.pt", weights_only=True)["rng"], torch.full((1264,), float(i)))


def test_other_threads_and_file_objects_are_not_deferred(tmp_path):
    import io

    out = {}
    with DeferredWrites(PinnedPool()) as deferred:
        t = threading.Thread(target=lambda: torch.save({"x": 1}, tmp_path / "other.pt"))
        t.start()
        t.join()
        buf = io.BytesIO()
        torch.save({"y": 2}, buf)
        out["buf"] = buf.getvalue()
    assert (tmp_path / "other.pt").exists() and out["buf"] and deferred.items == []


def test_pool_reuses_buffers():
    pool = PinnedPool()
    a = pool.take(64)
    pool.reset()
    assert pool.take(64) is a and pool.allocated_bytes == 64


# ---------------------------------------------------------------- a real transformers.Trainer

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")


def _trainer(cls, out: Path, max_steps: int = 6):
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments  # noqa: F401

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    g = torch.Generator().manual_seed(1)
    data = [{"input_ids": ids, "labels": ids} for ids in torch.randint(0, 32, (24, 8), generator=g)]
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        learning_rate=1e-2,
        save_strategy="steps",
        save_steps=3,
        save_total_limit=None,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
        seed=0,
        dataloader_num_workers=0,
    )
    return cls(model=model, args=args, train_dataset=data)


def _load(path: Path):
    if path.suffix == ".safetensors":
        return safetensors_torch.load_file(path)
    if path.suffix in (".pt", ".pth", ".bin"):
        return torch.load(path, weights_only=False)
    if path.suffix == ".json":
        state = json.loads(path.read_text())
        for row in state.get("log_history", []) if isinstance(state, dict) else []:
            row.pop("train_runtime", None)
        return state
    return path.read_bytes()


def _same(a, b) -> bool:
    import numpy as np

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return bool(np.array_equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if type(a).__name__ == "TrainingArguments":  # the two runs differ only in their output dirs

        def norm(args) -> dict:
            return {k: v for k, v in args.to_dict().items() if k not in ("output_dir", "logging_dir")}

        return norm(a) == norm(b)
    return a == b


def test_trainer_async_checkpoint_equals_sync_and_resumes(tmp_path):
    from transformers import Trainer

    from train.async_recovery import AsyncRecoveryTrainerMixin

    sync_run, async_run = tmp_path / "sync", tmp_path / "async"
    sync = _trainer(Trainer, sync_run / "recovery")
    sync.train()

    cls = type("AsyncTrainer", (AsyncRecoveryTrainerMixin, Trainer), {})
    trainer = _trainer(cls, async_run / "recovery")
    trainer.async_recovery = AsyncRecovery(async_run, marker_extra={"run_id": "t"}, log=lambda _m: None)
    trainer.train()
    assert trainer.async_recovery.join(60)
    rec = trainer.async_recovery
    assert 3 in rec.completed  # the first save never finds a write in flight; 6 may have been skipped
    assert [c.number for c in list_recovery_checkpoints(async_run) if c.complete] == rec.completed
    for step in rec.completed:
        a, s = recovery_dir(async_run, step), sync_run / "recovery" / f"checkpoint-{step}"
        names = sorted(p.name for p in a.iterdir() if p.name != COMPLETE)
        assert names == sorted(p.name for p in s.iterdir())
        assert {"model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"} <= set(
            names
        )
        for name in names:
            assert _same(_load(a / name), _load(s / name)), name

    # resuming from the async checkpoint-3 reaches exactly the uninterrupted weights
    resumed = _trainer(Trainer, tmp_path / "resumed" / "recovery")
    resumed.train(resume_from_checkpoint=str(recovery_dir(async_run, 3)))
    for (name, p), (_, q) in zip(resumed.model.named_parameters(), sync.model.named_parameters(), strict=True):
        assert torch.equal(p, q), name
