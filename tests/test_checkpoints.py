import json
from pathlib import Path

import pytest

from train.checkpoints import (
    check_final_integrity,
    eval_step_dir,
    final_eval_checkpoint,
    latest_recovery_checkpoint,
    list_eval_checkpoints,
    prune_recovery_checkpoints,
    recovery_dir,
    step_name,
)


def write_eval_step(rdir: Path, step: int, *, final: bool = False, files=None, weights=True):
    d = eval_step_dir(rdir, step)
    d.mkdir(parents=True)
    if weights:
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\0" * 8)
    marker = {"step": step, "final": final}
    if files is not None:
        marker["files"] = files
    (d / "COMPLETE").write_text(json.dumps(marker))
    return d


def write_recovery(rdir: Path, n: int, complete: bool = True) -> Path:
    d = recovery_dir(rdir, n)
    d.mkdir(parents=True)
    (d / "optimizer.pt").write_bytes(b"x")
    if complete:
        (d / "COMPLETE").write_text("")
    return d


@pytest.fixture
def rdir(tmp_path):
    d = tmp_path / "runs" / "2b-rob-s1"
    d.mkdir(parents=True)
    return d


def snapshot(root: Path) -> set[Path]:
    return set(root.rglob("*"))


def test_step_name_is_zero_padded():
    assert step_name(50) == "step-00050"
    assert step_name(123456) == "step-123456"


def test_eval_listing_ignores_incomplete_and_finds_final(rdir):
    write_eval_step(rdir, 50)
    write_eval_step(rdir, 100)
    write_eval_step(rdir, 130, final=True)
    eval_step_dir(rdir, 150).mkdir()  # no COMPLETE: in progress
    assert [c.step for c in list_eval_checkpoints(rdir)] == [50, 100, 130]
    final = final_eval_checkpoint(rdir)
    assert final is not None
    assert final.step == 130


def test_empty_complete_marker_reads_as_non_final(rdir):
    d = write_eval_step(rdir, 50)
    (d / "COMPLETE").write_text("")
    assert list_eval_checkpoints(rdir)[0].final is False
    assert final_eval_checkpoint(rdir) is None


def test_latest_recovery_skips_incomplete(rdir):
    write_recovery(rdir, 10)
    write_recovery(rdir, 20)
    write_recovery(rdir, 30, complete=False)
    latest = latest_recovery_checkpoint(rdir)
    assert latest is not None
    assert latest.number == 20


def test_prune_keeps_last_two_complete(rdir):
    for n in (10, 20, 30, 40):
        write_recovery(rdir, n)
    partial = write_recovery(rdir, 50, complete=False)
    write_eval_step(rdir, 50)
    write_eval_step(rdir, 60, final=True, files=["config.json", "model.safetensors"])
    eval_before = snapshot(rdir / "eval")

    res = prune_recovery_checkpoints(rdir, keep=2, enabled=True)

    assert res.refused is None
    assert sorted(p.name for p in res.pruned) == ["checkpoint-10", "checkpoint-20"]
    assert sorted(p.name for p in res.kept) == ["checkpoint-30", "checkpoint-40"]
    assert not recovery_dir(rdir, 10).exists() and not recovery_dir(rdir, 20).exists()
    assert recovery_dir(rdir, 30).exists() and recovery_dir(rdir, 40).exists()
    assert partial.exists()  # incomplete dirs are never touched
    assert snapshot(rdir / "eval") == eval_before  # eval/ is never touched


def test_prune_is_off_by_default(rdir):
    for n in (10, 20, 30):
        write_recovery(rdir, n)
    write_eval_step(rdir, 60, final=True)
    res = prune_recovery_checkpoints(rdir)
    assert res.refused and "disabled" in res.refused
    assert all(recovery_dir(rdir, n).exists() for n in (10, 20, 30))


@pytest.mark.parametrize(
    "setup, reason",
    [
        ("no_final", "no complete final eval step"),
        ("final_incomplete", "no complete final eval step"),
        ("missing_weights", "*.safetensors"),
        ("missing_listed", "tokenizer.json"),
    ],
)
def test_prune_refuses_when_integrity_fails(rdir, setup, reason):
    for n in (10, 20, 30):
        write_recovery(rdir, n)
    write_eval_step(rdir, 50)
    if setup == "final_incomplete":
        d = write_eval_step(rdir, 60, final=True)
        (d / "COMPLETE").unlink()
    elif setup == "missing_weights":
        d = write_eval_step(rdir, 60, final=True)
        (d / "model.safetensors").unlink()
    elif setup == "missing_listed":
        write_eval_step(rdir, 60, final=True, files=["config.json", "tokenizer.json"])

    res = prune_recovery_checkpoints(rdir, keep=2, enabled=True)

    assert res.refused and "integrity check failed" in res.refused
    assert reason in res.refused
    assert res.pruned == []
    assert all(recovery_dir(rdir, n).exists() for n in (10, 20, 30))


def test_integrity_rejects_listed_file_outside_step_dir(rdir):
    write_eval_step(rdir, 60, final=True, files=["../../status.json"])
    (rdir / "status.json").write_text("{}")
    result = check_final_integrity(rdir)
    assert not result.ok
    assert any("escapes" in r for r in result.reasons)


def test_keep_must_be_positive(rdir):
    with pytest.raises(ValueError):
        prune_recovery_checkpoints(rdir, keep=0, enabled=True)
