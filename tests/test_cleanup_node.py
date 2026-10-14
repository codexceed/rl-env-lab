"""scripts/cleanup_node.py on a tmp tree: what is kept, what is deleted, and every refusal."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from scripts import cleanup_node as cn

from train.checkpoints import COMPLETE, atomic_write_json, eval_step_dir, recovery_dir

RUN_RE = re.compile(cn.DEFAULT_RUN_PATTERN)


def eval_step(run: Path, step: int, complete: bool = True, final: bool = False, **marker) -> Path:
    d = eval_step_dir(run, step)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"w" * 1000)
    if complete:
        atomic_write_json(d / COMPLETE, {"step": step, "final": final, **marker})
    return d


def recovery(run: Path, number: int) -> Path:
    d = recovery_dir(run, number)
    d.mkdir(parents=True)
    (d / "optimizer.pt").write_bytes(b"o" * 5000)
    (d / COMPLETE).write_text("{}")
    return d


def scores(evals: Path, run_id: str, step: int, true_success: float, n: int = 20) -> None:
    out = evals / run_id / f"step-{step:05d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    k = round(true_success * n)
    out.write_text("".join(json.dumps({"robust_pass": i < k, "leaky_pass": True}) + "\n" for i in range(n)))
    traces = out.with_name(f"step-{step:05d}.traces.jsonl")  # ignored: never a score
    traces.write_text(json.dumps({"robust_pass": True}) + "\n")


@pytest.fixture
def tree(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "evals"
    # peak (0.60 at 50) beats final (0.40 at 150) by 0.20: kept
    a = runs / "2b-rob-s1"
    for step in (50, 100):
        eval_step(a, step)
    eval_step(a, 150, final=True)
    eval_step(a, 200, complete=False)  # partial write after the final: deleted
    recovery(a, 100)
    recovery(a, 150)
    for step, ts in ((50, 0.60), (100, 0.50), (150, 0.40)):
        scores(evals, "2b-rob-s1", step, ts)
    # peak (0.45 at 50) beats final (0.40) by only 0.05: not kept
    b = runs / "v3-4b-leak-s2"
    eval_step(b, 50)
    eval_step(b, 100, final=True)
    recovery(b, 100)
    scores(evals, "v3-4b-leak-s2", 50, 0.45)
    scores(evals, "v3-4b-leak-s2", 100, 0.40)
    # no final step: fails integrity, nothing touched
    c = runs / "0.8b-leak-s3"
    eval_step(c, 50)
    eval_step(c, 100)
    recovery(c, 100)
    # not a matrix id: never touched
    d = runs / "dry-2b-rob"
    eval_step(d, 5)
    eval_step(d, 10, final=True)
    recovery(d, 10)
    (runs / "models" / "Qwen3.5-2B-text").mkdir(parents=True)
    (runs / "models" / "Qwen3.5-2B-text" / "model.safetensors").write_bytes(b"m" * 100)
    (runs / "2b-rob-s1" / "logs").mkdir()
    (runs / "2b-rob-s1" / "logs" / "attempt-1.log").write_text("log\n")
    return runs, evals


def no_busy() -> list[str]:
    return []


def all_files(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


def test_final_kept_peak_kept_above_margin_and_rest_deleted(tree):
    runs, evals = tree
    plan = cn.build_plan(runs, evals, run_re=RUN_RE)
    by = {r.run_id: r for r in plan.runs}
    assert set(by) == {"2b-rob-s1", "v3-4b-leak-s2", "0.8b-leak-s3"}  # dry-2b-rob and models never selected

    a = by["2b-rob-s1"]
    assert (a.final_step, a.peak_step, a.peak_kept) == (150, 50, True)
    assert a.final_score == pytest.approx(0.40) and a.peak_score == pytest.approx(0.60)
    assert a.keep == [eval_step_dir(runs / "2b-rob-s1", 50), eval_step_dir(runs / "2b-rob-s1", 150)]
    assert a.delete == [
        eval_step_dir(runs / "2b-rob-s1", 100),
        eval_step_dir(runs / "2b-rob-s1", 200),
        runs / "2b-rob-s1" / "recovery",
    ]

    b = by["v3-4b-leak-s2"]
    assert (b.final_step, b.peak_step, b.peak_kept) == (100, 50, False)
    assert b.delete == [eval_step_dir(runs / "v3-4b-leak-s2", 50), runs / "v3-4b-leak-s2" / "recovery"]


def test_peak_margin_is_a_flag_and_exact_margin_counts(tree):
    runs, evals = tree
    by = {r.run_id: r for r in cn.build_plan(runs, evals, run_re=RUN_RE, peak_margin=0.05).runs}
    assert by["v3-4b-leak-s2"].peak_kept  # 0.45 - 0.40 = 0.05, despite float rounding
    by = {r.run_id: r for r in cn.build_plan(runs, evals, run_re=RUN_RE, peak_margin=0.30).runs}
    assert not by["2b-rob-s1"].peak_kept


def test_integrity_failure_keeps_recovery_and_every_checkpoint(tree):
    runs, evals = tree
    plan = cn.build_plan(runs, evals, run_re=RUN_RE)
    c = next(r for r in plan.runs if r.run_id == "0.8b-leak-s3")
    assert not c.integrity_ok and c.delete == []
    assert [r.run_id for r in plan.refusals] == ["0.8b-leak-s3"]
    assert not any("0.8b-leak-s3" in str(p) for p in plan.paths)
    assert "integrity refusals" in cn.format_plan(plan)


def test_recovery_deleted_only_with_integrity(tree):
    runs, _ = tree
    paths = [runs / "2b-rob-s1" / "recovery", runs / "0.8b-leak-s3" / "recovery"]
    res = cn.apply_plan(paths, runs, RUN_RE, busy_check=no_busy)
    assert [p for p, _ in res.deleted] == [runs / "2b-rob-s1" / "recovery"]
    assert [p for p, _ in res.refused] == [runs / "0.8b-leak-s3" / "recovery"]
    assert "check_final_integrity" in res.refused[0][1]
    assert (runs / "0.8b-leak-s3" / "recovery" / "checkpoint-100").is_dir()


def test_dry_run_deletes_nothing_and_writes_the_plan(tree, capsys):
    runs, evals = tree
    before = all_files(runs.parent)
    plan_out = runs.parent / "plan.txt"
    rc = cn.main(
        ["--runs-root", str(runs), "--evals-root", str(evals), "--plan-out", str(plan_out)], busy_check=no_busy
    )
    assert rc == 0
    assert all_files(runs.parent) - {plan_out} == before
    lines = plan_out.read_text().splitlines()
    assert str(runs / "2b-rob-s1" / "recovery") in lines and len(lines) == 5
    out = capsys.readouterr().out
    assert "DRY RUN" in out and cn.APPLY_COMMAND in out


def test_apply_end_to_end_keeps_only_policy_paths(tree, capsys):
    runs, evals = tree
    evals_before = all_files(evals)
    args = ["--runs-root", str(runs), "--evals-root", str(evals), "--plan-out", str(runs.parent / "p.txt"), "--apply"]
    assert cn.main(args, busy_check=no_busy) == 0
    a = runs / "2b-rob-s1"
    assert sorted(p.name for p in (a / "eval").iterdir()) == ["step-00050", "step-00150"]
    assert not (a / "recovery").exists() and (a / "logs" / "attempt-1.log").is_file()
    assert sorted(p.name for p in (runs / "v3-4b-leak-s2" / "eval").iterdir()) == ["step-00100"]
    assert (runs / "0.8b-leak-s3" / "recovery").is_dir()  # integrity refusal
    assert len(list((runs / "dry-2b-rob" / "eval").iterdir())) == 2 and (runs / "dry-2b-rob" / "recovery").is_dir()
    assert (runs / "models").is_dir()  # extras are opt-in
    assert all_files(evals) == evals_before  # eval outputs never touched
    out = capsys.readouterr().out
    assert "deleted 5 paths" in out and "remaining:" in out


def test_busy_processes_refuse_apply_unless_forced(tree, capsys):
    runs, evals = tree
    args = ["--runs-root", str(runs), "--evals-root", str(evals), "--plan-out", str(runs.parent / "p.txt"), "--apply"]

    def busy() -> list[str]:
        return ["train/grpo.py: pids 123"]

    assert cn.main(args, busy_check=busy) == 1
    assert (runs / "2b-rob-s1" / "recovery").is_dir()
    assert "REFUSED" in capsys.readouterr().out
    assert cn.main([*args, "--force-busy"], busy_check=busy) == 0
    assert not (runs / "2b-rob-s1" / "recovery").exists()


def test_paths_outside_the_patterns_are_refused(tree, tmp_path):
    runs, evals = tree
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (runs / "2b-rob-s1" / "eval" / "step-00100-link").symlink_to(eval_step_dir(runs / "2b-rob-s1", 100))
    cases = {
        outside: "outside the runs root",
        evals / "2b-rob-s1": "outside the runs root",
        runs / "2b-rob-s1": "matches neither",
        runs / "2b-rob-s1" / "logs": "matches neither",
        runs / "2b-rob-s1" / "eval": "matches neither",
        runs / "2b-rob-s1" / "eval" / "step-00100-link": "symlink",
        eval_step_dir(runs / "2b-rob-s1", 150): "final eval checkpoint",
        runs / "dry-2b-rob" / "recovery": "not under a matching run id",
        runs / "models": "not under a matching run id",
        runs / "2b-rob-s1" / "eval" / ".." / ".." / "dry-2b-rob" / "recovery": "not under a matching run id",
    }
    res = cn.apply_plan(list(cases), runs, RUN_RE, busy_check=no_busy)
    assert res.deleted == []
    for path, reason in res.refused:
        assert cases[path] in reason, (path, reason)
    assert len(res.refused) == len(cases)
    assert outside.is_dir() and (runs / "models").is_dir() and eval_step_dir(runs / "2b-rob-s1", 100).is_dir()


def test_extras_only_when_enabled(tree, capsys):
    runs, evals = tree
    hf = runs.parent / "hf" / "hub"
    hf.mkdir(parents=True)
    args = ["--runs-root", str(runs), "--evals-root", str(evals), "--plan-out", str(runs.parent / "p.txt")]
    cn.main([*args, "--include-models", "--include-hf-cache", "--hf-cache", str(hf)], busy_check=no_busy)
    lines = (runs.parent / "p.txt").read_text().splitlines()
    assert lines[-2:] == [str(runs / "models"), str(hf)]
    capsys.readouterr()
    assert cn.main([*args, "--include-models", "--apply"], busy_check=no_busy) == 0
    assert not (runs / "models").exists() and hf.is_dir()  # hf cache not enabled on this invocation


def test_final_without_score_keeps_the_peak(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "evals"
    run = runs / "4b-rob-s1"
    eval_step(run, 50)
    eval_step(run, 100, final=True)
    scores(evals, "4b-rob-s1", 50, 0.3)
    (plan,) = cn.build_plan(runs, evals, run_re=RUN_RE).runs
    assert plan.peak_kept and plan.delete == [] and "no held-out score" in plan.notes[0]
