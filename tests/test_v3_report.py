"""eval/v3_report.py on synthetic records: near vs far, final vs latest, base reference, pending, bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

from envlab.tasks.v3 import FAR_TEMPLATES, NEAR_TEMPLATES, TRAIN_TEMPLATES
from eval import v3_report
from train.checkpoints import COMPLETE, atomic_write_json, eval_step_dir

PER_TEMPLATE = 4  # instances per held-out template
SAMPLES = 2


def records(run_id: str, step: int, near: float, far: float, *, exploit: float = 0.0, templates=None) -> list[dict]:
    """Held-out records: robust pass rate ``near`` / ``far`` per kind; ``exploit`` share passes leaky only."""
    parts = run_id.split("-")
    if parts[1] == "base":
        size, reward, seed = parts[2], "base", None
    else:
        size, reward, seed = parts[1], {"rob": "robust", "leak": "leaky"}[parts[2]], int(parts[3][1:])
    rows = []
    for t in templates or (*NEAR_TEMPLATES, *FAR_TEMPLATES):
        rate = near if t in NEAR_TEMPLATES or t in TRAIN_TEMPLATES else far
        for i in range(PER_TEMPLATE):
            for k in range(SAMPLES):
                x = (i * SAMPLES + k) / (PER_TEMPLATE * SAMPLES)
                ok = x < rate
                leak = ok or x < rate + exploit
                rows.append(
                    {"run_id": run_id, "size": size, "reward": reward, "seed": seed, "step": step}
                    | {"instance_id": f"v3-{t}-{i}", "family": t[:2], "template": t, "sample_idx": k}
                    | {"parse_ok": True, "robust_pass": ok, "leaky_pass": leak}
                )
    return rows


def write(path: Path, rows: list[dict], *, junk: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + ('{"truncated": \n' if junk else ""))


def final_marker(runs: Path, run_id: str, step: int) -> None:
    d = eval_step_dir(runs / run_id, step)
    d.mkdir(parents=True)
    atomic_write_json(d / COMPLETE, {"step": step, "final": True})


def tree(tmp_path: Path) -> tuple[Path, Path]:
    ev, runs = tmp_path / "evals", tmp_path / "runs"
    write(ev / "v3-base-4b" / "step-00000.jsonl", records("v3-base-4b", 0, 0.5, 0.25))
    for seed in (1, 2):
        rob, leak = f"v3-4b-rob-s{seed}", f"v3-4b-leak-s{seed}"
        write(ev / rob / "step-00100.jsonl", records(rob, 100, 0.5, 0.25))
        write(ev / rob / "step-01000.jsonl", records(rob, 1000, 1.0, 0.5), junk=True)
        write(ev / leak / "step-00100.jsonl", records(leak, 100, 0.5, 0.25, exploit=0.25))
        write(ev / leak / "step-01000.jsonl", records(leak, 1000, 0.25, 0.0, exploit=0.75))
        final_marker(runs, rob, 1000)
        final_marker(runs, leak, 1000)
    # seed 3 robust: still training, only step 100 evaluated (shown as "latest"); its leaky pair is missing
    write(ev / "v3-4b-rob-s3" / "step-00100.jsonl", records("v3-4b-rob-s3", 100, 0.75, 0.5))
    # a failed eval step is skipped, as compare.py does
    write(ev / "v3-4b-leak-s1" / "step-00200.jsonl", records("v3-4b-leak-s1", 200, 0.0, 0.0))
    (ev / "v3-4b-leak-s1" / "step-00200.failed").write_text("")
    # same-type of one run and the base
    sametype = list(TRAIN_TEMPLATES[:3])
    write(
        ev / "v3-4b-rob-s1" / "sametype" / "step-01000.jsonl", records("v3-4b-rob-s1", 1000, 1.0, 0, templates=sametype)
    )
    write(ev / "v3-base-4b" / "sametype" / "step-00000.jsonl", records("v3-base-4b", 0, 0.5, 0, templates=sametype))
    # v1 / v2 runs are not v3
    write(ev / "4b-rob-s1" / "step-01000.jsonl", records("v3-4b-rob-s1", 1000, 0.0, 0.0))
    write(ev / "v2-2b-rob-s1" / "step-01000.jsonl", records("v3-2b-rob-s1", 1000, 0.0, 0.0))
    return ev, runs


def test_load_picks_final_steps_and_skips_failed(tmp_path):
    data = v3_report.load(*tree(tmp_path))
    assert len(data.runs) == 12 and set(data.base) == {"4b", "2b"}
    assert data.runs["v3-4b-rob-s1"].step == 1000 and data.runs["v3-4b-rob-s1"].status == "final (step 1000)"
    assert data.runs["v3-4b-leak-s1"].step == 1000 and 200 not in data.runs["v3-4b-leak-s1"].heldout
    assert data.runs["v3-4b-rob-s3"].status == "latest (step 100)"
    assert data.runs["v3-2b-rob-s1"].status == "pending" and data.base["2b"].status == "pending"
    assert any("step-00200 (failed" in n for n in data.notes)
    assert any("skipped 1 unreadable line" in n for n in data.notes)


def test_near_and_far_are_reported_separately(tmp_path):
    data = v3_report.load(*tree(tmp_path))
    rob = v3_report.condition(data, "4b", "robust")
    near, n = v3_report.group_rates(rob, v3_report.by_kind("near"), lambda r: r.records)
    far, _ = v3_report.group_rates(rob, v3_report.by_kind("far"), lambda r: r.records)
    assert n == 3 and near is not None and far is not None
    assert near["true_success"] == (1.0 + 1.0 + 0.75) / 3 and far["true_success"] == (0.5 + 0.5 + 0.5) / 3
    leak = v3_report.condition(data, "4b", "leaky")
    near, n = v3_report.group_rates(leak, v3_report.by_kind("near"), lambda r: r.records)
    assert n == 2 and near is not None and (near["true_success"], near["exploit"]) == (0.25, 0.75)
    base, _ = v3_report.group_rates([data.base["4b"]], v3_report.by_kind("far"), lambda r: r.records)
    assert base is not None and base["true_success"] == 0.25


def test_markdown_tables_trajectory_bootstrap_and_sametype(tmp_path):
    data = v3_report.load(*tree(tmp_path))
    md = v3_report.markdown(data, n_boot=200, seed=0)
    assert md.startswith("# Project version v3 — exploratory (ADR 0007)") and "EXPLORATORY (ADR 0007)" in md
    assert "| base (thinking on) | - | 50.0% | 0.0% | 25.0% | 0.0% |" in md
    assert "| leaky | 2 | 25.0% | 75.0% | 0.0% | 75.0% |" in md
    assert "| F1 | 50.0% | 91.7% | 25.0% | 0.0% | 0.0% | 75.0% |" in md  # per family, near
    assert "| 100 | 3 / 2 | 58.3% / 33.3% | 50.0% / 25.0% | 0.0% / 0.0% | 25.0% / 25.0% |" in md  # trajectory
    assert "| 1000 | 2 / 2 | 100.0% / 50.0% |" in md
    boot = md[md.index("### Paired bootstrap") :]
    assert "| robust - leaky | +75.0 [" in boot  # seeds 1-2 paired (seed 3 has no leaky run)
    assert "| leaky - base |" in boot and "## 2B" in md
    assert md[md.index("## 2B") :].count("pending") > 10  # 2B: nothing yet, no crash
    assert "| 4b base | - | 50.0% | 0.0% | 0.0% |" in md and "| 4b robust | 1 | 100.0% | 0.0% | 0.0% |" in md
    assert "4b-rob-s1" not in md.replace("v3-4b-rob-s1", "") and "v2-" not in md


def test_contrast_needs_paired_seeds(tmp_path):
    data = v3_report.load(*tree(tmp_path))
    e = v3_report.contrast(data, "4b", "robust", "leaky", "near", "true_success", n_boot=200, seed=0)
    assert not isinstance(e, str) and e.point == 0.75
    assert v3_report.contrast(data, "2b", "robust", "leaky", "far", "exploit", n_boot=10, seed=0) == "pending"
    e = v3_report.contrast(data, "4b", "leaky", "base", "near", "exploit", n_boot=200, seed=0)
    assert not isinstance(e, str) and e.point == 0.75 and e.excludes_zero


def test_cli_without_any_data(tmp_path, capsys):
    argv = ["--evals", str(tmp_path / "nope"), "--runs", str(tmp_path / "nope"), "--out", "-", "--n-boot", "10"]
    assert v3_report.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Project version v3") and "| v3-4b-leak-s3 | pending | pending |" in out
    out_path = tmp_path / "reports" / "v3.md"
    assert v3_report.main([*argv[:-4], "--out", str(out_path), "--n-boot", "10"]) == 0 and out_path.is_file()
