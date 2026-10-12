"""Tests for eval/compare.py on fake trees from eval/fake_data.py (known effects)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import compare
from eval.compare import (
    BEGIN_MARKER,
    END_MARKER,
    Analysis,
    HypothesisResult,
    analyze,
    load_dataset,
    read_train_log,
    render_markdown,
    replace_generated,
    write_report,
)
from eval.fake_data import eval_records, write_fake_tree
from eval.metrics import STATUS_FAIL, STATUS_INCOMPLETE, STATUS_PASS, Verdict

N_BOOT = 1000
REPO = Path(__file__).resolve().parents[1]
FAKE: dict[str, Any] = {"final_step": 300, "eval_every": 100}


def _analyze(root: Path):
    data = load_dataset(root / "evals", root / "runs")
    return analyze(data, n_boot=N_BOOT)


def _by_h(an: Analysis) -> dict[str, HypothesisResult]:
    return {h.hypothesis: h for h in an.hypotheses}


def _v(h: HypothesisResult) -> Verdict:
    assert h.verdict is not None, h.error
    return h.verdict


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root = tmp_path_factory.mktemp("full")
    write_fake_tree(root, **FAKE)
    return root, _analyze(root)


# ── end to end on the full fake tree ─────────────────────────────────────────


def test_full_verdicts_match_known_effects(full):
    _, an = full
    hs = _by_h(an)
    assert an.complete
    assert an.count("final") == 18
    for name in ("H1", "H2", "H5"):
        assert hs[name].verdict is not None, hs[name].error
        assert _v(hs[name]).status == STATUS_PASS, _v(hs[name]).reason
        assert not hs[name].provisional
        assert hs[name].label == "PASS"
    h5 = _v(hs["H5"]).numbers
    assert h5["exploit_0.8b"] < h5["exploit_2b"] < h5["exploit_4b"]


def test_full_tables_filled(full):
    _, an = full
    assert len(an.conditions) == 9  # 3 sizes x (base, robust, leaky)
    for c in an.conditions:
        assert set(c.cells) == {"true_success", "exploit", "format_failure"}
        for cell in c.cells.values():
            assert cell.ci[0] <= cell.point <= cell.ci[1]
        if c.reward != "base":
            assert c.n_seeds == 3 and set(c.steps) == {300}
    assert len(an.run_rows) == 21  # 18 runs + 3 base
    assert all(r.cells for r in an.run_rows)
    assert {r.info.state for r in an.run_rows if r.info} == {"final"}

    cond = {(c.size, c.reward): c for c in an.conditions}
    for size in ("0.8b", "2b", "4b"):
        rob, leak, base = (cond[(size, r)].cells for r in ("robust", "leaky", "base"))
        assert rob["true_success"].point > base["true_success"].point
        assert leak["exploit"].point > rob["exploit"].point


def test_full_h3_in_band_gains_more(full):
    _, an = full
    assert len(an.families) == 15
    for size in ("0.8b", "2b", "4b"):
        rows = [f for f in an.families if f.size == size]
        inside = [f.gain["robust"] for f in rows if f.in_band]
        outside = [f.gain["robust"] for f in rows if not f.in_band]
        assert inside and outside
        assert min(inside) > max(outside)


def test_full_markdown(full):
    root, an = full
    md = render_markdown(an, None, root)
    assert "Complete: all 18 runs" in md
    assert "Incomplete" not in md
    assert "provisional" not in md
    assert md.count("| **PASS** |") == 3
    assert "`2b-leak-s3` | final | 300 |" in md
    assert "### H3 (exploratory)" in md


# ── partial data ─────────────────────────────────────────────────────────────


def test_partial_data_gives_provisional_verdicts(tmp_path):
    runs = ["2b-rob-s1", "2b-rob-s2", "2b-rob-s3", "2b-leak-s1", "2b-leak-s2"]
    write_fake_tree(tmp_path, run_ids=runs, stop_at={"2b-rob-s2": 200}, base_sizes=("2b",), **FAKE)
    an = _analyze(tmp_path)
    hs = _by_h(an)
    assert not an.complete
    assert an.count("final") == 4 and an.count("partial") == 1 and an.count("missing") == 13

    h1 = hs["H1"]
    assert _v(h1).status in (STATUS_PASS, STATUS_FAIL)
    assert h1.provisional and h1.label.endswith("(provisional)")
    assert any("2b-rob-s2@200" in g for g in h1.gaps)
    assert _v(h1).n["seeds"] == 3

    h2 = hs["H2"]
    assert h2.verdict is not None and h2.provisional
    assert _v(h2).n["seed_pairs"] == 2
    assert any("2b-leak-s3" in g for g in h2.gaps)
    assert any("only 2 seed(s)" in n for n in _v(h2).notes)

    h5 = hs["H5"]
    assert _v(h5).status == STATUS_INCOMPLETE
    assert _v(h5).n["seeds_2b"] == 2 and _v(h5).n["seeds_4b"] == 0

    md = render_markdown(an, None, tmp_path)
    assert "**Incomplete data: verdicts are provisional.**" in md
    assert "Runs at final checkpoint: 4/18" in md
    assert "`4b-leak-s1` | missing |" in md
    assert "`2b-rob-s2` | partial | 200 |" in md


def test_no_data_at_all(tmp_path):
    an = _analyze(tmp_path)
    hs = _by_h(an)
    assert hs["H1"].verdict is None and hs["H1"].label == "NOT AVAILABLE"
    assert hs["H2"].verdict is None
    assert _v(hs["H5"]).status == STATUS_INCOMPLETE
    md = render_markdown(an, None, tmp_path)
    assert "Runs at final checkpoint: 0/18" in md
    assert md.count("| missing |") == 21


def test_missing_training_logs_flagged(tmp_path):
    runs = [f"2b-{r}-s{s}" for r in ("rob", "leak") for s in (1, 2, 3)]
    write_fake_tree(tmp_path, run_ids=runs, base_sizes=("2b",), train_logs=False, **FAKE)
    hs = _by_h(_analyze(tmp_path))
    assert _v(hs["H1"]).status == STATUS_PASS and not hs["H1"].provisional
    assert hs["H2"].verdict is None and "training logs" in (hs["H2"].error or "")
    assert any("no training log" in g for g in hs["H2"].gaps)


# ── loading rules ────────────────────────────────────────────────────────────


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_loading_skips_claimed_failed_truncated_and_takes_reruns(tmp_path):
    ev = tmp_path / "evals"
    kw = {"n_instances": 50, "n_samples": 2}
    _write(ev / "base-2b/step-00000.jsonl", eval_records("2b", "base", None, 0, **kw))
    full = eval_records("2b", "robust", 1, 100, **kw)
    _write(ev / "2b-rob-s1/step-00100.jsonl", full)
    # step 200: being evaluated; step 300: failed; step 150: half the instances
    _write(ev / "2b-rob-s1/step-00200.jsonl", eval_records("2b", "robust", 1, 200, **kw))
    (ev / "2b-rob-s1/step-00200.claim").write_text("{}")
    _write(ev / "2b-rob-s1/step-00300.jsonl", eval_records("2b", "robust", 1, 300, **kw))
    (ev / "2b-rob-s1/step-00300.failed").write_text("{}")
    _write(ev / "2b-rob-s1/step-00150.jsonl", eval_records("2b", "robust", 1, 150, **kw)[:50])
    # truncated last line
    with open(ev / "2b-rob-s1/step-00100.jsonl", "a") as f:
        f.write('{"run_id": "2b-rob-s1", "si')
    # a rerun replaces s2; an ablation run ID is ignored
    rerun = [dict(r, run_id="2b-rob-s2-r2") for r in eval_records("2b", "robust", 2, 100, **kw)]
    _write(ev / "2b-rob-s2/step-00100.jsonl", eval_records("2b", "robust", 2, 100, **kw))
    _write(ev / "2b-rob-s2-r2/step-00100.jsonl", rerun)
    lora = [dict(r, run_id="2b-rob-s3-lora") for r in eval_records("2b", "robust", 3, 100, **kw)]
    _write(ev / "2b-rob-s3-lora/step-00100.jsonl", lora)

    data = load_dataset(ev, tmp_path / "runs")
    s1 = data.infos[("2b", "robust", 1)]
    assert s1.eval_steps == [100] and s1.state == "partial"
    assert data.infos[("2b", "robust", 2)].run_id == "2b-rob-s2-r2"
    assert data.infos[("2b", "robust", 3)].state == "missing"
    notes = " ".join(data.notes)
    assert "step-00200 (being evaluated)" in notes
    assert "step-00300 (failed)" in notes
    assert "1 unreadable eval line" in notes
    assert "2b-rob-s1@150" in notes
    assert "replaced by a rerun: 2b-rob-s2" in notes
    assert "ignored: 2b-rob-s3-lora" in notes


def test_train_log_last_line_per_step_wins(tmp_path):
    p = tmp_path / "train_metrics.jsonl"
    rows = [{"step": s, "reward": 0.1} for s in (1, 2, 3)] + [{"step": 2, "reward": 0.9}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"step": 4, "rew')
    log = read_train_log(p)
    assert log is not None
    assert [r["step"] for r in log] == [1, 2, 3]
    assert log[1]["reward"] == 0.9
    assert read_train_log(tmp_path / "nope.jsonl") is None


# ── report markers ───────────────────────────────────────────────────────────


def test_replace_generated_only_touches_marked_region():
    head = "# Results\n\nHand-written intro.\n\n"
    tail = "\n\n## Limits\n\nHand-written limits.\n"
    text = f"{head}{BEGIN_MARKER}\nold block\n{END_MARKER}{tail}"
    out = replace_generated(text, "new block\n")
    assert out == f"{head}{BEGIN_MARKER}\nnew block\n{END_MARKER}{tail}"
    assert replace_generated(out, "new block") == out  # idempotent
    with pytest.raises(ValueError):
        replace_generated("no markers", "x")
    with pytest.raises(ValueError):
        replace_generated(f"{END_MARKER}\n{BEGIN_MARKER}", "x")
    with pytest.raises(ValueError):
        replace_generated(text + BEGIN_MARKER, "x")


def test_write_report_keeps_hand_text_and_creates_missing(tmp_path):
    p = tmp_path / "r.md"
    write_report(p, "first")
    assert BEGIN_MARKER in p.read_text() and "first" in p.read_text()
    p.write_text(p.read_text().replace("# Results", "# Results\n\nmy notes"))
    write_report(p, "second")
    text = p.read_text()
    assert "my notes" in text and "second" in text and "first" not in text


def test_skeleton_report_has_one_marker_pair():
    text = (REPO / "reports" / "results.md").read_text()
    assert text.count(BEGIN_MARKER) == 1 and text.count(END_MARKER) == 1
    assert text.index(BEGIN_MARKER) < text.index(END_MARKER)


# ── CLI, plots, fake data ────────────────────────────────────────────────────


def test_cli_writes_report_between_markers(tmp_path, full):
    root, _ = full
    report = tmp_path / "results.md"
    report.write_text(f"intro\n{BEGIN_MARKER}\n{END_MARKER}\noutro\n")
    rc = compare.main(
        [
            "--evals", str(root / "evals"),
            "--runs", str(root / "runs"),
            "--report", str(report),
            "--n-boot", "200",
            "--no-plots",
        ]
    )  # fmt: skip
    assert rc == 0
    text = report.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert "| **PASS** |" in text and "Plots not written" in text


def test_cli_root_reads_and_writes_inside_root(tmp_path):
    write_fake_tree(tmp_path, run_ids=["2b-rob-s1"], base_sizes=("2b",), **FAKE)
    assert compare.main(["--root", str(tmp_path), "--n-boot", "100", "--no-plots"]) == 0
    text = (tmp_path / "results.md").read_text()
    assert BEGIN_MARKER in text and "`2b-rob-s1` | final | 300 |" in text


def test_plots_written(tmp_path, full):
    pytest.importorskip("matplotlib")
    _, an = full
    paths = compare.make_plots(an, tmp_path / "figures")
    assert [p.name for p in paths] == [compare.FIG_CURVES, compare.FIG_EXPLOIT]
    assert all(p.stat().st_size > 1000 for p in paths)
    md = render_markdown(an, paths, tmp_path)
    assert "![reward_vs_true_success](figures/reward_vs_true_success.png)" in md


def test_plots_with_partial_data(tmp_path):
    pytest.importorskip("matplotlib")
    write_fake_tree(tmp_path, run_ids=["2b-leak-s1"], base_sizes=("2b",), **FAKE)
    paths = compare.make_plots(_analyze(tmp_path), tmp_path / "figures")
    assert all(p.exists() for p in paths)


def test_fake_data_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        write_fake_tree(d, run_ids=["0.8b-leak-s2"], base_sizes=("0.8b",), **FAKE)
    for rel in (
        "evals/0.8b-leak-s2/step-00300.jsonl",
        "evals/base-0.8b/step-00000.jsonl",
        "runs/0.8b-leak-s2/train_metrics.jsonl",
    ):
        assert (a / rel).read_bytes() == (b / rel).read_bytes()
    marker = json.loads((a / "runs/0.8b-leak-s2/eval/step-00300/COMPLETE").read_text())
    assert marker == {"step": 300, "final": True}
