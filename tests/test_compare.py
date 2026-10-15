"""Tests for eval/compare.py on fake trees from eval/fake_data.py (known effects)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from eval import compare
from eval.compare import (
    BEGIN_MARKER,
    END_MARKER,
    HELDOUT_IDS,
    Analysis,
    HypothesisResult,
    Protocol,
    analyze,
    check_protocol,
    load_dataset,
    load_protocol,
    read_train_log,
    render_markdown,
    replace_generated,
    write_report,
)
from eval.fake_data import eval_records, write_fake_tree
from eval.fake_data import run_manifest as fake_run_manifest
from eval.metrics import STATUS_FAIL, STATUS_INCOMPLETE, STATUS_PASS, EvalRecord, Verdict
from train.run_manifest import MANIFEST_FILE

N_BOOT = 1000
REPO = Path(__file__).resolve().parents[1]
FAKE: dict[str, Any] = {"final_step": 300, "eval_every": 100}


def _protocol(root: Path, samples: int = 4) -> Protocol | None:
    path = root / HELDOUT_IDS
    return load_protocol(path, samples=samples) if path.exists() else None


def _analyze(root: Path, samples: int = 4):
    data = load_dataset(root / "evals", root / "runs", _protocol(root, samples))
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


# ── eval protocol (PREREG §3: 250 frozen held-out instances x 4 samples) ─────


def _rewrite_evals(root: Path, pattern: str, change) -> list[Path]:
    """Apply ``change(rows) -> rows`` to every eval records file matching ``pattern``."""
    paths = sorted((root / "evals").glob(pattern))
    for p in paths:
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        p.write_text("".join(json.dumps(r) + "\n" for r in change(rows)))
    return paths


def test_three_of_four_samples_removed_everywhere_is_incomplete(tmp_path):
    """The case that slipped through: 250 samples per checkpoint instead of 1,000 counted as complete."""
    write_fake_tree(tmp_path, **FAKE)
    paths = _rewrite_evals(tmp_path, "*/step-*.jsonl", lambda rows: [r for r in rows if r["sample_idx"] == 0])
    assert len(paths) == 18 * 3 + 3
    an = _analyze(tmp_path)
    assert an.count("final") == 18 and all(s in an.data.base for s in ("0.8b", "2b", "4b"))
    assert not an.complete and not an.protocol_ok
    assert len(an.data.protocol_failures) == len(paths)
    for h in an.hypotheses:
        assert h.provisional, h.hypothesis
        assert any("fails the eval protocol" in g for g in h.gaps)
    md = render_markdown(an, None, tmp_path)
    assert "**Incomplete data: verdicts are provisional.**" in md
    assert "eval protocol: 21 checkpoint(s) in verdicts fail it" in md
    assert "| `2b-rob-s1@300` | yes | 250 of 1000 samples; 250 instance(s) without sample_idx 0..3" in md
    assert "| `2b-rob-s1@100` | no |" in md
    assert "Complete:" not in md


def test_one_short_final_checkpoint_only_affects_its_hypotheses(tmp_path):
    write_fake_tree(tmp_path, **FAKE)
    _rewrite_evals(tmp_path, "2b-rob-s1/step-00300.jsonl", lambda rows: [r for r in rows if r["sample_idx"] < 3])
    _rewrite_evals(tmp_path, "2b-leak-s1/step-00100.jsonl", lambda rows: [r for r in rows if r["sample_idx"] < 3])
    an = _analyze(tmp_path)
    hs = _by_h(an)
    assert not an.complete
    assert list(an.data.verdict_failures()) == [("2b-rob-s1", 300)]
    assert set(an.data.protocol_failures) == {("2b-rob-s1", 300), ("2b-leak-s1", 100)}
    assert hs["H1"].provisional and hs["H2"].provisional and not hs["H5"].provisional
    md = render_markdown(an, None, tmp_path)
    assert "2 of 57 checkpoint(s) fail it" in md
    assert "| `2b-leak-s1@100` | no |" in md


def test_only_an_intermediate_checkpoint_failing_keeps_data_complete(tmp_path):
    write_fake_tree(tmp_path, **FAKE)
    _rewrite_evals(tmp_path, "4b-leak-s2/step-00200.jsonl", lambda rows: rows[:-1])
    an = _analyze(tmp_path)
    assert an.complete and set(an.data.protocol_failures) == {("4b-leak-s2", 200)}


def test_instance_set_must_equal_frozen_ids_not_the_common_set(tmp_path):
    runs = [f"2b-{r}-s{s}" for r in ("rob", "leak") for s in (1, 2, 3)]
    write_fake_tree(tmp_path, run_ids=runs, base_sizes=("2b",), **FAKE)
    # every checkpoint lacks the same instance: the most-common-set rule would accept this
    _rewrite_evals(tmp_path, "*/step-*.jsonl", lambda rows: [r for r in rows if r["instance_id"] != "F1T3-000"])
    an = _analyze(tmp_path)
    assert not an.data.runs and len(an.data.dropped) == 6 * 3 + 1
    assert any("frozen 250-instance OOD-test set" in n for n in an.data.notes)
    assert all(
        v == ["1 of 250 OOD-test instances missing", "996 of 1000 samples"] for v in an.data.protocol_failures.values()
    )
    md = render_markdown(an, None, tmp_path)
    assert "| `2b-rob-s1@300` | no (dropped) |" in md
    # without a protocol the old rule keeps them, but the data never counts as complete
    data = load_dataset(tmp_path / "evals", tmp_path / "runs")
    assert len(data.runs) == 19
    an = analyze(data, n_boot=N_BOOT)
    assert not an.complete and all(h.provisional for h in an.hypotheses)
    assert "**Not checked:**" in render_markdown(an, None, tmp_path)


def test_grader_fields_and_duplicates_are_protocol_failures(tmp_path):
    runs = [f"2b-{r}-s{s}" for r in ("rob", "leak") for s in (1, 2, 3)]
    write_fake_tree(tmp_path, run_ids=runs, base_sizes=("2b",), **FAKE)

    def null_grader(rows):
        rows[0] = {**rows[0], "leaky_pass": None}
        rows[1] = {k: v for k, v in rows[1].items() if k != "robust_pass"}
        return rows

    _rewrite_evals(tmp_path, "2b-leak-s1/step-00300.jsonl", null_grader)
    _rewrite_evals(tmp_path, "2b-rob-s2/step-00300.jsonl", lambda rows: [*rows, rows[5]])
    an = _analyze(tmp_path)  # no crash on the duplicate
    fails = an.data.verdict_failures()
    assert fails[("2b-leak-s1", 300)][0] == "2 row(s) without boolean parse_ok/robust_pass/leaky_pass (skipped)"
    assert "998 of 1000 samples" in fails[("2b-leak-s1", 300)]
    assert fails[("2b-rob-s2", 300)] == ["1 duplicate sample(s) (first kept)"]
    assert not an.complete


def test_check_protocol_unit():
    recs = [
        EvalRecord("r", "2b", "robust", 1, 5, iid, "F1", "F1T3", k, True, True, True)
        for iid in ("a", "b")
        for k in (0, 1)
    ]
    p = Protocol(frozenset({"a", "b"}), 2)
    kept, fails = check_protocol(recs, p)
    assert kept == recs and fails == {}
    _, fails = check_protocol(recs, Protocol(frozenset({"a", "b"}), 3))
    assert fails[("r", 5)] == ["4 of 6 samples", "2 instance(s) without sample_idx 0..2 exactly once (e.g. a: [0, 1])"]
    _, fails = check_protocol([*recs[:3], dataclasses.replace(recs[3], sample_idx=0)], p)
    assert fails[("r", 5)] == [
        "1 duplicate sample(s) (first kept)",
        "3 of 4 samples",
        "1 instance(s) without sample_idx 0..1 exactly once (e.g. b: [0])",
    ]


def test_load_protocol_reads_samples_from_config(tmp_path):
    ids = tmp_path / "heldout.txt"
    ids.write_text("a\nb\n\n")
    p = load_protocol(ids)
    assert p.instance_ids == {"a", "b"} and p.samples == 4 and p.n_samples == 8
    real = load_protocol(REPO / HELDOUT_IDS)
    assert len(real.instance_ids) == 250 and real.n_samples == 1000
    ids.write_text("a\na\n")
    with pytest.raises(ValueError):
        load_protocol(ids)


# ── matched pairs (PREREG §1: same config, seed, start weights, step count) ──

PAIR_RUNS = [f"2b-{r}-s{s}" for r in ("rob", "leak") for s in (1, 2, 3)]


def _edit_manifest(root: Path, run_id: str, change) -> None:
    p = root / "runs" / run_id / MANIFEST_FILE
    p.write_text(json.dumps(change(json.loads(p.read_text()))))


def _pair_tree(root: Path, **kw) -> None:
    write_fake_tree(root, run_ids=PAIR_RUNS, base_sizes=("2b",), **{**FAKE, **kw})


def test_matched_pairs_verified(tmp_path):
    _pair_tree(tmp_path)
    data = load_dataset(tmp_path / "evals", tmp_path / "runs", _protocol(tmp_path))
    assert [(p.size, p.seed, p.status) for p in data.pairs] == [("2b", s, "ok") for s in (1, 2, 3)]
    assert all(not p.reasons for p in data.pairs)
    md = render_markdown(analyze(data, n_boot=N_BOOT), None, tmp_path)
    assert "| 2b | 1 | `2b-rob-s1`, `2b-leak-s1` | ok | — |" in md


def test_pair_with_different_data_is_rejected_and_excluded(tmp_path):
    _pair_tree(tmp_path)
    _edit_manifest(tmp_path, "2b-leak-s1", lambda m: {**m, "data": {**m["data"], "sha256": "f" * 64}})
    an = _analyze(tmp_path)
    pair = an.data.pair_of("2b-leak-s1")
    assert pair is not None and pair.status == "rejected" and pair.reasons == ["manifests differ in data"]
    assert {i.run_id: i.state for i in an.data.infos.values() if i.size == "2b"} == {
        "2b-rob-s1": "excluded",
        "2b-rob-s2": "final",
        "2b-rob-s3": "final",
        "2b-leak-s1": "excluded",
        "2b-leak-s2": "final",
        "2b-leak-s3": "final",
    }
    assert not any(rid in ("2b-rob-s1", "2b-leak-s1") for rid, _ in an.data.runs)
    hs = _by_h(an)
    assert _v(hs["H1"]).n["seeds"] == 2 and _v(hs["H2"]).n["seed_pairs"] == 2
    assert hs["H1"].provisional and hs["H2"].provisional
    assert any("excluded, matched pair rejected" in g and "2b-rob-s1" in g for g in hs["H1"].gaps)
    md = render_markdown(an, None, tmp_path)
    assert "| 2b | 1 | `2b-rob-s1`, `2b-leak-s1` | **rejected** | manifests differ in data |" in md
    assert "`2b-leak-s1` | excluded |" in md
    assert "matched pairs: 1 rejected, 0 unverified" in md


def test_pair_differing_only_in_reward_weights_is_kept(tmp_path):
    _pair_tree(tmp_path)
    _edit_manifest(tmp_path, "2b-leak-s2", lambda m: {**m, "reward_weights": {"robust": 0.5, "leaky": 0.5}})
    _edit_manifest(tmp_path, "2b-leak-s2", lambda m: {**m, "git": {"commit": "other", "dirty": True}})
    data = load_dataset(tmp_path / "evals", tmp_path / "runs", _protocol(tmp_path))
    pair = data.pair_of("2b-rob-s2")
    assert pair is not None and pair.status == "ok"
    assert pair.notes == ["git commits differ (fake vs other)"]


def test_pair_with_different_final_steps_is_rejected(tmp_path):
    _pair_tree(tmp_path)
    for d in ("runs", "evals"):
        shutil.rmtree(tmp_path / d / "2b-leak-s3")
    write_fake_tree(tmp_path, run_ids=["2b-leak-s3"], base_sizes=(), final_step=250, eval_every=100)
    _edit_manifest(tmp_path, "2b-leak-s3", lambda m: fake_run_manifest("2b", "leaky", 3, 300))
    an = _analyze(tmp_path)
    pair = an.data.pair_of("2b-leak-s3")
    assert pair is not None and pair.status == "rejected"
    assert pair.reasons == ["final steps differ (300 vs 250)"]
    assert _v(_by_h(an)["H2"]).n["seed_pairs"] == 2


def test_resume_change_and_wrong_label_reject_the_pair(tmp_path):
    _pair_tree(tmp_path)

    def resumed(m):
        m["starts"].append({"resume_from": "recovery/checkpoint-100", "git": None, "mismatch": ["data"]})
        return m

    _edit_manifest(tmp_path, "2b-rob-s1", resumed)
    _edit_manifest(tmp_path, "2b-leak-s2", lambda m: {**m, "reward": "robust"})
    data = load_dataset(tmp_path / "evals", tmp_path / "runs", _protocol(tmp_path))
    p1, p2 = data.pair_of("2b-rob-s1"), data.pair_of("2b-rob-s2")
    assert p1 is not None and p1.reasons == ["`2b-rob-s1` changed on a resume: data"]
    assert p2 is not None and p2.status == "rejected"
    assert p2.reasons[0] == "`2b-leak-s2` manifest says size/reward/seed ('2b', 'robust', 2)"


def test_missing_manifest_leaves_pair_unverified_and_provisional(tmp_path):
    _pair_tree(tmp_path, manifests=False)
    an = _analyze(tmp_path)
    assert {p.status for p in an.data.pairs} == {"unverified"}
    assert all(i.state == "final" for i in an.data.infos.values() if i.size == "2b")
    hs = _by_h(an)
    assert _v(hs["H2"]).n["seed_pairs"] == 3  # kept in the verdicts
    assert hs["H1"].provisional and hs["H2"].provisional
    assert any("matched pair not verified" in g for g in hs["H2"].gaps)
    assert not an.pairs_ok
    md = render_markdown(an, None, tmp_path)
    assert "| unverified | no run_manifest.json: `2b-rob-s1`, `2b-leak-s1` |" in md


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
    # step 100: a claim left behind next to .done (old watchers) does not hide a finished eval
    (ev / "2b-rob-s1/step-00100.claim").write_text("{}")
    (ev / "2b-rob-s1/step-00100.done").write_text("{}")
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
            "--heldout-ids", str(root / HELDOUT_IDS),
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
    assert all(p.stat().st_size > 1000 for p in paths)
    md = render_markdown(an, paths, tmp_path)
    assert "[`2-true-success`](figures/2-true-success-light.png)" in md


def test_plots_skip_excluded_runs(tmp_path):
    pytest.importorskip("matplotlib")
    _pair_tree(tmp_path)
    _edit_manifest(tmp_path, "2b-leak-s1", lambda m: {**m, "max_steps": 10})
    paths = compare.make_plots(_analyze(tmp_path), tmp_path / "figures")
    assert all(p.exists() for p in paths)


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


# ── ADR 0005 v2 runs stay out of the verdicts ────────────────────────────────


def test_v2_runs_never_reach_a_verdict(tmp_path):
    runs = ["2b-rob-s1", "2b-rob-s2", "2b-leak-s1", "2b-leak-s2"]
    write_fake_tree(tmp_path, run_ids=runs, base_sizes=("2b",), **FAKE)
    before = _analyze(tmp_path)
    for rid in ("2b-rob-s1", "2b-leak-s1"):  # v2 copies that would move every verdict if read
        v2 = f"v2-{rid}"
        rob = rid.startswith("2b-rob")
        shutil.copytree(tmp_path / "runs" / rid, tmp_path / "runs" / v2)
        shutil.copytree(tmp_path / "evals" / rid, tmp_path / "evals" / v2)
        _rewrite_evals(
            tmp_path,
            f"{v2}/step-*.jsonl",
            lambda rows, v2=v2, rob=rob: [{**r, "run_id": v2, "robust_pass": rob, "leaky_pass": True} for r in rows],
        )
    assert compare.parse_run_id("v2-2b-rob-s1") is None and compare.parse_run_id("v2-2b-leak-s3") is None
    after = _analyze(tmp_path)
    assert {k: _v(h) for k, h in _by_h(after).items()} == {k: _v(h) for k, h in _by_h(before).items()}
    data = load_dataset(tmp_path / "evals", tmp_path / "runs", _protocol(tmp_path))
    assert not any(rid.startswith("v2-") for rid, _ in data.runs)
    assert any("not in the run matrix, ignored: v2-2b-leak-s1, v2-2b-rob-s1" in n for n in data.notes)


# ── ADR 0007 v3 runs and the v3 base stay out of the verdicts ────────────────


def test_v3_runs_and_the_v3_base_never_reach_a_verdict(tmp_path):
    runs = ["2b-rob-s1", "2b-rob-s2", "2b-leak-s1", "2b-leak-s2"]
    write_fake_tree(tmp_path, run_ids=runs, base_sizes=("2b",), **FAKE)
    before = _analyze(tmp_path)
    for rid in ("2b-rob-s1", "2b-leak-s1"):  # v3 copies that would move every verdict if read
        v3 = f"v3-{rid}"
        rob = rid.startswith("2b-rob")
        shutil.copytree(tmp_path / "runs" / rid, tmp_path / "runs" / v3)
        shutil.copytree(tmp_path / "evals" / rid, tmp_path / "evals" / v3)
        _rewrite_evals(
            tmp_path,
            f"{v3}/step-*.jsonl",
            lambda rows, v3=v3, rob=rob: [{**r, "run_id": v3, "robust_pass": rob, "leaky_pass": True} for r in rows],
        )
    # the v3 base (`eval/heldout.py --checkpoint base --variant v3`): reward "base", size 2b, a v3 run id and
    # (here) the v1 instance ids, so only its run id keeps it out; sorted after base-2b it would otherwise win
    shutil.copytree(tmp_path / "evals" / "base-2b", tmp_path / "evals" / "v3-base-2b")
    _rewrite_evals(
        tmp_path,
        "v3-base-2b/step-*.jsonl",
        lambda rows: [{**r, "run_id": "v3-base-2b", "robust_pass": False, "leaky_pass": True} for r in rows],
    )
    assert compare.parse_run_id("v3-2b-rob-s1") is None and compare.parse_run_id("v3-4b-leak-s3") is None
    after = _analyze(tmp_path)
    assert {k: _v(h) for k, h in _by_h(after).items()} == {k: _v(h) for k, h in _by_h(before).items()}
    for protocol in (_protocol(tmp_path), None):
        data = load_dataset(tmp_path / "evals", tmp_path / "runs", protocol)
        assert not any(rid.startswith("v3-") for rid, _ in data.runs)
        assert data.base["2b"].run_id == "base-2b"
        assert any("ignored: v3-2b-leak-s1, v3-2b-rob-s1, v3-base-2b" in n for n in data.notes)


def test_blog_figures_written(tmp_path, full):
    pytest.importorskip("matplotlib")
    from eval import figures as blog_figures

    _, an = full
    paths = blog_figures.make_figures(an, tmp_path / "blog")
    names = {p.name for p in paths}
    assert len(paths) == len(blog_figures.FIGURES) * len(blog_figures.THEMES) * 2
    assert {"4-final-by-size-light.png", "4-final-by-size-dark.svg"} <= names
    assert all(p.stat().st_size > 1000 for p in paths)
