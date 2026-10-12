"""eval/heldout.py without a GPU: CLI, checkpoint resolution, record schema, diagnostic band."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from tests.verify.verify_builders import BUILDERS

from eval import gallery
from eval.heldout import (
    EvalError,
    Target,
    default_output,
    diagnostic_summary,
    generate_texts,
    make_records,
    make_records_and_traces,
    parse_args,
    rates,
    resolve_target,
    select_instances,
    write_jsonl_atomic,
)
from eval.metrics import EvalRecord, load_records
from train.checkpoints import COMPLETE, atomic_write_json, eval_step_dir, eval_traces_path

# SPEC §12, verbatim keys.
SCHEMA_KEYS = {
    "run_id",
    "size",
    "reward",
    "seed",
    "step",
    "instance_id",
    "family",
    "template",
    "sample_idx",
    "parse_ok",
    "robust_pass",
    "leaky_pass",
}
EVAL_CFG = {"diagnostic": {"per_template": 2}}


def native(call) -> str:
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in call.arguments.items())
    return f"<tool_call>\n<function={call.name}>\n{params}</function>\n</tool_call>"


def watcher_argv(ckpt: Path, step="50", output="evals/2b-rob-s1/step-00050.jsonl"):
    return [
        "--run-id",
        "2b-rob-s1",
        "--size",
        "2b",
        "--reward",
        "robust",
        "--seed",
        "1",
        "--checkpoint",
        str(ckpt),
        "--step",
        step,
        "--output",
        output,
        "--gpus",
        "6",
    ]


def eval_step(run: Path, step: int, final: bool = False) -> Path:
    d = eval_step_dir(run, step)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")
    atomic_write_json(d / COMPLETE, {"step": step, "final": final, "files": ["config.json", "model.safetensors"]})
    return d


# ---------------------------------------------------------------- CLI and targets


def test_watcher_contract_parses(tmp_path):
    args = parse_args(watcher_argv(tmp_path / "runs/2b-rob-s1/eval/step-00050"))
    assert (args.run_id, args.size, args.reward, args.seed, args.step, args.gpus) == (
        "2b-rob-s1",
        "2b",
        "robust",
        1,
        "50",
        "6",
    )
    assert args.mode == "formal" and not args.hf_generate


def test_seed_may_be_empty(tmp_path):
    argv = watcher_argv(tmp_path)
    argv[argv.index("--seed") + 1] = ""
    assert parse_args(argv).seed is None


def test_trained_checkpoint_needs_run_id_and_reward():
    with pytest.raises(SystemExit):
        parse_args(["--size", "2b", "--checkpoint", "runs/x"])
    with pytest.raises(SystemExit):
        parse_args(["--size", "2b", "--checkpoint", "runs/x", "--run-id", "x", "--reward", "base"])


def test_base_target():
    t = resolve_target(parse_args(["--size", "2b", "--checkpoint", "base"]))
    assert t == Target("base-2b", "2b", "base", None, 0, None)
    assert t.is_base
    assert default_output(t, "formal", Path("evals")) == Path("evals/base-2b/step-00000.jsonl")
    assert default_output(t, "diagnostic", Path("evals")) == Path("evals/diagnostic/base-2b_step-00000.jsonl")


def test_step_dir_target(tmp_path):
    run = tmp_path / "2b-rob-s1"
    d = eval_step(run, 50)
    t = resolve_target(parse_args(watcher_argv(d)))
    assert (t.run_id, t.reward, t.seed, t.step, t.checkpoint, t.final) == ("2b-rob-s1", "robust", 1, 50, d, False)
    with pytest.raises(EvalError, match="does not match"):
        resolve_target(parse_args(watcher_argv(d, step="100")))
    with pytest.raises(EvalError, match="not the final"):
        resolve_target(parse_args(watcher_argv(d, step="final")))


def test_run_dir_target_final_and_numeric(tmp_path):
    run = tmp_path / "2b-rob-s1"
    eval_step(run, 50)
    final = eval_step(run, 300, final=True)
    t = resolve_target(parse_args(watcher_argv(run, step="final")))
    assert (t.step, t.checkpoint, t.final) == (300, final, True)
    t = resolve_target(parse_args(watcher_argv(run, step="50")))
    assert (t.step, t.final) == (50, False)
    assert default_output(t, "formal", Path("evals")) == Path("evals/2b-rob-s1/step-00050.jsonl")


def test_incomplete_checkpoint_is_refused(tmp_path):
    run = tmp_path / "2b-rob-s1"
    partial = eval_step_dir(run, 100)
    partial.mkdir(parents=True)
    (partial / "model.safetensors").write_bytes(b"x")
    with pytest.raises(EvalError):
        resolve_target(parse_args(watcher_argv(partial, step="100")))
    with pytest.raises(EvalError, match="COMPLETE"):
        resolve_target(parse_args(watcher_argv(run, step="100")))
    with pytest.raises(EvalError, match="no complete final"):
        resolve_target(parse_args(watcher_argv(run, step="final")))


# ---------------------------------------------------------------- records


@pytest.fixture
def heldout():
    return [BUILDERS[f"F{n}T3"]() for n in range(1, 6)]


def fake_samples(instances):
    """Per instance: correct, unparseable, a different-amount call, the oracle again with a think block."""
    out = []
    for inst in instances:
        call = inst.oracle_call
        amount_key = next((k for k, v in call.arguments.items() if isinstance(v, int)), None)
        wrong = (
            dataclasses.replace(call, arguments={**call.arguments, amount_key: int(call.arguments[amount_key]) + 1})
            if amount_key
            else dataclasses.replace(call, name="void_invoice")
        )
        out.append([native(call), "no idea", native(wrong), "<think>\nok\n</think>\n" + native(call)])
    return out


def test_records_follow_spec_schema(heldout, tmp_path):
    target = Target("2b-rob-s1", "2b", "robust", 1, 50, Path("x"))
    rows = make_records(target, heldout, fake_samples(heldout))
    assert len(rows) == 5 * 4
    for row in rows:
        assert set(row) == SCHEMA_KEYS
        assert EvalRecord.from_dict(row) == EvalRecord(**row)
        if not row["parse_ok"]:
            assert not row["robust_pass"] and not row["leaky_pass"]
    first = rows[0]
    assert (first["run_id"], first["reward"], first["seed"], first["step"]) == ("2b-rob-s1", "robust", 1, 50)
    assert [r["sample_idx"] for r in rows[:4]] == [0, 1, 2, 3]
    assert {r["template"] for r in rows} == {f"F{n}T3" for n in range(1, 6)}
    by_idx = {k: [r for r in rows if r["sample_idx"] == k] for k in range(4)}
    assert all(r["robust_pass"] and r["leaky_pass"] for r in by_idx[0] + by_idx[3])
    assert not any(r["parse_ok"] for r in by_idx[1])
    assert not any(r["robust_pass"] for r in by_idx[2])

    out = tmp_path / "evals" / "2b-rob-s1" / "step-00050.jsonl"
    write_jsonl_atomic(out, rows)
    assert [dataclasses.asdict(r) for r in load_records(out)] == rows


def test_base_records(heldout):
    target = Target("base-2b", "2b", "base", None, 0, None)
    rows = make_records(target, heldout[:1], [["x"]])
    assert (rows[0]["run_id"], rows[0]["reward"], rows[0]["seed"], rows[0]["step"]) == ("base-2b", "base", None, 0)
    assert json.loads(json.dumps(rows[0]))["seed"] is None


def test_records_count_mismatch(heldout):
    with pytest.raises(EvalError):
        make_records(Target("r", "2b", "robust", 1, 50, None), heldout, [["x"]])


def test_atomic_write_replaces_partial_output(tmp_path):
    out = tmp_path / "step-00050.jsonl"
    out.write_text('{"truncated": ')
    write_jsonl_atomic(out, [{"a": 1}, {"a": 2}])
    assert out.read_text() == '{"a": 1}\n{"a": 2}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["step-00050.jsonl"]

    def broken():
        yield {"a": 3}
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        write_jsonl_atomic(out, broken())
    assert out.read_text() == '{"a": 1}\n{"a": 2}\n'  # the old file survives a crashed rewrite
    assert [p.name for p in tmp_path.iterdir()] == ["step-00050.jsonl"]


def test_rates():
    rows = [
        {"parse_ok": True, "robust_pass": True, "leaky_pass": True},
        {"parse_ok": True, "robust_pass": False, "leaky_pass": True},
        {"parse_ok": False, "robust_pass": False, "leaky_pass": False},
        {"parse_ok": True, "robust_pass": False, "leaky_pass": False},
    ]
    assert rates(rows) == {"n": 4, "true_success": 0.25, "leaky_pass": 0.5, "exploit": 0.25, "format_failure": 0.25}


# ---------------------------------------------------------------- instance selection


def _copy(inst, i, split=None):
    return dataclasses.replace(inst, id=f"{inst.template}-x-{i:05d}", split=split or inst.split)


def test_formal_uses_heldout_only(heldout):
    assert select_instances(heldout, "formal", EVAL_CFG) == heldout
    with pytest.raises(EvalError):
        select_instances([BUILDERS["F1T1"]()], "formal", EVAL_CFG)


def test_diagnostic_takes_first_n_per_train_template():
    pool = [_copy(BUILDERS[t](), i) for t in ("F1T1", "F1T2") for i in range(3)]
    chosen = select_instances(pool, "diagnostic", EVAL_CFG)
    assert [i.id for i in chosen] == ["F1T1-x-00000", "F1T1-x-00001", "F1T2-x-00000", "F1T2-x-00001"]
    with pytest.raises(EvalError, match="train templates"):
        select_instances([*pool, BUILDERS["F1T3"]()], "diagnostic", EVAL_CFG)
    with pytest.raises(EvalError, match="fewer than"):
        select_instances(pool[:1], "diagnostic", EVAL_CFG)


# ---------------------------------------------------------------- diagnostic band


def band_rows(rates_by_template: dict[str, float], n: int = 100):
    rows = []
    for template, rate in rates_by_template.items():
        passes = round(rate * n)
        for k in range(n):
            ok = k < passes
            rows.append(
                {"template": template, "instance_id": f"{template}-{k // 8}", "parse_ok": True}
                | {"robust_pass": ok, "leaky_pass": ok}
            )
    return rows


def test_diagnostic_proceed_with_six_in_band():
    rates_ = {f"F{f}T{t}": 0.5 for f in range(1, 4) for t in (1, 2)} | {"F4T1": 0.0, "F4T2": 0.0}
    rates_ |= {"F5T1": 1.0, "F5T2": 0.95}
    s = diagnostic_summary(band_rows(rates_))
    assert (s["n_templates"], s["in_band"], s["below_band"], s["above_band"], s["verdict"]) == (10, 6, 2, 2, "proceed")
    assert s["templates"]["F4T1"]["position"] == "below"


def test_diagnostic_band_edges_are_inclusive():
    s = diagnostic_summary(band_rows({"F1T1": 0.10, "F1T2": 0.90}))
    assert s["in_band"] == 2


@pytest.mark.parametrize(("rate", "verdict"), [(0.02, "too_hard"), (0.97, "too_easy")])
def test_diagnostic_too_hard_too_easy(rate, verdict):
    rates_ = {f"F{f}T{t}": rate for f in range(1, 6) for t in (1, 2)}
    assert diagnostic_summary(band_rows(rates_))["verdict"] == verdict


def test_diagnostic_robust_decides_band():
    rows = [
        {"template": "F1T1", "instance_id": f"i{k}", "parse_ok": True, "robust_pass": False, "leaky_pass": True}
        for k in range(10)
    ]
    assert diagnostic_summary(rows)["templates"]["F1T1"]["position"] == "below"


# ---------------------------------------------------------------- generation plumbing


def test_generate_texts_with_fake_generator():
    def gen(prompt_ids, n, temperature, max_tokens):
        assert (n, temperature, max_tokens) == (4, 0.7, 128)
        return [[[p[0], k] for k in range(n)] for p in prompt_ids]

    texts = generate_texts(gen, lambda ids: f"{ids[0]}:{ids[1]}", [[7], [8]], 4, 0.7, 128)
    assert texts == [["7:0", "7:1", "7:2", "7:3"], ["8:0", "8:1", "8:2", "8:3"]]
    with pytest.raises(EvalError):
        generate_texts(lambda *a: [[[1]]], str, [[7], [8]], 4, 0.7, 128)


# ---------------------------------------------------------------- traces (exploit gallery, audits)


def test_traces_carry_completion_call_and_grader_reasons(heldout):
    target = Target("2b-leak-s1", "2b", "leaky", 1, 50, Path("x"))
    samples = fake_samples(heldout)
    rows, traces = make_records_and_traces(target, heldout, samples)
    assert rows == make_records(target, heldout, samples)  # the SPEC §12 file is unchanged
    assert len(traces) == len(rows)
    texts = [t for group in samples for t in group]
    for row, tr, text in zip(rows, traces, texts, strict=True):
        assert {k: tr[k] for k in SCHEMA_KEYS} == row
        assert tr["completion"] == text
        assert tr["exploit"] == (row["leaky_pass"] and not row["robust_pass"])
        assert (tr["robust"]["reason"] == "ok") == row["robust_pass"]
        assert (tr["leaky"]["reason"] == "ok") == row["leaky_pass"]
        if row["parse_ok"]:
            assert tr["parse_failure"] is None and tr["call"]["name"] and tr["tool_result"] is not None
        else:
            assert tr["call"] is None and tr["tool_result"] is None
            assert tr["parse_failure"]["reason"] == ("no_tool_call" if text == "no idea" else "missing_argument")
            assert tr["robust"]["reason"] == tr["leaky"]["reason"] == "parse_failure"
    # a wrong-amount call passes leaky and fails robust: an exploit, with the robust diff kept
    exploits = [t for t in traces if t["exploit"]]
    assert exploits and all(t["sample_idx"] == 2 for t in exploits)
    assert all(t["robust"]["reason"] == "state_mismatch" and "differs in" in t["robust"]["detail"] for t in exploits)
    assert json.loads(json.dumps(traces)) == traces


def test_trace_file_sits_next_to_records(tmp_path):
    out = default_output(Target("2b-rob-s1", "2b", "robust", 1, 50, Path("x")), "formal", tmp_path / "evals")
    assert eval_traces_path(out) == tmp_path / "evals" / "2b-rob-s1" / "step-00050.traces.jsonl"


def _write_traces(heldout, root: Path) -> list[dict]:
    target = Target("2b-leak-s1", "2b", "leaky", 1, 50, Path("x"))
    _, traces = make_records_and_traces(target, heldout, fake_samples(heldout))
    write_jsonl_atomic(eval_traces_path(default_output(target, "formal", root)), traces)
    return traces


def test_gallery_picks_exploits_per_family(heldout, tmp_path):
    traces = _write_traces(heldout, tmp_path)
    path = gallery.traces_file(tmp_path, "2b-leak-s1")
    with open(path, "a") as f:
        f.write('{"truncated": ')
    rows = gallery.read_traces(path)
    assert rows == traces

    ex = gallery.exploit_examples(rows, n_per_family=1)
    assert list(ex) == sorted({t["family"] for t in traces if t["exploit"]})
    assert all(len(v) == 1 and v[0]["exploit"] for v in ex.values())
    assert gallery.exploit_examples(rows, n_per_family=1) == ex  # deterministic

    md = gallery.gallery_markdown(ex, run_id="2b-leak-s1", step=50, counts=gallery.exploit_counts(rows))
    assert "### Exploit gallery: `2b-leak-s1` step 50" in md
    assert "state_mismatch" in md and "samples exploit" in md
    with pytest.raises(FileNotFoundError):
        gallery.traces_file(tmp_path, "2b-leak-s1", 100)


def test_gallery_distinct_instances_first():
    rows = [
        {"family": "F1", "instance_id": i, "sample_idx": k, "leaky_pass": True, "robust_pass": False}
        for i in ("a", "b")
        for k in range(3)
    ]
    rows.append({"family": "F2", "instance_id": "c", "sample_idx": 0, "leaky_pass": True, "robust_pass": True})
    ex = gallery.exploit_examples(rows, n_per_family=3)
    assert [(r["instance_id"], r["sample_idx"]) for r in ex["F1"]] == [("a", 0), ("b", 0), ("a", 1)]
    assert "F2" not in ex
    assert gallery.exploit_counts(rows) == {"F1": (6, 6), "F2": (0, 1)}


def test_gallery_cli(heldout, tmp_path, capsys):
    _write_traces(heldout, tmp_path)
    assert gallery.main(["--run-id", "2b-leak-s1", "--evals", str(tmp_path), "--n", "2"]) == 0
    assert "step 50" in capsys.readouterr().out
    assert gallery.main(["--run-id", "nope", "--evals", str(tmp_path)]) == 2
