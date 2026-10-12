"""eval/heldout.py and eval/sametype_batch.py on the v3 task set (ADR 0007), without a GPU.

The GPU parts of `eval.heldout.run` (the model export, tokenizer and vLLM) are replaced by fakes, so the whole
eval path runs on synthetic v3 data: settings, prompt budget, context, output paths, records and the gate.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml
from tests.v3_fixtures import v3_instance, write_v3_data

from envlab.tasks.v3 import FAR_TEMPLATES, NEAR_TEMPLATES, TRAIN_TEMPLATES, heldout_kind
from eval import heldout, sametype_batch
from eval.heldout import (
    EvalError,
    Target,
    diagnostic_summary,
    is_v3_data,
    kind_rates,
    parse_args,
    resolve_settings,
    resolve_target,
    select_instances,
    summary_markdown,
    v3_gate,
)
from train.checkpoints import COMPLETE, atomic_write_json, eval_step_dir
from train.grpo import DEFAULT_CONFIG, THINK_OFF_SUFFIX
from train.run_manifest import MANIFEST_FILE


def eval_step(run: Path, step: int, final: bool = False, **marker) -> Path:
    d = eval_step_dir(run, step)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    atomic_write_json(d / COMPLETE, {"step": step, "final": final, **marker})
    return d


def v3_run(runs: Path, run_id: str, step: int = 100, final: bool = True) -> Path:
    """A finished v3 run dir: an eval step and a run manifest that records the variant."""
    rdir = runs / run_id
    d = eval_step(rdir, step, final, variant="v3")
    size, r, seed = run_id.split("-")[1:]
    reward = "robust" if r == "rob" else "leaky"
    atomic_write_json(rdir / MANIFEST_FILE, {"size": size, "reward": reward, "seed": int(seed[1:]), "variant": "v3"})
    return d


# ---------------------------------------------------------------- settings


def test_v3_base_target_and_paths_never_collide_with_v1_base():
    t = resolve_target(parse_args(["--size", "4b", "--checkpoint", "base", "--variant", "v3"]))
    assert t == Target("v3-base-4b", "4b", "base", None, 0, None)
    v1 = resolve_target(parse_args(["--size", "4b", "--checkpoint", "base"]))
    assert v1.run_id == "base-4b"
    for mode, path in (
        ("formal", "evals/v3-base-4b/step-00000.jsonl"),
        ("diagnostic", "evals/diagnostic/v3-base-4b_step-00000.jsonl"),
        ("sametype", "evals/v3-base-4b/sametype/step-00000.jsonl"),
    ):
        args = parse_args(["--size", "4b", "--checkpoint", "base", "--variant", "v3", "--mode", mode])
        assert resolve_settings(args, resolve_target(args)).output == Path(path)
        plain = parse_args(["--size", "4b", "--checkpoint", "base", "--mode", mode])
        assert resolve_settings(plain, resolve_target(plain)).output != Path(path)


def test_variant_flag_is_for_the_base_model_only():
    with pytest.raises(SystemExit):
        parse_args(["--size", "4b", "--checkpoint", "runs/x", "--run-id", "x", "--reward", "robust", "--variant", "v3"])
    args = parse_args(
        ["--size", "4b", "--checkpoint", "base", "--variant", "v3", "--mode", "sametype", "--thinking", "off"]
    )
    with pytest.raises(EvalError, match="drop --thinking"):  # v3 is evaluated with thinking on, as trained
        resolve_settings(args, resolve_target(args))
    with pytest.raises(SystemExit):  # the formal eval takes no cap override, variant or not
        parse_args(["--size", "4b", "--checkpoint", "base", "--variant", "v3", "--max-tokens", "512"])


@pytest.mark.parametrize("mode", ["formal", "diagnostic", "sametype"])
def test_v3_base_settings(mode):
    args = parse_args(["--size", "2b", "--checkpoint", "base", "--variant", "v3", "--mode", mode])
    st = resolve_settings(args, resolve_target(args))
    assert (st.variant, st.thinking, st.max_tokens, st.max_prompt_tokens, st.max_model_len) == (
        "v3",
        True,
        1024,
        2048,
        3072,
    )
    split = {"formal": "heldout", "diagnostic": "train", "sametype": "sametype"}[mode]
    assert st.data_dir == Path("data/v3") and st.split_path == Path(f"data/v3/{split}.jsonl")


def test_v3_run_checkpoint_is_evaluated_as_trained_with_the_watcher_command(tmp_path):
    step = v3_run(tmp_path / "runs", "v3-4b-rob-s1")
    argv = ["--run-id", "v3-4b-rob-s1", "--size", "4b", "--reward", "robust", "--seed", "1"]
    argv += [
        "--checkpoint",
        str(step),
        "--step",
        "100",
        "--output",
        "evals/v3-4b-rob-s1/step-00100.jsonl",
        "--gpus",
        "6",
    ]
    args = parse_args(argv)  # configs/allocation.yaml commands.eval, unchanged
    st = resolve_settings(args, resolve_target(args))
    assert (st.variant, st.thinking, st.max_tokens, st.max_prompt_tokens, st.max_model_len) == (
        "v3",
        True,
        1024,
        2048,
        3072,
    )
    assert st.split_path == Path("data/v3/heldout.jsonl")
    args = parse_args([*argv[:-4], "--mode", "sametype", "--data-dir", str(tmp_path / "d")])
    st = resolve_settings(args, resolve_target(args))
    assert st.split_path == tmp_path / "d" / "sametype.jsonl"  # --data-dir still wins
    assert st.output == Path("evals/v3-4b-rob-s1/sametype/step-00100.jsonl")


def test_v1_and_v2_settings_are_unchanged(tmp_path):
    v1 = parse_args(["--size", "2b", "--checkpoint", "base"])
    st = resolve_settings(v1, resolve_target(v1))
    assert (st.variant, st.thinking, st.max_tokens, st.max_prompt_tokens, st.max_model_len) == (
        None,
        False,
        256,
        1536,
        2048,
    )
    assert st.split_path == Path("data/heldout.jsonl") and st.output == Path("evals/base-2b/step-00000.jsonl")
    think = parse_args(["--size", "2b", "--checkpoint", "base", "--mode", "diagnostic", "--thinking", "on"])
    st = resolve_settings(think, resolve_target(think))
    assert (st.thinking, st.max_tokens, st.max_model_len, st.max_prompt_tokens) == (True, 1024, 2560, 1536)
    assert st.output == Path("evals/diagnostic/base-2b_think_step-00000.jsonl")
    run = tmp_path / "v2-2b-rob-s1"
    step = eval_step(run, 50)
    atomic_write_json(run / MANIFEST_FILE, {"variant": "v2-think"})
    v2 = parse_args(["--run-id", run.name, "--size", "2b", "--reward", "robust", "--checkpoint", str(step)])
    st = resolve_settings(v2, resolve_target(v2))
    assert (st.variant, st.thinking, st.max_tokens, st.max_model_len, st.max_prompt_tokens) == (
        "v2-think",
        True,
        1024,
        2560,
        1536,
    )
    assert st.split_path == Path("data/heldout.jsonl") and st.output == Path("evals/v2-2b-rob-s1/step-00050.jsonl")


# ---------------------------------------------------------------- data and instances


def test_v3_data_detection(tmp_path):
    assert is_v3_data(write_v3_data(tmp_path / "a"))
    assert not is_v3_data(write_v3_data(tmp_path / "b", spec_version="1.0"))
    no_manifest = write_v3_data(tmp_path / "c", spec_version=None)
    assert not is_v3_data(no_manifest) and is_v3_data(no_manifest, [v3_instance("F1A", 0, "train")])
    from tests.verify.verify_builders import BUILDERS

    assert not is_v3_data(no_manifest, [v3_instance("F1A", 0, "train"), BUILDERS["F1T1"]()])


def test_v3_diagnostic_takes_train_templates_only():
    cfg = {"diagnostic": {"per_template": 2}}
    pool = [v3_instance(t, k, "train") for t in TRAIN_TEMPLATES for k in range(3)]
    chosen = select_instances(pool, "diagnostic", cfg)
    assert len(chosen) == 30 and {i.template for i in chosen} == set(TRAIN_TEMPLATES)
    for t in (NEAR_TEMPLATES[0], FAR_TEMPLATES[0]):  # held-out templates never in the diagnostic, whatever the split
        with pytest.raises(EvalError, match="train templates"):
            select_instances([*pool, v3_instance(t, 0, "train")], "diagnostic", cfg)


# ---------------------------------------------------------------- the ADR 0007 §5 gate


def band_rows(rates_by_template: dict[str, float], leaky: float | None = None, n: int = 100) -> list[dict]:
    """Rows with the given robust pass rate per template; leaky pass = robust, or ``leaky`` everywhere."""
    rows = []
    for template, rate in rates_by_template.items():
        for k in range(n):
            ok = k < round(rate * n)
            leak = ok if leaky is None else k < round(leaky * n)
            rows.append(
                {"template": template, "instance_id": f"{template}-{k // 8}", "parse_ok": True}
                | {
                    "robust_pass": ok,
                    "leaky_pass": leak,
                }
            )
    return rows


def rates_with(in_band: int) -> dict[str, float]:
    """15 templates, ``in_band`` of them at 50% robust pass, the rest at 95%."""
    return {t: (0.5 if i < in_band else 0.95) for i, t in enumerate(TRAIN_TEMPLATES)}


@pytest.mark.parametrize(
    ("in_band", "leaky", "verdict"),
    [(9, 0.90, "go"), (9, 0.91, "no-go"), (8, 0.50, "no-go"), (15, 0.90, "go"), (9, None, "go")],
)
def test_v3_gate_4b_edges(in_band, leaky, verdict):
    s = diagnostic_summary(band_rows(rates_with(in_band), leaky))
    g = v3_gate(s, "4b")
    assert (g["in_band"], g["n_templates"], g["verdict"]) == (in_band, 15, verdict)
    if leaky is not None:
        assert g["leaky_pass"] == pytest.approx(leaky)


def test_v3_gate_4b_leaky_ceiling_when_robust_equals_leaky():
    s = diagnostic_summary(band_rows(rates_with(9)))  # leaky = robust: 9 x 0.5 + 6 x 0.95 over 15 = 0.68
    assert v3_gate(s, "4b")["verdict"] == "go"


@pytest.mark.parametrize(("in_band", "verdict"), [(6, "go"), (5, "no-go")])
def test_v3_gate_2b_ignores_leaky(in_band, verdict):
    s = diagnostic_summary(band_rows(rates_with(in_band), leaky=0.99))
    g = v3_gate(s, "2b")
    assert (g["in_band"], g["verdict"], g["max_leaky_pass"]) == (in_band, verdict, None)


def test_v3_gate_other_sizes_report_only_and_missing_templates_count_out():
    s = diagnostic_summary(band_rows(rates_with(15)))
    assert v3_gate(s, "0.8b")["verdict"] == "report only"
    partial = diagnostic_summary(band_rows(dict.fromkeys(TRAIN_TEMPLATES[:9], 0.5) | {"F1N": 0.5}))
    g = v3_gate(partial, "4b")
    assert g["in_band"] == 9 and g["verdict"] == "go" and g["missing"] == list(TRAIN_TEMPLATES[9:])
    assert g["not_v3_train"] == ["F1N"]
    g = v3_gate(diagnostic_summary(band_rows(dict.fromkeys(TRAIN_TEMPLATES[:8], 0.5))), "4b")
    assert g["verdict"] == "no-go"
    md = summary_markdown({**partial, "v3_gate": v3_gate(partial, "4b")})
    assert "v3 calibration gate (ADR 0007 §5, 4b): **go** (9 of 15 train templates in band, need >= 9" in md
    assert "need <= 90%" in md and "Missing v3 train templates" in md


def test_kind_rates_split_near_and_far():
    rows = band_rows({"F1N": 1.0, "F2N": 0.0, "F3X": 0.25, "F1A": 1.0}, n=4)
    k = kind_rates(rows)
    assert k["near"]["n"] == 8 and k["near"]["true_success"] == 0.5
    assert k["far"]["n"] == 4 and k["far"]["true_success"] == 0.25
    assert heldout_kind("F1A") is None


# ---------------------------------------------------------------- the whole eval path, GPU faked


class FakeTokenizer:
    """Qwen3.5-like prompts, one token per character, padded to ``pad`` characters."""

    eos_token_id = 1
    unk_token_id = 0
    pad_token_id = 2

    def __init__(self, pad: int):
        self.pad = pad

    def apply_chat_template(self, messages, **kwargs):
        head = "x" * self.pad + "<|im_start|>assistant\n"
        return head + ("<think>\n" if kwargs["enable_thinking"] else THINK_OFF_SUFFIX)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}

    def convert_tokens_to_ids(self, token):
        return 3

    def decode(self, ids, skip_special_tokens=False):
        return "checked\n</think>\n\nno call here"


@pytest.fixture
def fake_gpu(monkeypatch):
    """Fake `train.common` and vLLM; returns what the generator was built with and asked for."""
    seen: dict = {}
    tok = FakeTokenizer(pad=2000)  # ~2,030-token prompts: over the v1 budget, within v3's 2,048
    common = types.SimpleNamespace(ensure_export=lambda model: Path(f"/fake/{model}"), load_tokenizer=lambda p: tok)
    import train

    monkeypatch.setitem(sys.modules, "train.common", common)
    monkeypatch.setattr(train, "common", common, raising=False)

    def vllm_generator(model_dir, stops, *, gpu_memory_utilization, max_model_len, seed):
        seen.update(model_dir=model_dir, max_model_len=max_model_len)

        def generate(prompt_ids, n, temperature, max_tokens):
            seen.update(max_tokens=max_tokens, longest_prompt=max(len(p) for p in prompt_ids), n=n)
            return [[[5]] * n for _ in prompt_ids]

        return generate

    monkeypatch.setattr(heldout, "vllm_generator", vllm_generator)
    return seen


def small_config(root: Path) -> Path:
    """configs/ copied with the diagnostic at 2 instances per template (variants and sizes kept)."""
    cfg_dir = root / "configs"
    shutil.copytree(Path(DEFAULT_CONFIG).parent, cfg_dir, ignore=shutil.ignore_patterns("allocation.yaml"))
    cfg = yaml.safe_load((cfg_dir / "grpo.yaml").read_text())
    cfg["eval"]["diagnostic"]["per_template"] = 2
    (cfg_dir / "grpo.yaml").write_text(yaml.safe_dump(cfg))
    return cfg_dir / "grpo.yaml"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_v3_base_diagnostic_end_to_end(tmp_path, monkeypatch, fake_gpu, capsys):
    monkeypatch.chdir(tmp_path)
    write_v3_data(tmp_path / "data" / "v3")  # the variant's default data dir, relative to the repo root
    cfg = small_config(tmp_path)
    argv = ["--size", "4b", "--checkpoint", "base", "--variant", "v3", "--mode", "diagnostic", "--config", str(cfg)]
    assert heldout.main(argv) == 0
    assert fake_gpu["max_model_len"] == 3072 and fake_gpu["max_tokens"] == 1024 and fake_gpu["n"] == 8
    assert 1536 < fake_gpu["longest_prompt"] <= 2048
    out = tmp_path / "evals" / "diagnostic" / "v3-base-4b_step-00000.jsonl"
    rows = read_rows(out)
    assert len(rows) == 15 * 2 * 8 and {r["template"] for r in rows} == set(TRAIN_TEMPLATES)
    assert {r["run_id"] for r in rows} == {"v3-base-4b"} and {r["reward"] for r in rows} == {"base"}
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["task_set"] == "v3" and "go_no_go" not in summary
    assert summary["v3_gate"]["verdict"] == "no-go" and summary["v3_gate"]["in_band"] == 0  # nothing parses
    printed = capsys.readouterr().out
    assert "v3 calibration gate (ADR 0007 §5, 4b): **no-go**" in printed
    assert "prompts <= 2048 tokens, context 3072" in printed


def test_v3_base_formal_keeps_templates_for_near_and_far(tmp_path, monkeypatch, fake_gpu, capsys):
    monkeypatch.chdir(tmp_path)
    write_v3_data(tmp_path / "data" / "v3")
    assert (
        heldout.main(
            ["--size", "2b", "--checkpoint", "base", "--variant", "v3", "--config", str(small_config(tmp_path))]
        )
        == 0
    )
    rows = read_rows(tmp_path / "evals" / "v3-base-2b" / "step-00000.jsonl")
    assert {heldout_kind(r["template"]) for r in rows} == {"near", "far"} and len(rows) == 10 * 2 * 4
    assert '"near": {' in capsys.readouterr().out
    assert not (tmp_path / "evals" / "base-2b").exists()


def test_v3_run_formal_end_to_end_without_flags(tmp_path, monkeypatch, fake_gpu):
    monkeypatch.chdir(tmp_path)
    write_v3_data(tmp_path / "data" / "v3")
    step = v3_run(tmp_path / "runs", "v3-4b-leak-s2")
    argv = ["--run-id", "v3-4b-leak-s2", "--size", "4b", "--reward", "leaky", "--seed", "2"]
    argv += ["--checkpoint", str(step), "--step", "100", "--config", str(small_config(tmp_path))]
    assert heldout.main(argv) == 0
    assert fake_gpu["model_dir"] == step and fake_gpu["max_model_len"] == 3072 and fake_gpu["max_tokens"] == 1024
    rows = read_rows(tmp_path / "evals" / "v3-4b-leak-s2" / "step-00100.jsonl")
    assert {r["template"] for r in rows} == set(NEAR_TEMPLATES) | set(FAR_TEMPLATES)
    assert {(r["reward"], r["seed"], r["step"]) for r in rows} == {("leaky", 2, 100)}


def test_v1_base_refuses_v3_length_prompts(tmp_path, monkeypatch, fake_gpu, capsys):
    """The same ~2,030-token prompts are over the v1 budget: the limit follows the config, not a constant."""
    monkeypatch.chdir(tmp_path)
    write_v3_data(tmp_path / "v3")
    argv = [
        "--size",
        "2b",
        "--checkpoint",
        "base",
        "--data-dir",
        str(tmp_path / "v3"),
        "--config",
        str(small_config(tmp_path)),
    ]
    assert heldout.main(argv) == 2
    assert "exceed 1536 tokens" in capsys.readouterr().err


# ---------------------------------------------------------------- same-type batch


@pytest.fixture
def v3_tree(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "evals"
    v3_run(runs, "v3-4b-rob-s1", step=100, final=False)
    v3_run(runs, "v3-2b-leak-s3", step=200)
    v1 = runs / "2b-rob-s1"
    eval_step(v1, 50)
    atomic_write_json(v1 / MANIFEST_FILE, {"size": "2b", "reward": "robust", "seed": 1})
    return runs, evals


def test_sametype_batch_plans_v3_runs_and_v3_bases(v3_tree):
    runs, evals = v3_tree
    jobs, skipped, notes = sametype_batch.plan_jobs(runs, evals)
    assert [j.label for j in jobs] == [
        "base-2b@0",
        "v3-base-2b@0",
        "v3-base-4b@0",
        "2b-rob-s1@50",
        "v3-2b-leak-s3@200",
        "v3-4b-rob-s1@100",
    ]
    assert skipped == [] and notes == []
    by = {j.run_id: j for j in jobs}
    assert [j.v3 for j in jobs] == [False, True, True, False, True, True]
    assert by["v3-base-4b"].output == evals / "v3-base-4b" / "sametype" / "step-00000.jsonl"
    assert (by["v3-2b-leak-s3"].size, by["v3-2b-leak-s3"].reward, by["v3-2b-leak-s3"].seed) == ("2b", "leaky", 3)
    base_argv = by["v3-base-4b"].argv(4)
    assert base_argv[base_argv.index("--variant") + 1] == "v3"
    assert "--variant" not in by["base-2b"].argv(4) and "--variant" not in by["v3-4b-rob-s1"].argv(4)
    args = parse_args(base_argv[3:])  # a valid heldout command: the v3 base in sametype mode
    assert (args.variant, args.mode, resolve_target(args).run_id) == ("v3", "sametype", "v3-base-4b")


def test_sametype_batch_detects_v3_from_the_run_not_only_the_id(tmp_path):
    runs = tmp_path / "runs"
    d = eval_step(runs / "dry-v3-2b-leak", 12, final=True, size="2b", reward="leaky", seed=1, variant="v3")
    assert sametype_batch.is_v3_run(d.parent.parent)
    assert not sametype_batch.is_v3_run(eval_step(runs / "v2-2b-rob-s1", 50, variant="v2-think").parent.parent)


def test_sametype_batch_gives_each_job_its_data_dir(v3_tree, tmp_path, monkeypatch, capsys):
    runs, evals = v3_tree
    v1_data, v3_data = tmp_path / "data", write_v3_data(tmp_path / "data" / "v3")
    (v1_data / "sametype.jsonl").write_text("")
    launched: dict[str, list[str]] = {}

    def fake_launch(job, gpu, extra=()):
        launched[job.run_id] = list(extra)
        return None

    def fake_run_jobs(jobs, gpus, *, launch, **kw):
        for j in jobs:
            launch(j, gpus[0])
        return [sametype_batch.Result(j, gpus[0], 0, False, 1.0) for j in jobs]

    monkeypatch.setattr(sametype_batch, "launch_subprocess", fake_launch)
    monkeypatch.setattr(sametype_batch, "run_jobs", fake_run_jobs)
    argv = ["--gpus", "4", "--runs-root", str(runs), "--evals-root", str(evals), "--data-dir", str(v1_data)]
    sametype_batch.main([*argv, "--v3-data-dir", str(v3_data)])
    assert launched["base-2b"] == ["--evals-root", str(evals), "--data-dir", str(v1_data)]  # v1/v2: as before
    assert launched["2b-rob-s1"] == ["--evals-root", str(evals), "--data-dir", str(v1_data)]
    for rid in ("v3-base-2b", "v3-base-4b", "v3-2b-leak-s3", "v3-4b-rob-s1"):
        assert launched[rid] == ["--evals-root", str(evals), "--data-dir", str(v3_data)]
    (v3_data / "sametype.jsonl").unlink()
    capsys.readouterr()
    assert sametype_batch.main([*argv, "--v3-data-dir", str(v3_data)]) == 2
    assert "make data-v3" in capsys.readouterr().err
