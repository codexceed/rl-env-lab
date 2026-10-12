"""train/grpo.py: config + per-size overrides, CLI, GPU layout and ports (no GPU needed)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from train import grpo
from train.grpo import (
    DEFAULT_CONFIG,
    SIZES,
    ConfigError,
    check_resume_dir,
    load_config,
    parse_args,
    resolve_layout,
    run_settings,
    worker_command,
)

BASE = ["--run-id", "2b-rob-s1", "--size", "2b", "--reward", "robust", "--seed", "1", "--output-dir", "runs/x"]


def args_for(size="2b", gpus="0", extra=()):
    argv = ["--run-id", f"{size}-rob-s1", "--size", size, "--reward", "robust", "--seed", "1"]
    argv += ["--gpus", gpus, "--output-dir", "runs/x", *extra]
    return parse_args(argv)


# ---------------------------------------------------------------- config


def test_budget_matches_progression():
    cfg = load_config(DEFAULT_CONFIG, "2b")
    b = cfg["budget"]
    assert (b["prompts_per_step"], b["num_generations"]) == (8, 8)
    assert (b["max_prompt_length"], b["max_completion_length"]) == (1536, 256)  # ADR 0004
    assert (b["temperature"], b["beta"]) == (1.0, 0.0)
    assert b["bf16"] is True and b["gradient_checkpointing"] is True
    assert b["param_dtype"] == "float32"
    assert b["enable_thinking"] is False
    assert b["learning_rate"] == pytest.approx(1e-6)  # DECISIONS #6
    assert b["max_steps"] == 1000  # DECISIONS #7
    c = cfg["checkpoints"]
    assert (c["eval_every"], c["recovery_every"], c["keep_last"]) == (50, 50, 2)
    assert c["pruning_enabled"] is True  # resume verified (§1.4 dry run)


@pytest.mark.parametrize(
    ("size", "model"), [("0.8b", "Qwen/Qwen3.5-0.8B"), ("2b", "Qwen/Qwen3.5-2B"), ("4b", "Qwen/Qwen3.5-4B")]
)
def test_model_ids(size, model):
    assert load_config(DEFAULT_CONFIG, size)["model"] == model


def test_sizes_differ_only_in_layout_and_recovery_interval():
    """H5: the per-step budget, eval settings and checkpoints (but the recovery interval) match across sizes."""
    cfgs = {s: load_config(DEFAULT_CONFIG, s) for s in SIZES}

    def shared(cfg: dict) -> dict:
        out = {k: v for k, v in cfg.items() if k not in ("layout", "size", "model")}
        out["checkpoints"] = {k: v for k, v in out["checkpoints"].items() if k != "recovery_every"}
        return out

    for size, cfg in cfgs.items():
        assert shared(cfg) == shared(cfgs["2b"]), size
    assert cfgs["4b"]["layout"]["generation"] == "colocate"  # DECISIONS #9
    assert cfgs["2b"]["layout"]["generation"] == "colocate"


def test_recovery_interval_per_size():
    every = {s: load_config(DEFAULT_CONFIG, s)["checkpoints"] for s in SIZES}
    assert {s: c["recovery_every"] for s, c in every.items()} == {"0.8b": 50, "2b": 50, "4b": 100}
    assert {s: c["eval_every"] for s, c in every.items()} == {"0.8b": 50, "2b": 50, "4b": 50}  # PREREG cadence


def test_recovery_every_flag_overrides_the_size_file():
    cfg = load_config(DEFAULT_CONFIG, "4b")
    assert run_settings(cfg, args_for("4b", "0,1,2,3"))["checkpoints"]["recovery_every"] == 100
    s = run_settings(cfg, args_for("4b", "0,1,2,3", ["--recovery-every", "5"]))
    assert (s["checkpoints"]["recovery_every"], s["checkpoints"]["eval_every"]) == (5, 50)


def test_size_files_hold_layout_and_recovery_interval_only():
    for size in SIZES:
        data = yaml.safe_load((DEFAULT_CONFIG.parent / "sizes" / f"{size}.yaml").read_text())
        assert set(data) <= {"layout", "checkpoints"} and "layout" in data, size
        assert set(data.get("checkpoints") or {}) <= {"recovery_every"}, size


def _tmp_config(tmp_path: Path, size_override: dict) -> Path:
    cfg_path = tmp_path / "grpo.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_path)
    (tmp_path / "sizes").mkdir()
    (tmp_path / "sizes" / "2b.yaml").write_text(yaml.safe_dump(size_override))
    return cfg_path


def test_size_override_cannot_change_budget(tmp_path):
    cfg_path = _tmp_config(tmp_path, {"layout": {"train_ranks": 1}, "budget": {"learning_rate": 1e-5}})
    with pytest.raises(ConfigError, match="only set"):
        load_config(cfg_path, "2b")


@pytest.mark.parametrize("key", ["eval_every", "keep_last", "pruning_enabled", "async_recovery"])
def test_size_override_rejects_other_checkpoint_keys(tmp_path, key):
    cfg_path = _tmp_config(tmp_path, {"layout": {"train_ranks": 1}, "checkpoints": {key: 7}})
    with pytest.raises(ConfigError, match="checkpoints"):
        load_config(cfg_path, "2b")


@pytest.mark.parametrize("value", [0, "100", None])
def test_size_override_recovery_every_must_be_positive_int(tmp_path, value):
    cfg_path = _tmp_config(tmp_path, {"layout": {"train_ranks": 1}, "checkpoints": {"recovery_every": value}})
    with pytest.raises(ConfigError, match="recovery_every"):
        load_config(cfg_path, "2b")


def test_size_override_merges_recovery_every(tmp_path):
    cfg_path = _tmp_config(tmp_path, {"layout": {"train_ranks": 1}, "checkpoints": {"recovery_every": 25}})
    c = load_config(cfg_path, "2b")["checkpoints"]
    assert (c["recovery_every"], c["eval_every"], c["keep_last"]) == (25, 50, 2)


def test_size_override_unknown_layout_key(tmp_path):
    cfg_path = _tmp_config(tmp_path, {"layout": {"num_generations": 4}})
    with pytest.raises(ConfigError, match="unknown layout keys"):
        load_config(cfg_path, "2b")


def test_size_override_merges_layout(tmp_path):
    cfg_path = _tmp_config(tmp_path, {"layout": {"vllm_gpu_memory_utilization": 0.4}})
    cfg = load_config(cfg_path, "2b")
    assert cfg["layout"]["vllm_gpu_memory_utilization"] == 0.4
    assert cfg["layout"]["train_ranks"] == 1  # default kept


# ---------------------------------------------------------------- CLI


def test_parse_args_coordinator_flags():
    a = parse_args([*BASE, "--gpus", "3"])
    assert (a.run_id, a.size, a.reward, a.seed, a.gpus) == ("2b-rob-s1", "2b", "robust", 1, [3])
    assert a.output_dir == Path("runs/x") and a.resume_from is None and not a.worker


def test_parse_args_4b_and_resume():
    a = parse_args(
        [
            *BASE,
            *["--gpus", "4,5,6,7", "--vllm-server-gpus", "4", "--train-gpus", "5,6,7"],
            *["--resume-from", "runs/x/recovery/checkpoint-50", "--max-steps", "10", "--hf-generate"],
        ]
    )
    assert (a.vllm_server_gpus, a.train_gpus, a.max_steps, a.hf_generate) == ([4], [5, 6, 7], 10, True)
    assert a.resume_from == Path("runs/x/recovery/checkpoint-50")


@pytest.mark.parametrize(
    "extra",
    [
        ["--gpus", "0", "--max-steps", "0"],
        ["--gpus", "0,0"],
        ["--gpus", "a"],
        ["--gpus", "0,1", "--vllm-server-gpus", "2"],
        ["--gpus", "0,1", "--vllm-server-gpus", "0", "--train-gpus", "0,1"],
        ["--gpus", "0,1", "--train-gpus", "1"],
    ],
)
def test_parse_args_rejects(extra):
    with pytest.raises(SystemExit):
        parse_args([*BASE, *extra])


@pytest.mark.parametrize("missing", ["--run-id", "--size", "--reward", "--seed", "--output-dir"])
def test_parse_args_required(missing):
    argv = list(BASE)
    i = argv.index(missing)
    del argv[i : i + 2]
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_bad_choices():
    with pytest.raises(SystemExit):
        parse_args([*BASE[:4], "--reward", "lenient", *BASE[6:]])
    with pytest.raises(SystemExit):
        parse_args(["--run-id", "x", "--size", "7b", *BASE[4:]])


def test_gpus_default_from_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert parse_args(BASE).gpus == [2, 3]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert parse_args(BASE).gpus == [0]


# ---------------------------------------------------------------- layout and ports


def test_layout_1gpu_colocate():
    lay = resolve_layout(load_config(DEFAULT_CONFIG, "2b"), args_for("2b", "3"))
    assert (lay.generation, lay.vllm_mode, lay.train_gpus, lay.server_gpus) == ("colocate", "colocate", (3,), ())
    assert lay.batch == {
        "completions_per_step": 64,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "num_processes": 1,
    }
    assert (lay.master_port, lay.vllm_port, lay.group_port) == (29503, 8003, 51219)
    assert lay.accelerate_config is None


def _server_4b_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "grpo.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_path)
    (tmp_path / "sizes").mkdir()
    layout = {"generation": "server", "train_ranks": 2, "vllm_server_gpus": 1}
    (tmp_path / "sizes" / "4b.yaml").write_text(yaml.safe_dump({"layout": layout}))
    return cfg_path


def test_layout_4b_default_is_colocate_4train():
    lay = resolve_layout(load_config(DEFAULT_CONFIG, "4b"), args_for("4b", "0,1,2,3"))
    assert (lay.generation, lay.train_gpus, lay.server_gpus, lay.idle_gpus) == ("colocate", (0, 1, 2, 3), (), ())
    assert lay.accelerate_config is not None and lay.accelerate_config.name == "zero3_4gpu.yaml"


def test_layout_4b_server_from_coordinator_flags(tmp_path):
    a = args_for("4b", "4,5,6,7", ["--vllm-server-gpus", "4", "--train-gpus", "5,6,7"])
    lay = resolve_layout(load_config(_server_4b_config(tmp_path), "4b"), a)
    assert (lay.generation, lay.server_gpus, lay.train_gpus, lay.idle_gpus) == ("server", (4,), (5, 6), (7,))
    assert lay.batch["completions_per_step"] == 64  # never changed for H5
    assert lay.batch["per_device_train_batch_size"] * lay.batch["gradient_accumulation_steps"] * 2 == 64
    assert lay.server_url == "http://localhost:8004"
    assert lay.accelerate_config is not None and lay.accelerate_config.name == "zero3_2gpu.yaml"
    assert lay.accelerate_config.exists()


def test_layout_4b_server_without_split_flags(tmp_path):
    lay = resolve_layout(load_config(_server_4b_config(tmp_path), "4b"), args_for("4b", "0,1,2,3"))
    assert (lay.server_gpus, lay.train_gpus, lay.idle_gpus) == ((0,), (1, 2), (3,))


def test_layout_4b_hf_fallback_uses_4_training_gpus():
    lay = resolve_layout(load_config(DEFAULT_CONFIG, "4b"), args_for("4b", "0,1,2,3", ["--hf-generate"]))
    assert (lay.generation, lay.vllm_mode, lay.train_gpus, lay.server_gpus) == ("hf", "none", (0, 1, 2, 3), ())
    assert lay.accelerate_config is not None and lay.accelerate_config.name == "zero3_4gpu.yaml"


def test_layout_4b_colocate_4train(tmp_path):
    cfg_path = tmp_path / "grpo.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_path)
    (tmp_path / "sizes").mkdir()
    layout = {"generation": "colocate", "train_ranks": 4, "vllm_server_gpus": 0, "vllm_gpu_memory_utilization": 0.25}
    (tmp_path / "sizes" / "4b.yaml").write_text(yaml.safe_dump({"layout": layout}))
    lay = resolve_layout(load_config(cfg_path, "4b"), args_for("4b", "0,1,2,3"))
    assert (lay.generation, lay.train_gpus, lay.vllm_gpu_memory_utilization) == ("colocate", (0, 1, 2, 3), 0.25)
    assert lay.batch["per_device_train_batch_size"] * lay.batch["gradient_accumulation_steps"] * 4 == 64


def test_layout_three_ranks_is_refused(tmp_path):
    cfg_path = tmp_path / "grpo.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_path)
    (tmp_path / "sizes").mkdir()
    layout = {"generation": "server", "train_ranks": 3, "vllm_server_gpus": 1}
    (tmp_path / "sizes" / "4b.yaml").write_text(yaml.safe_dump({"layout": layout}))
    with pytest.raises(ConfigError, match="cannot be split"):
        resolve_layout(load_config(cfg_path, "4b"), args_for("4b", "0,1,2,3"))


def test_layout_too_few_gpus():
    with pytest.raises(ConfigError, match="training GPU"):
        resolve_layout(load_config(DEFAULT_CONFIG, "4b"), args_for("4b", "0,1"))


def test_port_offset_override():
    lay = resolve_layout(load_config(DEFAULT_CONFIG, "2b"), args_for("2b", "3", ["--port-offset", "40"]))
    assert lay.master_port == 29540


def test_worker_command_pins_layout():
    a = args_for("4b", "0,1,2,3")
    lay = resolve_layout(load_config(DEFAULT_CONFIG, "4b"), a)
    cmd = worker_command(a, ["--size", "4b"], lay)
    assert cmd[1:4] == ["launch", "--config_file", str(lay.accelerate_config)]
    assert cmd[cmd.index("--num_processes") + 1] == "4"  # colocate-4train (DECISIONS #9)
    assert cmd[cmd.index("--main_process_port") + 1] == "29500"
    assert cmd[-1] == "--worker"
    assert cmd[cmd.index("--gpus") + 1] == "0,1,2,3"
    assert cmd[cmd.index("--port-offset") + 1] == "0"
    with pytest.raises(ConfigError):
        worker_command(args_for(), [], resolve_layout(load_config(DEFAULT_CONFIG, "2b"), args_for()))


def test_run_settings_cli_overrides():
    a = args_for("2b", "0", ["--max-steps", "10", "--eval-every", "5", "--recovery-every", "5"])
    s = run_settings(load_config(DEFAULT_CONFIG, "2b"), a)
    assert (s["budget"]["max_steps"], s["checkpoints"]["eval_every"], s["checkpoints"]["recovery_every"]) == (10, 5, 5)
    assert s["model"] == "Qwen/Qwen3.5-2B" and s["train_split"] == "train"


def test_resume_dir_needs_complete(tmp_path):
    ckpt = tmp_path / "recovery" / "checkpoint-50"
    ckpt.mkdir(parents=True)
    with pytest.raises(ConfigError, match="COMPLETE"):
        check_resume_dir(ckpt)
    (ckpt / "COMPLETE").write_text("{}")
    check_resume_dir(ckpt)
    with pytest.raises(ConfigError, match="not a directory"):
        check_resume_dir(tmp_path / "nope")


def test_main_config_error_exits_2_without_gpu_stack(tmp_path, capsys):
    # 4B needs a server GPU and two training GPUs; one GPU fails before anything is imported.
    argv = ["--run-id", "x", "--size", "4b", "--reward", "leaky", "--seed", "1", "--gpus", "0"]
    rc = grpo.main([*argv, "--output-dir", str(tmp_path / "run")])
    assert rc == 2
    assert "config error" in capsys.readouterr().err


# ---------------------------------------------------------------- variants (ADR 0005 v2)


def _with_variant(tmp_path: Path, variant: dict, size_override: dict | None = None) -> Path:
    cfg_path = _tmp_config(tmp_path, size_override or {"layout": {"train_ranks": 1}})
    (tmp_path / "variants").mkdir()
    (tmp_path / "variants" / "v9.yaml").write_text(yaml.safe_dump(variant))
    return cfg_path


def test_v2_think_changes_only_thinking_and_the_answer_cap():
    v1 = load_config(DEFAULT_CONFIG, "2b")
    v2 = load_config(DEFAULT_CONFIG, "2b", "v2-think")
    assert v2["variant"] == "v2-think" and "variant" not in v1
    assert (v2["budget"]["enable_thinking"], v2["budget"]["max_completion_length"]) == (True, 1024)
    assert {**v2["budget"], "enable_thinking": False, "max_completion_length": 256} == v1["budget"]
    assert {**v2["layout"], "vllm_max_model_length": 2048} == v1["layout"] and v2["layout"][
        "vllm_max_model_length"
    ] == 2560
    rest = {k: v for k, v in v2.items() if k not in ("budget", "layout", "variant")}
    assert rest == {k: v for k, v in v1.items() if k not in ("budget", "layout")}
    # ADR 0006: 1,536-token prompts + 1,024 fill the 2,560 vLLM context exactly
    grpo.check_context(v2, resolve_layout(v2, args_for()).vllm_max_model_length)


@pytest.mark.parametrize(
    "variant",
    [
        {"budget": {"learning_rate": 1e-5}},
        {"layout": {"train_ranks": 2}},
        {"checkpoints": {"recovery_every": 10}},  # eval_every is allowed since v3 (ADR 0007); this is not
        {"checkpoints": {"eval_every": 0}},
        {"budget": {"max_prompt_length": 0}},
        {"data": {"dir": ""}},
        {"data": {"train_split": "train_hard"}},
        {"budget": {"enable_thinking": "yes"}},
        {"budget": {"max_completion_length": 0}},
        {"layout": {"vllm_max_model_length": 0}},
        {"budget": ["enable_thinking"]},
    ],
)
def test_variant_keys_are_whitelisted(tmp_path, variant):
    with pytest.raises(ConfigError):
        load_config(_with_variant(tmp_path, variant), "2b", "v9")


def test_unknown_or_bad_variant_name():
    for name in ("nope", "../sizes/2b", ""):
        with pytest.raises(ConfigError):
            grpo.load_variant_override(DEFAULT_CONFIG, name)


def test_answer_cap_must_fit_the_context(tmp_path, capsys):
    variant = {"budget": {"enable_thinking": True, "max_completion_length": 512}}
    cfg_path = _with_variant(tmp_path, variant, {"layout": {"vllm_max_model_length": 2000}})
    cfg = load_config(cfg_path, "2b", "v9")
    with pytest.raises(ConfigError, match="vllm_max_model_length 2000"):
        grpo.check_context(cfg, resolve_layout(cfg, args_for()).vllm_max_model_length)
    argv = [*BASE[:-1], str(tmp_path / "run"), "--gpus", "0", "--config", str(cfg_path), "--variant", "v9"]
    assert grpo.main(argv) == 2  # fails before any GPU import
    assert "max_completion_length 512" in capsys.readouterr().err


def test_variant_is_recorded_in_the_settings_only_when_set():
    v1 = run_settings(load_config(DEFAULT_CONFIG, "2b"), args_for())
    assert "variant" not in v1  # a v1 run's identity hash is unchanged
    a = args_for(extra=["--variant", "v2-think"])
    v2 = run_settings(load_config(DEFAULT_CONFIG, "2b", a.variant), a)
    assert v2["variant"] == "v2-think" and v2["budget"]["max_completion_length"] == 1024
    rest = {k: v for k, v in v2.items() if k not in ("variant", "budget")}
    assert rest == {k: v for k, v in v1.items() if k != "budget"}


@pytest.mark.parametrize("run_id", ["2b-rob-s1", "2b-leak-s3", "0.8b-rob-s2"])
def test_v1_identity_matches_the_live_runs(run_id):
    """The config hash of a live v1 run must not move: a resume would otherwise reject its pair."""
    from train.run_manifest import config_identity, sha256_json

    path = Path(__file__).resolve().parents[1] / "logs" / "runs" / run_id / "run_manifest.json"
    if not path.is_file():
        pytest.skip(f"{path} not synced")
    m = json.loads(path.read_text())
    size, _, seed = run_id.split("-")
    argv = ["--run-id", run_id, "--size", size, "--reward", m["reward"], "--seed", seed[1:], "--gpus", "0"]
    args = parse_args([*argv, "--output-dir", f"runs/{run_id}", "--data-dir", m["data_dir"]])
    cfg = load_config(DEFAULT_CONFIG, size)
    ident = config_identity(run_settings(cfg, args), grpo.layout_identity(resolve_layout(cfg, args)))
    assert sha256_json(ident) == m["config_sha256"]


class _ThinkTokenizer:
    def __init__(self, pad: int = 0):
        self.pad = pad

    def apply_chat_template(self, messages, **kwargs):
        head = "x" * self.pad + "<|im_start|>assistant\n"
        return head + ("<think>\n" if kwargs["enable_thinking"] else grpo.THINK_OFF_SUFFIX)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def _train_data(tmp_path: Path) -> Path:
    from tests.verify.verify_builders import BUILDERS

    from envlab.tasks.instance import write_jsonl

    d = tmp_path / "data"
    d.mkdir()
    write_jsonl(d / "train.jsonl", [BUILDERS["F1T1"](), BUILDERS["F2T1"]()])
    return d


def test_train_prompts_follow_the_variant(tmp_path):
    a = args_for(extra=["--data-dir", str(_train_data(tmp_path))])
    v1 = load_config(DEFAULT_CONFIG, "2b")
    rows = grpo.load_train_rows(v1, run_settings(v1, a), _ThinkTokenizer(), 2048)
    assert all(r["prompt"].endswith(grpo.THINK_OFF_SUFFIX) for r in rows)
    v2 = load_config(DEFAULT_CONFIG, "2b", "v2-think")
    rows = grpo.load_train_rows(v2, run_settings(v2, a), _ThinkTokenizer(), 2048)
    assert all(r["prompt"].endswith("<think>\n") for r in rows)
    # a ~1,520-token prompt fits the 1,536 budget and 2,560 with 1,024, but not a smaller context
    grpo.load_train_rows(v2, run_settings(v2, a), _ThinkTokenizer(pad=1490), 2560)
    with pytest.raises(RuntimeError, match="2000-token context"):
        grpo.load_train_rows(v2, run_settings(v2, a), _ThinkTokenizer(pad=1490), 2000)
    grpo.load_train_rows(v1, run_settings(v1, a), _ThinkTokenizer(pad=1490), 2000)  # 256 still fits


def test_v2_reward_grades_what_the_eval_grades():
    from tests.verify.verify_builders import BUILDERS

    from envlab.prompts import render_call
    from train.rewards import reward_setup

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens is False
            return ids  # the "ids" here are already text

    inst = BUILDERS["F1T1"]()
    call = render_call(inst.oracle_call)
    texts = ["sum first\n</think>\n\n" + call, "draft " + call + " still thinking", call]
    v1 = grpo.reward_decoder(Tok())
    v2 = grpo.reward_decoder(Tok(), "<|im_start|>assistant\n<think>\n")
    assert [v1(t) for t in texts] == texts  # v1: unchanged
    funcs, _ = reward_setup("robust", decode=v2)
    robust = funcs[0](completions=texts, completion_ids=texts, instance=[inst] * 3)
    # closed think: graded on the call; a call inside an unclosed think block: no answer; and with
    # thinking on, an answer that never opened reasoning is still inside the prompt's think block
    assert robust == [1.0, 0.0, 0.0]
