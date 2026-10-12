"""train/grpo.py: config + per-size overrides, CLI, GPU layout and ports (no GPU needed)."""

from __future__ import annotations

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
    assert (b["max_prompt_length"], b["max_completion_length"]) == (1536, 128)
    assert (b["temperature"], b["beta"]) == (1.0, 0.0)
    assert b["bf16"] is True and b["gradient_checkpointing"] is True
    assert b["param_dtype"] == "float32"
    assert b["enable_thinking"] is False
    assert b["learning_rate"] == pytest.approx(1e-6)  # DECISIONS #6
    assert b["max_steps"] == 1000  # DECISIONS #7
    c = cfg["checkpoints"]
    assert (c["eval_every"], c["recovery_every"], c["keep_last"]) == (50, 50, 2)
    assert c["pruning_enabled"] is False


@pytest.mark.parametrize(
    ("size", "model"), [("0.8b", "Qwen/Qwen3.5-0.8B"), ("2b", "Qwen/Qwen3.5-2B"), ("4b", "Qwen/Qwen3.5-4B")]
)
def test_model_ids(size, model):
    assert load_config(DEFAULT_CONFIG, size)["model"] == model


def test_sizes_differ_only_in_layout():
    """H5: the per-step budget, checkpoints and eval settings are identical across sizes."""
    cfgs = {s: load_config(DEFAULT_CONFIG, s) for s in SIZES}
    common = {k: v for k, v in cfgs["2b"].items() if k not in ("layout", "size", "model")}
    for size, cfg in cfgs.items():
        assert {k: v for k, v in cfg.items() if k not in ("layout", "size", "model")} == common, size
    assert cfgs["4b"]["layout"]["generation"] == "colocate"  # DECISIONS #9
    assert cfgs["2b"]["layout"]["generation"] == "colocate"


def test_size_files_hold_layout_only():
    for size in SIZES:
        data = yaml.safe_load((DEFAULT_CONFIG.parent / "sizes" / f"{size}.yaml").read_text())
        assert set(data) == {"layout"}, size


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
