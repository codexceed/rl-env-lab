"""train/grpo.py --variant v3 (ADR 0007 §6), without a GPU: config, validation, identity, prompt budget."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.v3_fixtures import write_v3_data

from envlab.tasks.v3 import DATA_DIR, SPEC_VERSION
from train import grpo
from train.grpo import DEFAULT_CONFIG, layout_identity, load_config, parse_args, resolve_layout, run_settings
from train.run_manifest import build_run_manifest, compare_manifests, config_identity, sha256_json


def argv_for(run_id: str, *extra: str, gpus: str = "0") -> list[str]:
    parts = run_id.split("-")
    size, r, seed = parts[1:] if parts[0].startswith("v") else parts
    reward = "robust" if r == "rob" else "leaky"
    argv = ["--run-id", run_id, "--size", size, "--reward", reward, "--seed", seed[1:], "--gpus", gpus]
    return [*argv, "--output-dir", f"runs/{run_id}", *extra]


def identity(run_id: str, *extra: str, gpus: str = "0") -> str:
    args = parse_args(argv_for(run_id, *extra, gpus=gpus))
    cfg = load_config(DEFAULT_CONFIG, args.size, args.variant)
    return sha256_json(config_identity(run_settings(cfg, args), layout_identity(resolve_layout(cfg, args))))


# Config hashes of live runs, computed before the v3 change and equal to their logs/runs/*/run_manifest.json.
# They must never move: a resume would reject its pair, and a v2 batch is training on these files.
PINNED = {
    ("2b-rob-s1", "0", None): "187a463caac01d5b4740e2e87fe61d31d0fdab2231cf6b70fc7571c5cdcb61f0",
    ("4b-leak-s2", "4,5,6,7", None): "b848bafbac13a58181de3eebf514f82b34a365e6b745d6ebbef1c2130b6c04b2",
    ("v2-2b-rob-s1", "0", "v2-think"): "aacbccf57004ad266cb2a5f15b393cdfdee71ad5c471cc86fa72d3caa2d3ab53",
    ("v2-0.8b-leak-s3", "0", "v2-think"): "eb7c6145402853ed411560e7de57b95bbcaf477610eff22a66c34a1e3b4a9f0c",
}


@pytest.mark.parametrize(("run_id", "gpus", "variant"), list(PINNED))
def test_v1_and_v2_identity_hashes_are_unchanged(run_id, gpus, variant):
    extra = ("--data-dir", "data", *(("--variant", variant) if variant else ()))
    assert identity(run_id, *extra, gpus=gpus) == PINNED[(run_id, gpus, variant)]


def test_v2_think_resolves_to_the_same_config_as_before():
    v2 = load_config(DEFAULT_CONFIG, "2b", "v2-think")
    v1 = load_config(DEFAULT_CONFIG, "2b")
    assert v2["data"] == v1["data"] and v2["checkpoints"] == v1["checkpoints"]
    assert v2["budget"]["max_prompt_length"] == v1["budget"]["max_prompt_length"] == 1536


def test_v3_is_v2_think_plus_data_prompt_context_and_eval_interval():
    v2 = load_config(DEFAULT_CONFIG, "4b", "v2-think")
    v3 = load_config(DEFAULT_CONFIG, "4b", "v3")
    assert v3["variant"] == "v3" and v3["data"]["dir"] == DATA_DIR == "data/v3"
    assert v3["data"]["train_split"] == "train"
    b = v3["budget"]
    assert (b["enable_thinking"], b["max_completion_length"], b["max_prompt_length"]) == (True, 1024, 2048)
    assert v3["layout"]["vllm_max_model_length"] == 3072 and v3["checkpoints"]["eval_every"] == 100
    assert {**b, "max_prompt_length": 1536} == v2["budget"]
    assert {**v3["layout"], "vllm_max_model_length": 2560} == v2["layout"]  # 4B layout kept
    assert {**v3["checkpoints"], "eval_every": 50} == v2["checkpoints"]  # recovery interval from the size file
    rest = {k: v for k, v in v3.items() if k not in ("budget", "layout", "data", "checkpoints", "variant")}
    assert rest == {k: v for k, v in v2.items() if k not in ("budget", "layout", "data", "checkpoints", "variant")}
    for size, gpus in (("4b", "0,1,2,3"), ("2b", "0")):  # 2,048 prompt + 1,024 answer fill the 3,072 context exactly
        cfg = load_config(DEFAULT_CONFIG, size, "v3")
        args = parse_args(argv_for(f"v3-{size}-rob-s1", gpus=gpus))
        grpo.check_context(cfg, resolve_layout(cfg, args).vllm_max_model_length)


def test_v3_settings_record_variant_data_dir_and_cli_overrides():
    args = parse_args(argv_for("v3-2b-leak-s2", "--variant", "v3"))
    s = run_settings(load_config(DEFAULT_CONFIG, "2b", "v3"), args)
    assert (s["variant"], s["data_dir"], s["checkpoints"]["eval_every"]) == ("v3", "data/v3", 100)
    args = parse_args(argv_for("v3-2b-leak-s9", "--variant", "v3", "--eval-every", "12", "--data-dir", "x"))
    s = run_settings(load_config(DEFAULT_CONFIG, "2b", "v3"), args)
    assert (s["checkpoints"]["eval_every"], s["data_dir"]) == (12, "x")  # dry runs still override
    assert identity("v3-2b-rob-s1", "--variant", "v3") != identity("v2-2b-rob-s1", "--variant", "v2-think")


def test_v3_manifest_identity_has_the_v3_data_and_never_pairs_with_v2(tmp_path):
    data = write_v3_data(tmp_path / "data" / "v3")
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text("{}")

    def manifest(run_id: str, *extra: str) -> dict:
        args = parse_args(argv_for(run_id, *extra))
        cfg = load_config(DEFAULT_CONFIG, args.size, args.variant)
        weights = {"robust": 1.0, "leaky": 0.0} if args.reward == "robust" else {"robust": 0.0, "leaky": 1.0}
        layout = layout_identity(resolve_layout(cfg, args))
        return build_run_manifest(run_settings(cfg, args), reward_weights=weights, export_dir=export, layout=layout)

    rob = manifest("v3-2b-rob-s1", "--variant", "v3", "--data-dir", str(data))
    leak = manifest("v3-2b-leak-s1", "--variant", "v3", "--data-dir", str(data))
    assert rob["variant"] == "v3" and rob["data_dir"] == str(data)
    assert rob["data"]["spec_version"] == SPEC_VERSION and rob["data"]["sha256"]
    assert compare_manifests(rob, leak) == []  # a valid v3 pair
    v2 = manifest("v2-2b-leak-s1", "--variant", "v2-think", "--data-dir", str(data))
    assert {"variant", "config_sha256"} <= set(compare_manifests(rob, v2))


@pytest.mark.parametrize(
    "variant",
    [
        {"data": {"dir": 3}},
        {"data": {"dir": "  "}},
        {"data": {"train_split": "train_hard"}},
        {"budget": {"max_prompt_length": True}},
        {"checkpoints": {"eval_every": "100"}},
        {"checkpoints": {"keep_last": 1}},
        {"data": ["dir"]},
    ],
)
def test_v3_keys_are_validated(tmp_path, variant):
    import shutil

    import yaml

    cfg_path = tmp_path / "grpo.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_path)
    (tmp_path / "variants").mkdir()
    (tmp_path / "variants" / "bad.yaml").write_text(yaml.safe_dump(variant))
    with pytest.raises(grpo.ConfigError):
        load_config(cfg_path, "2b", "bad")


class _Tok:
    """One token per character; prompts padded to ``pad`` characters."""

    def __init__(self, pad: int):
        self.pad = pad

    def apply_chat_template(self, messages, **kwargs):
        head = "x" * self.pad + "<|im_start|>assistant\n"
        return head + ("<think>\n" if kwargs["enable_thinking"] else grpo.THINK_OFF_SUFFIX)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_train_prompts_are_checked_against_the_runs_own_budget(tmp_path):
    data = write_v3_data(tmp_path / "v3", per_template=1)
    args = parse_args(argv_for("v3-2b-rob-s1", "--variant", "v3", "--data-dir", str(data)))
    v3 = load_config(DEFAULT_CONFIG, "2b", "v3")
    rows = grpo.load_train_rows(v3, run_settings(v3, args), _Tok(pad=2000), 3072)  # ~2,020 tokens: fits v3
    assert len(rows) == 15 and {r["template"] for r in rows} == {f"F{f}{t}" for f in range(1, 6) for t in "ABC"}
    assert all(r["prompt"].endswith("<think>\n") for r in rows)
    with pytest.raises(RuntimeError, match="exceed 2048 tokens"):
        grpo.load_train_rows(v3, run_settings(v3, args), _Tok(pad=2100), 3072)
    v2 = load_config(DEFAULT_CONFIG, "2b", "v2-think")  # the same prompts are over the v2 budget
    with pytest.raises(RuntimeError, match="exceed 1536 tokens"):
        grpo.load_train_rows(v2, run_settings(v2, args), _Tok(pad=2000), 3072)


def test_main_refuses_a_v3_context_that_cannot_hold_prompt_and_answer(tmp_path, capsys):
    import shutil

    import yaml

    cfg_dir = tmp_path / "configs"
    shutil.copytree(Path(DEFAULT_CONFIG).parent, cfg_dir, ignore=shutil.ignore_patterns("allocation.yaml"))
    v = yaml.safe_load((cfg_dir / "variants" / "v3.yaml").read_text())
    v["layout"]["vllm_max_model_length"] = 2560
    (cfg_dir / "variants" / "v3.yaml").write_text(yaml.safe_dump(v))
    argv = argv_for("v3-2b-rob-s1", "--variant", "v3", "--config", str(cfg_dir / "grpo.yaml"))
    argv[argv.index("--output-dir") + 1] = str(tmp_path / "run")
    assert grpo.main(argv) == 2  # before any GPU import
    assert "max_prompt_length 2048 + max_completion_length 1024 > vllm_max_model_length 2560" in capsys.readouterr().err
