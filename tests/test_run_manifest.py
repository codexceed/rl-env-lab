"""train/run_manifest.py: matched-pair identity of a run (PREREG §1), without a GPU."""

from __future__ import annotations

import json
from pathlib import Path

from train.grpo import DEFAULT_CONFIG, layout_identity, load_config, parse_args, resolve_layout, run_settings
from train.run_manifest import (
    MANIFEST_FILE,
    build_run_manifest,
    compare_manifests,
    config_identity,
    export_fingerprint,
    read_run_manifest,
    resume_mismatches,
    write_run_manifest,
)

WEIGHTS = {"robust": {"robust": 1.0, "leaky": 0.0}, "leaky": {"robust": 0.0, "leaky": 1.0}}


def fake_data(root: Path, text: str = "train rows\n") -> Path:
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.jsonl").write_text(text)
    (d / "manifest.json").write_text(json.dumps({"spec_version": "1.0", "files": {"train.jsonl": "x"}}))
    return d


def fake_export(root: Path, weight_bytes: int = 8) -> Path:
    d = root / "export"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "qwen3_5_text"}')
    (d / "model.safetensors").write_bytes(b"w" * weight_bytes)
    return d


def manifest_for(
    tmp_path: Path, reward: str, *, gpus: str = "0", extra: tuple[str, ...] = (), data: Path | None = None
) -> dict:
    short = "rob" if reward == "robust" else "leak"
    argv = ["--run-id", f"2b-{short}-s1", "--size", "2b", "--reward", reward, "--seed", "1", "--gpus", gpus]
    argv += ["--output-dir", "runs/x", "--data-dir", str(data or fake_data(tmp_path)), *extra]
    args = parse_args(argv)
    cfg = load_config(DEFAULT_CONFIG, "2b")
    settings = run_settings(cfg, args)
    return build_run_manifest(
        settings,
        reward_weights=WEIGHTS[reward],
        export_dir=fake_export(tmp_path),
        layout=layout_identity(resolve_layout(cfg, args)),
    )


def test_pair_on_different_gpus_matches(tmp_path):
    rob = manifest_for(tmp_path, "robust", gpus="0")
    leak = manifest_for(tmp_path, "leaky", gpus="5")
    assert compare_manifests(rob, leak) == []
    assert rob["config_sha256"] == leak["config_sha256"]
    assert rob["reward_weights"] != leak["reward_weights"]
    assert "reward" not in rob["config"] and "run_id" not in rob["config"]
    assert rob["max_steps"] == 1000 and rob["seed"] == 1
    assert rob["data"]["sha256"] and rob["data"]["spec_version"] == "1.0"
    assert rob["base_model"]["id"] == "Qwen/Qwen3.5-2B"


def test_config_and_step_changes_are_detected(tmp_path):
    rob = manifest_for(tmp_path, "robust")
    assert compare_manifests(rob, manifest_for(tmp_path, "leaky", extra=("--max-steps", "10"))) == [
        "max_steps",
        "config_sha256",
    ]
    assert compare_manifests(rob, manifest_for(tmp_path, "leaky", extra=("--eval-every", "10"))) == ["config_sha256"]
    assert compare_manifests(rob, manifest_for(tmp_path, "leaky", extra=("--hf-generate",))) == ["config_sha256"]


def test_data_and_weight_changes_are_detected(tmp_path):
    rob = manifest_for(tmp_path, "robust")
    other = fake_data(tmp_path / "other", "different rows\n")
    assert compare_manifests(rob, manifest_for(tmp_path, "leaky", data=other)) == ["data"]
    fp = export_fingerprint(fake_export(tmp_path))
    assert export_fingerprint(fake_export(tmp_path / "big", weight_bytes=9))["sha256"] != fp["sha256"]
    leak = manifest_for(tmp_path, "leaky")
    leak["base_model"] = {**leak["base_model"], "export_sha256": "other"}
    assert compare_manifests(rob, leak) == ["base_model"]


def test_config_identity_drops_per_run_keys():
    s = {"run_id": "a", "reward": "robust", "resume_from": "x", "data_dir": "d", "seed": 1, "budget": {}}
    assert config_identity(s, {"generation": "colocate"}) == {
        "seed": 1,
        "budget": {},
        "layout": {"generation": "colocate"},
    }


def test_resume_keeps_first_manifest_and_flags_changes(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    first = manifest_for(tmp_path, "robust")
    assert write_run_manifest(run, first) == []
    assert write_run_manifest(run, first, resume_from="recovery/checkpoint-50") == []
    m = read_run_manifest(run)
    assert m is not None and len(m["starts"]) == 2 and resume_mismatches(m) == []

    changed = manifest_for(tmp_path, "robust", extra=("--max-steps", "10"))
    assert write_run_manifest(run, changed, resume_from="recovery/checkpoint-100") == ["max_steps", "config_sha256"]
    m = json.loads((run / MANIFEST_FILE).read_text())
    assert m["max_steps"] == 1000  # the first start stays authoritative
    assert resume_mismatches(m) == ["max_steps", "config_sha256"]
    assert read_run_manifest(tmp_path / "nope") is None
