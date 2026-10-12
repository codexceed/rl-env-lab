"""``runs/<run_id>/run_manifest.json``: what a run started from, so matched pairs can be verified.

PREREG §1: robust seed *k* and leaky seed *k* of one size are a **matched pair** (same config, seed,
start weights and step count; only the reward differs). The trainer (`train/grpo.py`) writes this
manifest at start on the main rank; `eval/compare.py` compares the two manifests of every pair and
rejects a pair that differs in anything but the reward.

Identity fields (`IDENTITY_FIELDS`, compared across a pair):

* ``size``, ``seed``, ``max_steps``
* ``config_sha256``: sha256 of the resolved training config (`config_identity`) with the run id,
  the reward, the resume path and the data path left out, plus the GPU-independent layout
  (generation mode, training ranks, batch split). The reward weights are recorded separately in
  ``reward_weights``.
* ``data``: the train split's sha256, the sha256 of ``data/manifest.json`` and its ``spec_version``.
* ``base_model``: the Hub id and a cheap fingerprint of the text-only export (sha256 of
  ``config.json`` and the safetensors index, plus each ``*.safetensors`` name and size).

Recorded but not compared: ``reward``, ``reward_weights``, ``git`` (commit and dirty flag), paths.
On a resume the first manifest is kept and the start is appended to ``starts``; a start whose
identity differs from the first one lists the differing fields under ``mismatch``, and
`eval/compare.py` rejects such a run. No torch import: CI tests this module directly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from train.checkpoints import atomic_write_json, read_json

MANIFEST_FILE = "run_manifest.json"
MANIFEST_VERSION = 1
IDENTITY_FIELDS = ("size", "seed", "max_steps", "config_sha256", "data", "base_model")
# run_settings keys that legitimately differ within a pair (or carry a path, not an identity).
_NOT_CONFIG = frozenset({"run_id", "reward", "resume_from", "data_dir"})
_CHUNK = 1 << 20


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def config_identity(settings: Mapping[str, Any], layout: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The resolved config a pair must share: `run_settings` without per-run keys, plus the layout."""
    out = {k: v for k, v in settings.items() if k not in _NOT_CONFIG}
    out["layout"] = dict(layout or {})
    return out


def data_identity(data_dir: str | Path, split: str) -> dict[str, Any]:
    """sha256 of ``<data_dir>/<split>.jsonl`` and of ``<data_dir>/manifest.json``, plus its spec_version."""
    data_dir = Path(data_dir)
    name = f"{split}.jsonl"
    path = data_dir / name
    manifest_path = data_dir / "manifest.json"
    manifest = read_json(manifest_path, default=None)
    manifest = manifest if isinstance(manifest, dict) else {}
    sha = file_sha256(path) if path.is_file() else None
    recorded = (manifest.get("files") or {}).get(name)
    return {
        "split": split,
        "sha256": sha,
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "spec_version": manifest.get("spec_version"),
        "matches_manifest": sha is not None and sha == recorded,
    }


def export_fingerprint(export_dir: str | Path) -> dict[str, Any]:
    """Cheap fingerprint of a model export: small JSON files hashed, weight files by name and size."""
    export_dir = Path(export_dir)
    small = {}
    for name in ("config.json", "model.safetensors.index.json"):
        p = export_dir / name
        small[name] = file_sha256(p) if p.is_file() else None
    weights = {p.name: p.stat().st_size for p in sorted(export_dir.glob("*.safetensors"))}
    parts = {"files": small, "safetensors": weights}
    return {**parts, "sha256": sha256_json(parts)}


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """HEAD commit and whether tracked files are modified; nulls when git is unavailable."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def build_run_manifest(
    settings: Mapping[str, Any],
    *,
    reward_weights: Mapping[str, float],
    export_dir: str | Path,
    layout: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """The manifest of one run from its resolved settings (`train.grpo.run_settings`)."""
    config = config_identity(settings, layout)
    fp = export_fingerprint(export_dir)
    return {
        "manifest_version": MANIFEST_VERSION,
        "run_id": settings["run_id"],
        "size": settings["size"],
        "seed": settings["seed"],
        "reward": settings["reward"],
        "reward_weights": dict(reward_weights),
        "max_steps": int(settings["budget"]["max_steps"]),
        "config_sha256": sha256_json(config),
        "config": config,
        "data": data_identity(settings["data_dir"], settings["train_split"]),
        "data_dir": str(settings["data_dir"]),
        "base_model": {"id": settings["model"], "export_sha256": fp["sha256"]},
        "export": {"dir": str(export_dir), **fp},
        "git": git_state(repo_root) if repo_root is not None else {"commit": None, "dirty": None},
    }


def _data_key(data: Any) -> Any:
    """The compared part of ``data`` (``matches_manifest`` is derived)."""
    if not isinstance(data, Mapping):
        return data
    return {k: data.get(k) for k in ("split", "sha256", "manifest_sha256", "spec_version")}


def compare_manifests(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Identity fields on which two manifests differ (empty: a valid matched pair)."""
    diffs = []
    for f in IDENTITY_FIELDS:
        x, y = a.get(f), b.get(f)
        if f == "data":
            x, y = _data_key(x), _data_key(y)
        if x != y:
            diffs.append(f)
    return diffs


def resume_mismatches(manifest: Mapping[str, Any]) -> list[str]:
    """Identity fields that changed on some resume of this run."""
    out: list[str] = []
    for start in manifest.get("starts") or []:
        for f in start.get("mismatch") or []:
            if f not in out:
                out.append(f)
    return out


def read_run_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    data = read_json(Path(run_dir) / MANIFEST_FILE)
    return data if isinstance(data, dict) else None


def write_run_manifest(
    run_dir: str | Path, manifest: Mapping[str, Any], *, resume_from: str | None = None
) -> list[str]:
    """Write the manifest at the start of a run; on a resume keep the first one and log the start.

    Returns the identity fields that differ from the run's first manifest (empty on a first start).
    """
    path = Path(run_dir) / MANIFEST_FILE
    start = {"resume_from": resume_from, "git": manifest.get("git"), "mismatch": []}
    existing = read_run_manifest(run_dir)
    if existing is None:
        atomic_write_json(path, {**manifest, "starts": [start]})
        return []
    diffs = compare_manifests(existing, manifest)
    if existing.get("reward") != manifest.get("reward"):
        diffs.append("reward")
    start["mismatch"] = diffs
    existing.setdefault("starts", []).append(start)
    atomic_write_json(path, existing)
    return diffs
