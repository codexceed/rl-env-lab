"""Node preflight: is this machine ready to train and evaluate? (``make preflight NODE=node1``)

    uv run python scripts/preflight.py --node node1    # what `make preflight` and train/coordinator.py run
    uv run python scripts/preflight.py                 # no node: GPU stack, ninja and GPU count only

Why: a studio stop/restart wipes uv's package cache on the system disk, which the venv's files link into, so imports
fail deep inside torch (``ModuleNotFoundError: torch.nn.utils.spectral_norm``); a plain ``uv sync`` (no
``--extra gpu``) then removes torch, vLLM and TRL. Jobs launched anyway failed one by one. Run this first:

| Check | FAIL when | Fix printed |
|---|---|---|
| gpu stack | torch, transformers, trl, vllm, causal_conv1d or fla fails to import (child, timeout) | make setup-gpu |
| ninja | not found with the venv's bin first on PATH (vLLM's sampler JIT-builds with it) | make setup-gpu |
| gpus | ``nvidia-smi`` sees fewer GPUs than ``nodes.<node>.gpus`` uses | a bigger machine |
| <dir>/ | a file listed in ``<dir>/manifest.json`` is missing or has another sha256 | make data / data-v3 |
| model exports | the text-only export of a size the node still trains is missing | train.export_text |
| disk | free space under ``paths.runs`` below a rough estimate of what the queue writes (WARN only) | free space |

The data dirs come from each of the node's runs (``--variant`` in its train command; v3 reads ``data/v3``). The v1
exploratory split (``manifest.exploratory.json``, same-type evals only) is a WARN with ``make data-sametype``.
Finished, failed and timed-out runs need no export and no disk.

Prints a table, then the fixes, and exits 1 on any FAIL (WARN never fails). `train/coordinator.py` runs it at startup
and refuses to launch on a failure (``--skip-preflight`` overrides). The checks take the subprocess runner, ``which``
and disk usage as arguments, so tests run them with fakes. No GPU import here: the stack is imported in a child.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envlab.tasks.v3 import DATA_DIR as V3_DATA_DIR
from train.checkpoints import TERMINAL_STATUSES, read_json, read_status, run_dir
from train.coordinator import build_plan, resolve_paths
from train.coordinator import load_config as load_allocation
from train.grpo import DEFAULT_CONFIG, load_config
from train.run_manifest import file_sha256
from train.toy_utils import model_slug
from train.venv_bin import path_with_venv_bin

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "runs" / "models"  # train/common.py MODELS_DIR (that module imports torch)
DEFAULT_ALLOCATION = Path("configs/allocation.yaml")

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

SETUP_GPU_FIX = "run `make setup-gpu` (a restart wipes the venv's package cache; a plain `uv sync` drops the gpu extra)"
# (name, import statement) run in a child interpreter; torch.nn.utils is what went missing after a restart
GPU_IMPORTS: tuple[tuple[str, str], ...] = (
    ("torch", "import torch, torch.nn.utils"),
    ("transformers", "import transformers"),
    ("trl", "from trl import GRPOTrainer"),
    ("vllm", "from vllm import LLM, SamplingParams"),
    ("causal_conv1d", "import causal_conv1d"),
    ("fla", "import fla"),
)
IMPORT_TIMEOUT_S = 300.0
IMPORT_MARKER = "PREFLIGHT_IMPORTS "
_IMPORT_SCRIPT = f"""
import json, sys
out = {{}}
for name, stmt in json.loads(sys.argv[1]):
    try:
        exec(stmt, {{}})
        out[name] = None
    except BaseException as exc:
        out[name] = f"{{type(exc).__name__}}: {{exc}}"
print({IMPORT_MARKER!r} + json.dumps(out), flush=True)
"""

# Disk estimate, deliberately rough: nominal parameters; bf16 eval weights (never deleted); recovery checkpoints as
# fp32 weights + Adam moments with slack (14 B/param: ~55 GB per 4B save, as measured), keep_last plus one in flight.
PARAMS = {"0.8b": 0.8e9, "2b": 2e9, "4b": 4e9}
EVAL_BYTES_PER_PARAM = 2
RECOVERY_BYTES_PER_PARAM = 14
_VARIANT_FLAG_RE = re.compile(r"--variant[= ]+(\S+)")  # as eval/watcher.py reads a run's train command

Runner = Callable[[Sequence[str], float, Mapping[str, str] | None], tuple[int, str]]
Which = Callable[..., str | None]


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | SKIP
    detail: str
    fix: str = ""


def run_command(argv: Sequence[str], timeout: float, env: Mapping[str, str] | None = None) -> tuple[int, str]:
    """(exit code, stdout + stderr); 127 when the program is missing, 124 on timeout."""
    try:
        proc = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, env=None if env is None else dict(env)
        )
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f} s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _tail(text: str, n: int = 3) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return " | ".join(lines[-n:]) or "no output"


def _show(path: Path) -> str:
    """``path`` relative to the working directory when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _gb(n: float) -> str:
    return f"{n / 1e9:,.0f} GB"


def variant_from_command(cmd: str) -> str | None:
    """The ``--variant`` of a train command template, or None (v1)."""
    m = _VARIANT_FLAG_RE.search(cmd or "")
    return m[1] if m else None


# ---------------------------------------------------------------- checks


def check_gpu_stack(
    runner: Runner = run_command, python: str | None = None, timeout: float = IMPORT_TIMEOUT_S
) -> Check:
    """Import the GPU stack in a child of ``python`` (the venv's), with the venv's bin first on PATH."""
    python = python or sys.executable
    env = dict(os.environ, PATH=path_with_venv_bin(os.environ.get("PATH"), python))
    rc, out = runner([python, "-c", _IMPORT_SCRIPT, json.dumps(GPU_IMPORTS)], timeout, env)
    line = next((x for x in reversed(out.splitlines()) if x.startswith(IMPORT_MARKER)), None)
    if line is None:
        return Check("gpu stack", FAIL, f"import check exited {rc}: {_tail(out)}", SETUP_GPU_FIX)
    errors = {k: v for k, v in json.loads(line[len(IMPORT_MARKER) :]).items() if v}
    if errors:
        return Check("gpu stack", FAIL, "; ".join(f"{k}: {v}" for k, v in errors.items()), SETUP_GPU_FIX)
    return Check("gpu stack", PASS, ", ".join(name for name, _ in GPU_IMPORTS) + " import")


def check_ninja(which: Which = shutil.which, path: str | None = None, python: str | None = None) -> Check:
    """``ninja`` on PATH with the venv's bin first, as `eval/heldout.py` and ``uv run`` set it."""
    search = path_with_venv_bin(os.environ.get("PATH") if path is None else path, python)
    found = which("ninja", path=search)
    if not found:
        return Check(
            "ninja", FAIL, "not found on PATH (venv bin first); vLLM's sampler JIT-builds with it", SETUP_GPU_FIX
        )
    return Check("ninja", PASS, str(found))


def check_gpus(runner: Runner = run_command, needed: int | None = None) -> Check:
    """GPUs visible to ``nvidia-smi``, at least ``needed`` (the node's config) when given."""
    rc, out = runner(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], 60.0, None)
    fix = "run on the node's GPU machine (from a laptop: `make studio-switch STUDIO=... MACHINE=H200_X_8`)"
    if rc != 0:
        return Check("gpus", FAIL, f"nvidia-smi failed (exit {rc}): {_tail(out)}", fix)
    count = sum(1 for line in out.splitlines() if line.strip().isdigit())
    if count == 0 or (needed is not None and count < needed):
        want = f", the node uses {needed}" if needed is not None else ""
        return Check("gpus", FAIL, f"{count} GPUs visible{want}", fix)
    return Check("gpus", PASS, f"{count} GPUs" + (f" (node uses {needed})" if needed is not None else ""))


def manifest_mismatches(data_dir: Path, manifest_name: str = "manifest.json") -> tuple[list[str], list[str]] | None:
    """(missing, changed) files of ``<data_dir>/<manifest_name>`` against their recorded sha256 (the hash
    `train/run_manifest.py` records for a run's data); None when the manifest is missing or has no ``files``."""
    manifest = read_json(Path(data_dir) / manifest_name, default=None)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        return None
    missing, changed = [], []
    for rel, sha in sorted(files.items()):
        path = Path(data_dir) / rel
        if not path.is_file():
            missing.append(rel)
        elif file_sha256(path) != sha:
            changed.append(rel)
    return missing, changed


def _describe(missing: list[str], changed: list[str]) -> str:
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing ({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''})")
    if changed:
        parts.append(f"{len(changed)} differ from the manifest ({', '.join(changed[:3])})")
    return "; ".join(parts)


def check_data(data_dir: Path, label: str | None = None) -> list[Check]:
    """The data set's files against its manifest; for v1 also the exploratory sametype split (WARN only)."""
    data_dir = Path(data_dir)
    label = label or str(data_dir)
    v3 = data_dir.as_posix().rstrip("/").endswith(V3_DATA_DIR)  # data/v3: the v3 task set (ADR 0007)
    fix = (
        "run `make data-v3`"
        if v3
        else "run `make data` (deterministic; never while runs on this node read data/, it rewrites the frozen files)"
    )
    name = f"{label.rstrip('/')}/"
    result = manifest_mismatches(data_dir)
    if result is None:
        return [Check(name, FAIL, f"no manifest.json with `files` in {data_dir}", fix + " (or restore it from git)")]
    missing, changed = result
    checks = [
        Check(name, FAIL, _describe(missing, changed), fix)
        if missing or changed
        else Check(name, PASS, "every file matches manifest.json")
    ]
    if not v3:
        sname, sfix = f"{name} sametype", "run `make data-sametype` (safe while runs read data/)"
        extra = manifest_mismatches(data_dir, "manifest.exploratory.json")
        if extra is None:
            checks.append(Check(sname, WARN, "no manifest.exploratory.json (same-type evals only)", sfix))
        elif extra[0] or extra[1]:
            checks.append(Check(sname, WARN, _describe(*extra) + " (same-type evals only)", sfix))
        else:
            checks.append(Check(sname, PASS, "every file matches manifest.exploratory.json"))
    return checks


def export_dir(model_id: str, models_dir: Path = MODELS_DIR) -> Path:
    """``runs/models/<name>-text``, as `train/common.py` ``export_dir_for`` names it."""
    return Path(models_dir) / f"{model_slug(model_id)}-text"


def check_exports(sizes: Sequence[str], models: Mapping[str, str], models_dir: Path = MODELS_DIR) -> Check:
    """A complete text-only export (``export_info.json`` is written last) for each size still to train."""
    if not sizes:
        return Check("model exports", PASS, "no run left to train")
    missing = [s for s in sizes if not (export_dir(models[s], models_dir) / "export_info.json").is_file()]
    if missing:
        where = ", ".join(f"{s} ({_show(export_dir(models[s], models_dir))})" for s in missing)
        cmds = "; ".join(f"`uv run python -m train.export_text --model {models[s]}`" for s in missing)
        return Check("model exports", FAIL, f"missing: {where}", f"run {cmds}")
    return Check("model exports", PASS, ", ".join(f"{s}: {export_dir(models[s], models_dir).name}" for s in sizes))


def eval_checkpoints(max_steps: int, every: int) -> int:
    """Eval weight dirs a run writes: every ``every`` steps plus the final step."""
    return max_steps // every + (1 if max_steps % every else 0)


def estimate_run_bytes(size: str, cfg: Mapping[str, Any]) -> int:
    """Rough bytes one run leaves on disk: its eval checkpoints plus the recovery checkpoints kept."""
    params = PARAMS.get(size, 4e9)
    ck = cfg["checkpoints"]
    evals = eval_checkpoints(int(cfg["budget"]["max_steps"]), int(ck["eval_every"]))
    recovery = int(ck.get("keep_last", 2)) + 1  # kept + one being written
    return int(params * (EVAL_BYTES_PER_PARAM * evals + RECOVERY_BYTES_PER_PARAM * recovery))


def check_disk(runs_root: Path, need: int, disk_usage: Callable[[Path], Any] = shutil.disk_usage) -> Check:
    """Free space under ``runs_root`` (or its nearest existing parent) against ``need``; WARN only."""
    probe = Path(runs_root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(disk_usage(probe).free)
    if free < need:
        return Check(
            "disk",
            WARN,
            f"{_gb(free)} free under {probe}; the queue may write ~{_gb(need)} (rough)",
            "free space, or point paths.runs at a bigger disk",
        )
    return Check("disk", PASS, f"{_gb(free)} free under {probe}; the queue writes ~{_gb(need)} (rough)")


# ---------------------------------------------------------------- what a node needs


@dataclass
class NodeNeeds:
    gpus: int  # GPUs the node's config uses (highest index + 1)
    data_dirs: list[Path] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)  # sizes still to train
    disk_bytes: int = 0
    runs_root: Path = Path("runs")
    models: dict[str, str] = field(default_factory=dict)
    remaining: list[str] = field(default_factory=list)  # run ids not finished, failed or timed out


def node_needs(alloc: dict[str, Any], node: str, grpo_config: Path = DEFAULT_CONFIG) -> NodeNeeds:
    """What ``node``'s runs need, from a path-resolved allocation config and the GRPO config (+ size, variant)."""
    plan = build_plan(alloc, node)
    s = plan.settings
    needs = NodeNeeds(gpus=max(plan.gpus) + 1 if plan.gpus else 0, runs_root=s.runs_root)
    needs.models = dict(load_config(grpo_config)["models"])
    dirs: dict[str, Path] = {}
    sizes: set[str] = set()
    for rid, spec in plan.runs.items():
        cfg = load_config(grpo_config, spec.size, variant_from_command(spec.train_command))
        data_dir = Path(cfg["data"]["dir"])
        data_dir = data_dir if data_dir.is_absolute() else Path(s.workdir) / data_dir
        dirs.setdefault(str(data_dir), data_dir)
        status = (read_status(run_dir(s.runs_root, rid)) or {}).get("status")
        if status in TERMINAL_STATUSES:
            continue
        needs.remaining.append(rid)
        sizes.add(spec.size)
        needs.disk_bytes += estimate_run_bytes(spec.size, cfg)
    needs.data_dirs = list(dirs.values())
    needs.sizes = sorted(sizes, key=lambda x: float(x.rstrip("b")))
    return needs


def run_checks(
    needs: NodeNeeds | None,
    *,
    runner: Runner = run_command,
    which: Which = shutil.which,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    python: str | None = None,
    models_dir: Path = MODELS_DIR,
    workdir: Path | None = None,
    import_timeout: float = IMPORT_TIMEOUT_S,
) -> list[Check]:
    """Every check; without ``needs`` (no node) only the GPU stack, ninja and the GPU count."""
    checks = [
        check_gpu_stack(runner, python, import_timeout),
        check_ninja(which, python=python),
        check_gpus(runner, needs.gpus if needs else None),
    ]
    if needs is None:
        checks.append(Check("node", SKIP, "no --node: data, model exports and disk not checked"))
        return checks
    base = Path(workdir or Path.cwd())
    for d in needs.data_dirs:
        label = os.path.relpath(d, base) if d.is_absolute() else str(d)
        checks += check_data(d, label)
    checks.append(check_exports(needs.sizes, needs.models, models_dir))
    checks.append(check_disk(needs.runs_root, needs.disk_bytes, disk_usage))
    return checks


def failed(checks: Sequence[Check]) -> bool:
    return any(c.status == FAIL for c in checks)


def report(checks: Sequence[Check], node: str | None = None) -> str:
    """The pass/fail table, then one fix line per FAIL or WARN, then the verdict."""
    width = max(len(c.name) for c in checks)
    lines = [f"preflight {node or '(no node)'}", f"{'check':<{width}}  status  detail"]
    lines += [f"{c.name:<{width}}  {c.status:<6}  {c.detail}" for c in checks]
    fixes = [c for c in checks if c.fix and c.status in (FAIL, WARN)]
    if fixes:
        lines.append("")
        lines += [f"fix {c.name}: {c.fix}" for c in fixes]
    n_fail = sum(c.status == FAIL for c in checks)
    lines.append("")
    lines.append(f"preflight FAILED: {n_fail} check(s); fix them before launching" if n_fail else "preflight passed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Node preflight before launching training or evals.")
    ap.add_argument("--node", default=None, help="node in the allocation config (data, exports, GPUs, disk)")
    ap.add_argument("--config", type=Path, default=DEFAULT_ALLOCATION, help="allocation config")
    ap.add_argument("--grpo-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--import-timeout", type=float, default=IMPORT_TIMEOUT_S, help="seconds for the GPU stack imports")
    args = ap.parse_args(argv)

    needs = None
    if args.node:
        alloc = resolve_paths(load_allocation(args.config))
        needs = node_needs(alloc, args.node, args.grpo_config)
    checks = run_checks(needs, import_timeout=args.import_timeout, workdir=Path.cwd())
    print(report(checks, args.node), flush=True)
    return 1 if failed(checks) else 0


if __name__ == "__main__":
    sys.exit(main())
