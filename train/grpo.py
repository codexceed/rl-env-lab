"""GRPO training run for one (size, reward, seed) of the run matrix (PROGRESSION §1.4).

    uv run python train/grpo.py --run-id 2b-rob-s1 --size 2b --reward robust --seed 1 \\
        --gpus 0 --output-dir runs/2b-rob-s1                                   # `make train RUN=2b-rob-s1`
    ... --max-steps 10                                                          # dry run
    ... --resume-from runs/2b-rob-s1/recovery/checkpoint-50                     # coordinator crash relaunch
    ... --hf-generate                                                           # fallback: no vLLM
    uv run python train/grpo.py --run-id 4b-rob-s1 --size 4b --reward robust --seed 1 \\
        --gpus 0,1,2,3 --vllm-server-gpus 0 --train-gpus 1,2,3 --output-dir runs/4b-rob-s1

Full fine-tune with TRL `GRPOTrainer` on the common budget in `configs/grpo.yaml`; the GPU layout
(and only otherwise the recovery-checkpoint interval) comes from `configs/sizes/<size>.yaml`:

* **1-GPU runs** (0.8B, 2B): one process, colocated vLLM.
* **4B, server layout**: this process starts (or attaches to) `trl vllm-serve` on the server GPU,
  then runs itself under `accelerate launch` with ZeRO-3 (`configs/accelerate/zero3_<n>gpu.yaml`)
  on the training GPUs. 64 completions per step must split evenly over the ranks: 2 ranks, not 3.
* **4B, colocate layout**: `accelerate launch` on 4 GPUs, each rank with a colocated engine.
* ``--hf-generate``: HF generate in the trainer (bf16 autocast subclass from `train/common.py`).

GPU indices (``--gpus`` etc.) are physical; ``CUDA_VISIBLE_DEVICES`` is set from them for every
child. Ports are offset by the run's lowest GPU index so runs sharing a node never collide.

Artifacts (`train/LAYOUT.md`): eval weights ``eval/step-NNNNN/`` every 50 steps and at the final
step, recovery checkpoints ``recovery/checkpoint-N/`` (full trainer state), ``COMPLETE`` written
last in both, and ``train_metrics.jsonl`` one row per optimizer step (`train/callbacks.py`). Rolling
recovery pruning only with ``checkpoints.pruning_enabled: true``. With ``checkpoints.async_recovery``
recovery checkpoints are snapshotted to CPU memory and written by a background thread per rank
(`train/async_recovery.py`); a save step that finds the previous write still in flight is skipped.
At start the main rank writes
``run_manifest.json`` (`train/run_manifest.py`): config hash without the reward, data and base-model
fingerprints, seed, max_steps, git commit; `eval/compare.py` checks matched pairs against it.

Prompts are rendered once with the SPEC §10 call (`envlab.prompts.render_prompt`, thinking off) and
given to TRL as plain text; every rendered prompt is checked against the 1,536-token budget, and
prompt + answer cap against ``layout.vllm_max_model_length``.

**Variants** (``--variant NAME``, ADR 0005): ``configs/variants/<NAME>.yaml`` is merged over the
config after the size file and may set only `VARIANT_KEYS`. ``v2-think`` turns thinking on in the
training prompts, caps answers at 1,024 and widens the vLLM context to 2,560 (ADR 0006). The
variant is recorded in ``train_config.json`` and
``run_manifest.json`` (so a v2 run never pairs with a v1 run), and the graders see `graded_text`
(the prompt's open think tag restored; an unclosed think block is a format failure), exactly as
`eval/heldout.py` grades that run's checkpoints. Without a variant nothing changes. The
instance travels in the ``instance`` column to the reward functions (`train/rewards.py`). Both
graders are scored on every generation (weight 1 on the run's own reward, 0 on the other), so
``train_metrics.jsonl`` logs ``robust_reward_mean`` and ``leaky_reward_mean`` every step.

Exit code: 0 when training reached its last step and a final eval step exists; non-zero otherwise.
Never shuts the machine down. Nothing at module level imports torch, so CI tests the CLI, config
and layout logic.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `train` and `eval` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from train.checkpoints import COMPLETE, final_eval_checkpoint
from train.toy_utils import grpo_batch_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "grpo.yaml"
SIZES_DIR_NAME = "sizes"
VARIANTS_DIR_NAME = "variants"
ACCELERATE_DIR = REPO_ROOT / "configs" / "accelerate"
SIZES = ("0.8b", "2b", "4b")
REWARDS = ("robust", "leaky")
# The top-level keys a configs/sizes/<size>.yaml may set: the GPU layout, and under `checkpoints` only
# the recovery interval (how often crash state is saved; it never changes training). The budget is
# common to all sizes (H5).
SIZE_OVERRIDE_KEYS = frozenset({"layout", "checkpoints"})
SIZE_CHECKPOINT_KEYS = frozenset({"recovery_every"})
LAYOUT_KEYS = frozenset(
    {
        "generation",
        "train_ranks",
        "hf_train_ranks",
        "vllm_server_gpus",
        "vllm_gpu_memory_utilization",
        "vllm_server_gpu_memory_utilization",
        "vllm_max_model_length",
    }
)
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"  # SPEC §10: generation prompt with enable_thinking=False
# The keys a configs/variants/<name>.yaml may set (ADR 0005 v2, ADR 0006): thinking, the answer cap and the
# vLLM context that cap needs. Nothing else, so a variant run differs from v1 only where the ADRs say it does.
VARIANT_KEYS: dict[str, frozenset[str]] = {
    "budget": frozenset({"enable_thinking", "max_completion_length"}),
    "layout": frozenset({"vllm_max_model_length"}),
}


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- CLI


def _gpu_list(text: str) -> list[int]:
    try:
        gpus = [int(x) for x in str(text).split(",") if x.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated GPU indices, got {text!r}") from exc
    if len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise argparse.ArgumentTypeError(f"bad GPU list {text!r}")
    return gpus


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO run (PROGRESSION §1.4)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--size", required=True, choices=SIZES)
    p.add_argument("--reward", required=True, choices=REWARDS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument(
        "--gpus",
        type=_gpu_list,
        default=None,
        help="physical GPU indices, e.g. 0 or 0,1,2,3 (default: CUDA_VISIBLE_DEVICES, else 0)",
    )
    p.add_argument("--output-dir", required=True, type=Path, help="the run dir, runs/<run_id>")
    p.add_argument("--resume-from", type=Path, help="recovery/checkpoint-N with COMPLETE")
    p.add_argument("--vllm-server-gpus", type=_gpu_list, help="4B server layout: physical GPU(s) for vLLM")
    p.add_argument("--train-gpus", type=_gpu_list, help="4B server layout: physical training GPUs")
    p.add_argument("--max-steps", type=int, help="override the step count (dry runs)")
    p.add_argument("--eval-every", type=int, help="override checkpoints.eval_every (dry runs)")
    p.add_argument("--recovery-every", type=int, help="override checkpoints.recovery_every (dry runs)")
    p.add_argument("--hf-generate", action="store_true", help="fallback: HF generate instead of vLLM")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--variant", help="ADR 0005 variant overlay configs/variants/<name>.yaml, e.g. v2-think")
    p.add_argument("--data-dir", type=Path, help="override data.dir")
    p.add_argument("--port-offset", type=int, help="override the port offset (default: lowest GPU index)")
    p.add_argument("--vllm-server-url", help="attach to a running vLLM server instead of starting one")
    p.add_argument("--server-timeout", type=float, default=900.0, help="seconds to wait for the vLLM server")
    p.add_argument("--allow-slow-kernels", action="store_true", help="run even if fla/causal-conv1d are missing")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # set when re-launched by accelerate
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    if args.gpus is None:
        try:
            args.gpus = _gpu_list(os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
        except argparse.ArgumentTypeError as exc:
            p.error(f"--gpus not given and CUDA_VISIBLE_DEVICES is unusable: {exc}")
    for name in ("max_steps", "eval_every", "recovery_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.train_gpus is not None and args.vllm_server_gpus is None:
        p.error("--train-gpus needs --vllm-server-gpus")
    if args.vllm_server_gpus is not None:
        rest = set(args.vllm_server_gpus) | set(args.train_gpus or [])
        if not rest <= set(args.gpus):
            p.error("--vllm-server-gpus and --train-gpus must be subsets of --gpus")
        if set(args.vllm_server_gpus) & set(args.train_gpus or []):
            p.error("--vllm-server-gpus and --train-gpus overlap")
    return args


# ---------------------------------------------------------------- config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def size_override_path(config_path: Path, size: str) -> Path:
    return Path(config_path).parent / SIZES_DIR_NAME / f"{size}.yaml"


def load_size_override(config_path: Path, size: str) -> dict[str, Any]:
    """configs/sizes/<size>.yaml, checked to hold GPU-layout keys and ``checkpoints.recovery_every`` only."""
    path = size_override_path(config_path, size)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    extra = set(data) - SIZE_OVERRIDE_KEYS
    if extra:
        raise ConfigError(f"{path}: size overrides may only set {sorted(SIZE_OVERRIDE_KEYS)}, got {sorted(extra)}")
    layout = data.get("layout") or {}
    if not isinstance(layout, dict):
        raise ConfigError(f"{path}: layout must be a mapping")
    unknown = set(layout) - LAYOUT_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown layout keys {sorted(unknown)}")
    ckpt = data.get("checkpoints") or {}
    if not isinstance(ckpt, dict):
        raise ConfigError(f"{path}: checkpoints must be a mapping")
    unknown = set(ckpt) - SIZE_CHECKPOINT_KEYS
    if unknown:
        raise ConfigError(
            f"{path}: size overrides may only set checkpoints.{sorted(SIZE_CHECKPOINT_KEYS)}, got {sorted(unknown)}"
        )
    every = ckpt.get("recovery_every", 1)
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ConfigError(f"{path}: checkpoints.recovery_every must be an integer >= 1")
    return data


def variant_path(config_path: Path, variant: str) -> Path:
    return Path(config_path).parent / VARIANTS_DIR_NAME / f"{variant}.yaml"


def load_variant_override(config_path: Path, variant: str) -> dict[str, Any]:
    """configs/variants/<variant>.yaml, checked to hold `VARIANT_KEYS` only, with sane values."""
    if not variant or "/" in variant or variant.startswith("."):
        raise ConfigError(f"bad variant name {variant!r}")
    path = variant_path(config_path, variant)
    if not path.is_file():
        raise ConfigError(f"unknown variant {variant!r}: {path} not found")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    extra = set(data) - set(VARIANT_KEYS)
    if extra:
        raise ConfigError(f"{path}: variants may only set {sorted(VARIANT_KEYS)}, got {sorted(extra)}")
    for section, allowed in VARIANT_KEYS.items():
        values = data.get(section) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"{path}: {section} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ConfigError(f"{path}: variants may only set {section}.{sorted(allowed)}, got {sorted(unknown)}")
    budget = data.get("budget") or {}
    if "enable_thinking" in budget and not isinstance(budget["enable_thinking"], bool):
        raise ConfigError(f"{path}: budget.enable_thinking must be true or false")
    cap = budget.get("max_completion_length", 1)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ConfigError(f"{path}: budget.max_completion_length must be an integer >= 1")
    ctx = (data.get("layout") or {}).get("vllm_max_model_length", 1)
    if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1:
        raise ConfigError(f"{path}: layout.vllm_max_model_length must be an integer >= 1")
    return data


def check_context(cfg: dict[str, Any], max_model_len: int) -> None:
    """Fail fast unless the longest allowed prompt plus the answer cap fits the vLLM context."""
    prompt, cap = int(cfg["budget"]["max_prompt_length"]), int(cfg["budget"]["max_completion_length"])
    if prompt + cap > max_model_len:
        raise ConfigError(
            f"max_prompt_length {prompt} + max_completion_length {cap} > vllm_max_model_length {max_model_len}"
        )


def load_config(
    config_path: Path = DEFAULT_CONFIG, size: str | None = None, variant: str | None = None
) -> dict[str, Any]:
    """configs/grpo.yaml, with configs/sizes/<size>.yaml merged into `layout` when `size` is given,
    then configs/variants/<variant>.yaml (ADR 0005) when `variant` is given (``cfg["variant"]``)."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text()) or {}
    for section in ("models", "data", "budget", "checkpoints", "ports", "layout", "eval"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"{config_path}: missing section {section!r}")
    unknown = set(cfg["layout"]) - LAYOUT_KEYS
    if unknown:
        raise ConfigError(f"{config_path}: unknown layout keys {sorted(unknown)}")
    if size is not None:
        if size not in cfg["models"]:
            raise ConfigError(f"{config_path}: no model for size {size!r}")
        cfg = _deep_merge(cfg, load_size_override(config_path, size))
        cfg["size"] = size
        cfg["model"] = cfg["models"][size]
    if variant:
        cfg = _deep_merge(cfg, load_variant_override(config_path, variant))
        cfg["variant"] = variant
    return cfg


# ---------------------------------------------------------------- layout and ports


@dataclass(frozen=True)
class Layout:
    """Where each part of one run runs. GPU indices are physical."""

    generation: str  # "colocate" | "server" | "hf"
    train_gpus: tuple[int, ...]
    server_gpus: tuple[int, ...] = ()
    idle_gpus: tuple[int, ...] = ()
    port_offset: int = 0
    master_port: int = 29500
    vllm_port: int = 8000
    group_port: int = 51216
    vllm_gpu_memory_utilization: float = 0.3
    vllm_server_gpu_memory_utilization: float = 0.9
    vllm_max_model_length: int = 2048
    batch: dict[str, int] = field(default_factory=dict)

    @property
    def train_ranks(self) -> int:
        return len(self.train_gpus)

    @property
    def vllm_mode(self) -> str:
        return "none" if self.generation == "hf" else self.generation

    @property
    def server_url(self) -> str:
        return f"http://localhost:{self.vllm_port}"

    @property
    def accelerate_config(self) -> Path | None:
        return ACCELERATE_DIR / f"zero3_{self.train_ranks}gpu.yaml" if self.train_ranks > 1 else None


def resolve_layout(cfg: dict[str, Any], args: argparse.Namespace) -> Layout:
    """GPU layout for this run from the merged config and the CLI. Raises `ConfigError` if it can't fit."""
    lay = cfg["layout"]
    budget = cfg["budget"]
    gpus: list[int] = list(args.gpus)
    generation = "hf" if args.hf_generate else str(lay.get("generation", "colocate"))
    if generation not in ("colocate", "server", "hf"):
        raise ConfigError(f"layout.generation must be colocate or server, got {generation!r}")

    server: list[int] = []
    if generation == "hf":
        n_train = int(lay.get("hf_train_ranks", 1))
        pool = gpus
    elif generation == "colocate":
        n_train = int(lay.get("train_ranks", 1))
        pool = gpus
    else:
        n_train = int(lay.get("train_ranks", 1))
        n_server = int(lay.get("vllm_server_gpus", 1))
        if n_server < 1:
            raise ConfigError("server layout needs layout.vllm_server_gpus >= 1")
        if args.vllm_server_gpus is not None:
            server = list(args.vllm_server_gpus)[:n_server]
            pool = list(args.train_gpus) if args.train_gpus is not None else [g for g in gpus if g not in server]
        else:
            server = gpus[:n_server]
            pool = gpus[n_server:]
        if len(server) < n_server:
            raise ConfigError(f"server layout needs {n_server} vLLM GPU(s), got {server}")
    if n_train < 1:
        raise ConfigError("need at least one training rank")
    if len(pool) < n_train:
        raise ConfigError(f"{generation} layout needs {n_train} training GPU(s), got {pool} (from --gpus {gpus})")
    train = pool[:n_train]
    used = set(train) | set(server)
    idle = tuple(g for g in gpus if g not in used)
    try:
        batch = grpo_batch_layout(int(budget["prompts_per_step"]), int(budget["num_generations"]), n_train)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    ports = cfg["ports"]
    offset = int(args.port_offset) if args.port_offset is not None else min(gpus)
    return Layout(
        generation=generation,
        train_gpus=tuple(train),
        server_gpus=tuple(server),
        idle_gpus=idle,
        port_offset=offset,
        master_port=int(ports["master_base"]) + offset,
        vllm_port=int(ports["vllm_base"]) + offset,
        group_port=int(ports["group_base"]) + offset,
        vllm_gpu_memory_utilization=float(lay.get("vllm_gpu_memory_utilization", 0.3)),
        vllm_server_gpu_memory_utilization=float(lay.get("vllm_server_gpu_memory_utilization", 0.9)),
        vllm_max_model_length=int(lay.get("vllm_max_model_length", 2048)),
        batch=batch,
    )


def csv(gpus: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(g) for g in gpus)


def run_settings(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """The resolved settings of this run (CLI overrides applied), as written to train_config.json."""
    budget = dict(cfg["budget"])
    ckpt = dict(cfg["checkpoints"])
    if args.max_steps is not None:
        budget["max_steps"] = args.max_steps
    if args.eval_every is not None:
        ckpt["eval_every"] = args.eval_every
    if args.recovery_every is not None:
        ckpt["recovery_every"] = args.recovery_every
    settings = {
        "run_id": args.run_id,
        "size": args.size,
        "reward": args.reward,
        "seed": args.seed,
        "model": cfg["model"],
        "data_dir": str(args.data_dir or cfg["data"]["dir"]),
        "train_split": cfg["data"]["train_split"],
        "budget": budget,
        "checkpoints": ckpt,
        "resume_from": str(args.resume_from) if args.resume_from else None,
    }
    if cfg.get("variant"):  # only then: a v1 run's settings (and identity hash) stay byte-identical
        settings["variant"] = cfg["variant"]
    return settings


def layout_identity(layout: Layout) -> dict[str, Any]:
    """The part of the layout a matched pair must share: no GPU indices, no ports."""
    return {"generation": layout.generation, "train_ranks": layout.train_ranks, "batch": dict(layout.batch)}


def worker_command(args: argparse.Namespace, argv: list[str], layout: Layout) -> list[str]:
    """`accelerate launch` command that re-runs this script as the training worker on the train GPUs."""
    cfg_path = layout.accelerate_config
    if cfg_path is None:
        raise ConfigError("a single training rank runs in-process, not under accelerate")
    return [
        _bin("accelerate"),
        "launch",
        "--config_file",
        str(cfg_path),
        "--num_processes",
        str(layout.train_ranks),
        "--main_process_port",
        str(layout.master_port),
        str(Path(__file__).resolve()),
        *argv,
        # pin what the launcher resolved, so every worker derives the same layout and ports
        "--gpus",
        csv(args.gpus),
        "--port-offset",
        str(layout.port_offset),
        "--worker",
    ]


def _bin(name: str) -> str:
    path = Path(sys.executable).with_name(name)
    return str(path) if path.exists() else name


def check_resume_dir(path: Path) -> None:
    if not path.is_dir():
        raise ConfigError(f"--resume-from {path} is not a directory")
    if not (path / COMPLETE).is_file():
        raise ConfigError(f"--resume-from {path} has no {COMPLETE} marker (partial checkpoint)")


# ---------------------------------------------------------------- vLLM server (4B server layout)


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


class VllmServer:
    """`trl vllm-serve` on the server GPU(s), or an already-running server at the same URL.

    Same command as `train/bench.py`'s server, but started in this run's process group (no new
    session), so the coordinator's timeout kill of the run also stops the server.
    """

    def __init__(self, model_dir: Path, layout: Layout, log_path: Path, timeout: float, url: str | None = None):
        self.url = url or layout.server_url
        self.attach_only = url is not None
        self.cmd = [
            _bin("trl"),
            "vllm-serve",
            "--model",
            str(model_dir),
            "--port",
            str(layout.vllm_port),
            "--gpu_memory_utilization",
            str(layout.vllm_server_gpu_memory_utilization),
            "--max_model_len",
            str(layout.vllm_max_model_length),
        ]
        self.gpus = layout.server_gpus
        self.log_path = log_path
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._log = None

    def __enter__(self) -> VllmServer:
        if _healthy(self.url):
            print(f"[grpo] attaching to running vLLM server at {self.url}", flush=True)
            return self
        if self.attach_only:
            raise RuntimeError(f"no healthy vLLM server at {self.url}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": csv(self.gpus)}
        print(f"[grpo] vLLM server: CUDA_VISIBLE_DEVICES={csv(self.gpus)} {' '.join(self.cmd)}", flush=True)
        self.proc = subprocess.Popen(self.cmd, env=env, cwd=REPO_ROOT, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.timeout
        while not _healthy(self.url):
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited with {self.proc.returncode}; see {self.log_path}")
            if time.time() > deadline:
                raise TimeoutError(f"vLLM server not healthy after {self.timeout}s; see {self.log_path}")
            time.sleep(5)
        print(f"[grpo] vLLM server healthy at {self.url}", flush=True)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self._log is not None:
            self._log.close()


# ---------------------------------------------------------------- launcher


def _raise_on_sigterm(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def launch(args: argparse.Namespace, argv: list[str], cfg: dict[str, Any], layout: Layout) -> int:
    """Launcher: export the model, start the vLLM server if needed, run the worker(s)."""
    if layout.train_ranks == 1:
        # Before torch is imported: CUDA reads the variable once per process.
        os.environ["CUDA_VISIBLE_DEVICES"] = csv(layout.train_gpus)
        os.environ.setdefault("MASTER_PORT", str(layout.master_port))
    from train import common  # GPU stack

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    export_dir = common.ensure_export(cfg["model"])  # before any process group / ZeRO-3 init
    print(f"[grpo] layout: {json.dumps(asdict(layout))}", flush=True)
    if layout.idle_gpus:
        print(f"[grpo] idle GPUs in this layout: {list(layout.idle_gpus)}", flush=True)

    if layout.train_ranks == 1:
        if layout.generation == "server":
            log_path = args.output_dir / "logs" / "vllm-server.log"
            with VllmServer(export_dir, layout, log_path, args.server_timeout, args.vllm_server_url):
                return run_worker(args, cfg, layout)
        return run_worker(args, cfg, layout)

    cmd = worker_command(args, argv, layout)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": csv(layout.train_gpus)}
    env.pop("MASTER_PORT", None)

    def run_children() -> int:
        print(f"[grpo] workers: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT)
        try:
            return proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()

    if layout.generation == "server":
        log_path = args.output_dir / "logs" / "vllm-server.log"
        with VllmServer(export_dir, layout, log_path, args.server_timeout, args.vllm_server_url):
            return run_children()
    return run_children()


# ---------------------------------------------------------------- worker (GPU)


def load_train_rows(
    cfg: dict[str, Any], settings: dict[str, Any], tokenizer: Any, max_model_len: int | None = None
) -> list[dict[str, Any]]:
    """Dataset rows: rendered prompt (plain text), serialized instance, id and template.

    Raises if a prompt exceeds the token budget, does not match the thinking mode (thinking off
    unless ``budget.enable_thinking``, set only by a variant), or with the answer cap does not fit
    ``max_model_len``.
    """
    from envlab.tasks.instance import read_jsonl

    try:
        from envlab.prompts import render_prompt
    except ImportError as exc:  # the tasks workstream (SPEC §11.8) is not merged yet
        raise RuntimeError("envlab.prompts is missing: merge the tasks workstream (SPEC §11.8) first") from exc
    from train.rewards import serialize_instance

    path = Path(settings["data_dir"]) / f"{settings['train_split']}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `make data`")
    instances = read_jsonl(path)
    if not instances:
        raise RuntimeError(f"{path} is empty")
    limit = int(cfg["budget"]["max_prompt_length"])
    cap = int(cfg["budget"]["max_completion_length"])
    thinking = bool(cfg["budget"].get("enable_thinking", False))
    rows, too_long, lengths = [], [], []
    for inst in instances:
        text = render_prompt(tokenizer, inst, thinking=thinking)
        if thinking and text.endswith(THINK_OFF_SUFFIX):
            raise RuntimeError(f"{inst.id}: thinking on, but the prompt ends with the thinking-off generation prompt")
        if not thinking and not text.endswith(THINK_OFF_SUFFIX):
            raise RuntimeError(f"{inst.id}: prompt does not end with the thinking-off generation prompt")
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        if n > limit or (max_model_len is not None and n + cap > max_model_len):
            too_long.append((inst.id, n))
        rows.append(
            {"prompt": text, "instance": serialize_instance(inst), "instance_id": inst.id, "template": inst.template}
        )
    if too_long:
        raise RuntimeError(
            f"{len(too_long)} prompts exceed {limit} tokens (SPEC §10) or, with the {cap}-token answer cap, "
            f"the {max_model_len}-token context; e.g. {too_long[:5]}"
        )
    print(f"[grpo] {len(rows)} train prompts from {path}; tokens min/max {min(lengths)}/{max(lengths)}", flush=True)
    return rows


def reward_decoder(tokenizer: Any, thinking_prompt: str | None = None) -> Callable[[Any], str]:
    """completion_ids -> the text the graders see: decoded with special tokens kept; with thinking on
    (``thinking_prompt`` = a rendered prompt) as `envlab.prompts.graded_text`, as eval/heldout.py does."""
    if thinking_prompt is None:
        return lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    from envlab.prompts import graded_text, think_prefix

    prefix = think_prefix(thinking_prompt)
    return lambda ids: graded_text(tokenizer.decode(ids, skip_special_tokens=False), prefix)


def run_worker(args: argparse.Namespace, cfg: dict[str, Any], layout: Layout) -> int:
    """One training process (rank). Single-GPU runs call it in-process; ZeRO-3 runs via accelerate."""
    from train import common  # GPU stack

    common.setup_torch()
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed

    from train.callbacks import ArtifactCallback, remove_incomplete_recovery
    from train.checkpoints import atomic_write_json
    from train.rewards import REWARD_KINDS, GraderErrors, reward_setup
    from train.run_manifest import build_run_manifest, write_run_manifest

    settings = run_settings(cfg, args)
    budget, ckpt = settings["budget"], settings["checkpoints"]
    run_dir: Path = args.output_dir
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    main_rank = common.is_main_rank()
    set_seed(args.seed)

    if args.resume_from is not None:
        check_resume_dir(args.resume_from)
    if main_rank:
        for path in remove_incomplete_recovery(run_dir, keep=args.resume_from):
            print(f"[grpo] removed partial recovery checkpoint {path}", flush=True)

    kernels = common.require_fast_kernels(args.allow_slow_kernels)
    export_dir = common.export_dir_for(cfg["model"])
    if not common.is_export(export_dir):
        if common.world_size() > 1:
            raise RuntimeError(f"no text-only export at {export_dir}; the launcher should have made it")
        export_dir = common.ensure_export(cfg["model"])
    tokenizer = common.load_tokenizer(export_dir)
    special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    print(f"[grpo] <tool_call> is a special token: {'<tool_call>' in special} (rewards decode with specials kept)")

    thinking = bool(budget.get("enable_thinking", False))
    if thinking and not settings.get("variant"):
        raise ConfigError("budget.enable_thinking must be false (SPEC §10); only a variant (ADR 0005) turns it on")
    rows = load_train_rows(cfg, settings, tokenizer, layout.vllm_max_model_length)

    # Both graders on every generation; weight 1 on the run's own (train/rewards.py). Rewards decode
    # completion_ids with special tokens kept, exactly as eval/heldout.py does.
    if thinking:
        print(f"[grpo] variant {settings['variant']}: thinking on, answer cap {budget['max_completion_length']}")
    funcs, weights = reward_setup(
        args.reward, decode=reward_decoder(tokenizer, rows[0]["prompt"] if thinking else None)
    )
    if main_rank:  # matched-pair evidence for eval/compare.py (train/run_manifest.py)
        manifest = build_run_manifest(
            settings,
            reward_weights=dict(zip(REWARD_KINDS, weights, strict=True)),
            export_dir=export_dir,
            layout=layout_identity(layout),
            repo_root=REPO_ROOT,
        )
        changed = write_run_manifest(run_dir, manifest, resume_from=settings["resume_from"])
        if changed:
            print(f"[grpo] WARNING: run identity changed since the first start: {changed}", flush=True)
    config = common.grpo_config(
        run_dir / "recovery",
        lr=float(budget["learning_rate"]),
        max_steps=int(budget["max_steps"]),
        layout=layout.batch,
        num_generations=int(budget["num_generations"]),
        max_completion_length=int(budget["max_completion_length"]),
        vllm_mode=layout.vllm_mode,
        vllm_server_url=args.vllm_server_url or layout.server_url,
        vllm_group_port=layout.group_port,
        vllm_gpu_memory_utilization=layout.vllm_gpu_memory_utilization,
        vllm_max_model_length=layout.vllm_max_model_length,
        seed=args.seed,
        lr_scheduler_type=str(budget["lr_scheduler_type"]),
        temperature=float(budget["temperature"]),
        beta=float(budget["beta"]),
        warmup_steps=int(budget["warmup_steps"]),
        loss_type=str(budget["loss_type"]),
        scale_rewards=str(budget["scale_rewards"]),
        bf16=bool(budget["bf16"]),
        gradient_checkpointing=bool(budget["gradient_checkpointing"]),
        chat_template_kwargs={"enable_thinking": thinking},
        save_strategy="steps",
        save_steps=int(ckpt["recovery_every"]),
        save_total_limit=None,  # never let the Trainer rotate; pruning is ours and COMPLETE-aware
        save_only_model=False,  # full state: model, optimizer, scheduler, RNG, trainer state
        logging_steps=1,
        reward_weights=weights,
    )
    model = common.load_policy(export_dir, str(budget["param_dtype"]))

    async_recovery = bool(ckpt.get("async_recovery", False))
    callback_cls = type("GRPOArtifactCallback", (ArtifactCallback, TrainerCallback), {})
    callback = callback_cls(
        run_dir,
        eval_every=int(ckpt["eval_every"]),
        keep_last=int(ckpt["keep_last"]),
        pruning_enabled=bool(ckpt["pruning_enabled"]),
        is_main=main_rank,
        marker_extra={
            "run_id": args.run_id,
            "size": args.size,
            "reward": args.reward,
            "seed": args.seed,
            **({"variant": settings["variant"]} if settings.get("variant") else {}),
        },
        async_recovery=async_recovery,
    )
    trainer_cls = common.grpo_trainer_class(hf_generate=layout.generation == "hf", async_recovery=async_recovery)
    trainer = trainer_cls(
        model=model,
        reward_funcs=funcs,
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    callback.barrier = trainer.accelerator.wait_for_everyone
    callback.is_main = trainer.accelerator.is_main_process
    callback.save_eval_weights = _eval_weight_saver(trainer, tokenizer, str(ckpt["eval_weights_dtype"]))
    callback.extra_metrics = lambda: {"grader_errors": GraderErrors.count}
    recovery = None
    if async_recovery:
        from train.async_recovery import AsyncRecovery, any_rank_busy

        recovery = AsyncRecovery(
            run_dir,
            rank=int(trainer.args.process_index),
            world_size=int(trainer.args.world_size),
            is_main=callback.is_main,
            keep_last=callback.keep_last,
            pruning_enabled=callback.pruning_enabled,
            marker_extra=callback.marker_extra,
            any_busy=any_rank_busy(trainer.accelerator.device),
        )
        trainer.async_recovery = recovery

    if callback.is_main:
        atomic_write_json(
            run_dir / "train_config.json",
            {
                **settings,
                "layout": asdict(layout),
                "export_dir": str(export_dir),
                "kernels_slow_fallback": kernels.get("slow_fallback"),
                "world_size": common.world_size(),
                "versions": common.package_versions(),
            },
        )
    t0 = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    except Exception:
        if recovery is not None:  # let an in-flight recovery write finish: a later resume point
            from train.async_recovery import CRASH_JOIN_TIMEOUT_S

            recovery.join(CRASH_JOIN_TIMEOUT_S)
        raise
    elapsed = time.perf_counter() - t0
    if recovery is not None:
        from train.async_recovery import JOIN_TIMEOUT_S

        # Every rank joins its own writer; the main rank's writer also waits for the others' done
        # files (files, not collectives), so joining before the barrier cannot deadlock.
        recovery.join(JOIN_TIMEOUT_S)
    trainer.accelerator.wait_for_everyone()
    if not callback.is_main:
        return 0
    if recovery is not None:  # every rank has joined its writer: drop staging dirs of abandoned checkpoints
        recovery.remove_staging()
        print(
            f"[grpo] async recovery: written {recovery.completed}, skipped {recovery.skipped}, "
            f"failed {recovery.failed}",
            flush=True,
        )
    final = final_eval_checkpoint(run_dir)
    max_steps = int(budget["max_steps"])
    ok = trainer.state.global_step >= max_steps and final is not None and final.step == trainer.state.global_step
    print(
        f"[grpo] done: step {trainer.state.global_step}/{max_steps} in {elapsed / 60:.1f} min; "
        f"final eval step {final.step if final else None}; grader errors {GraderErrors.count}",
        flush=True,
    )
    return 0 if ok else 1


def _eval_weight_saver(trainer: Any, tokenizer: Any, dtype_name: str):
    """`save_eval_weights(path)`: config + safetensors + tokenizer.

    ZeRO-3: `Trainer.save_model` on every rank (DeepSpeed gathers 16-bit weights; the main rank
    writes). One process: the unwrapped model's state dict cast to `dtype_name`, with clean
    (`model.layers.*`) names, which vLLM and `from_pretrained` load directly.
    """
    import torch

    dtype = getattr(torch, dtype_name)

    def save(path: Path) -> None:
        if trainer.is_deepspeed_enabled:
            trainer.save_model(str(path))
            return
        if not trainer.accelerator.is_main_process:
            return
        model = trainer.accelerator.unwrap_model(trainer.model)
        state = {k: (v.detach().to(dtype) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path), state_dict=state, save_original_format=False)
        tokenizer.save_pretrained(str(path))
        del state

    return save


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, args.size, args.variant)
        layout = resolve_layout(cfg, args)
        check_context(cfg, layout.vllm_max_model_length)
        if args.resume_from is not None:
            check_resume_dir(args.resume_from)
    except ConfigError as exc:
        print(f"[grpo] config error: {exc}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    if args.worker:
        return run_worker(args, cfg, layout)
    return launch(args, [a for a in argv if a != "--worker"], cfg, layout)


if __name__ == "__main__":
    sys.exit(main())
