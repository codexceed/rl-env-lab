"""Shared GPU-side helpers for `compat_check.py`, `smoke_toy.py` and `bench.py`.

Needs the `gpu` extra (torch, transformers, trl, vllm). Tests never import this module.

Key facts verified against the locked stack (transformers 5.17, TRL 1.15, vLLM 0.31):

- The Qwen3.5 Hub repos are vision-language checkpoints (`Qwen3_5ForConditionalGeneration`).
  `AutoModelForCausalLM` maps them to the text-only `Qwen3_5ForCausalLM` and drops the vision
  tower. TRL's `GRPOTrainer(model="<id>")` would instead build the VLM class from
  `config.architectures`, so we always pass a model instance plus a plain tokenizer.
- We export a clean text-only copy (`runs/models/<name>-text`, architecture `Qwen3_5ForCausalLM`,
  model type `qwen3_5_text`) and train and serve from it, because:
  * TRL's colocated vLLM is built from `model.name_or_path`; pointing vLLM at the Hub id would
    load the VLM, whose weight names do not match the text-only trainer during weight sync.
  * Trainer checkpoints of a model loaded straight from the Hub id are re-saved in the original
    `model.language_model.*` layout, and `Trainer` resume (`load_state_dict(strict=False)`) then
    silently loads nothing. Loading from the clean export avoids the renaming entirely.
- transformers picks the fast hybrid-attention kernels when `modeling_qwen3_5` is first imported:
  `fla` (flash-linear-attention) for the gated delta rule and `causal_conv1d` for the short
  convolution. If either is missing it silently uses a slow PyTorch reference path, logging one
  warning. `kernel_report()` inspects the resolved functions so we never accept that silently.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import pickle
import platform
import random
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train.toy_utils import append_jsonl, dead_group_share, metrics_record, model_slug

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
MODELS_DIR = RUNS_DIR / "models"

GB = 1024**3

CAUSAL_CONV1D_INSTALL = (
    "uv pip install setuptools wheel ninja packaging && "
    "CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 "
    "uv pip install --no-build-isolation --no-deps 'causal-conv1d==1.7.0'"
)

# Module-level functions in transformers' modeling_qwen3_5 and the package that accelerates each.
KERNEL_FUNCS = {
    "causal_conv1d_fn": "causal_conv1d",  # training / prefill short convolution
    "causal_conv1d_update": "causal_conv1d",  # cached decode (HF generate)
    "torch_chunk_gated_delta_rule": "fla",  # training / prefill linear attention
    "torch_recurrent_gated_delta_rule": "fla",  # cached decode (HF generate)
}
QWEN35_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
FALLBACK_WARNING = "falling back to its reference PyTorch implementation"

VERSIONED_PACKAGES = [
    "torch",
    "transformers",
    "trl",
    "vllm",
    "deepspeed",
    "accelerate",
    "datasets",
    "flash-linear-attention",
    "fla-core",
    "causal-conv1d",
    "triton",
]


# --------------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------------


def setup_torch() -> None:
    """Global torch switches shared by every script."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_gb": round(torch.cuda.get_device_properties(i).total_memory / GB, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def nvidia_smi() -> list[dict[str, Any]] | None:
    """Device-level memory for every visible GPU (includes other processes, e.g. a vLLM server)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except OSError, subprocess.SubprocessError:
        return None
    rows = []
    for line in out.strip().splitlines():
        idx, name, used, total, util = (x.strip() for x in line.split(","))
        rows.append(
            {
                "index": int(idx),
                "name": name,
                "used_gb": round(float(used) / 1024, 2),
                "total_gb": round(float(total) / 1024, 2),
                "util_pct": float(util),
            }
        )
    return rows


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> dict[str, float]:
    """Peak torch allocations since the last reset, plus current device-level usage."""
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3),
        "device_used_gb": round((total - free) / GB, 3),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


# --------------------------------------------------------------------------------------------
# Text-only export and loading
# --------------------------------------------------------------------------------------------


def is_export(path: str | Path) -> bool:
    return (Path(path) / "export_info.json").exists()


def export_dir_for(model_id: str) -> Path:
    """Where the text-only export of `model_id` lives (an existing export maps to itself)."""
    if is_export(model_id):
        return Path(model_id)
    return MODELS_DIR / f"{model_slug(model_id)}-text"


def export_text_only(model_id: str, force: bool = False) -> Path:
    """Save a clean text-only copy of a Qwen3.5 checkpoint and return its directory.

    `model_id` is a Hub id or a local checkpoint directory. Idempotent: returns the existing
    export when its completion marker is present; an export passed in is returned unchanged.
    """
    out = export_dir_for(model_id)
    if out == Path(model_id) or (is_export(out) and not force):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Another process is exporting the same model: wait for it.
        deadline = time.time() + 3600
        while not is_export(out):
            if time.time() > deadline:
                raise TimeoutError(f"stale export lock {lock}; delete it and retry") from None
            time.sleep(5)
        return out
    try:
        _export(model_id, out)
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)
    return out


def _export(model_id: str, out: Path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    source_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=None)
    text_only_check(model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # save_original_format=False: keep clean `model.layers.*` names instead of re-adding the
    # VLM's `model.language_model.*` prefix.
    model.save_pretrained(tmp, save_original_format=False)
    tokenizer.save_pretrained(tmp)
    info = {
        "source": model_id,
        "source_architectures": list(source_config.architectures or []),
        "export_architectures": list(model.config.architectures or []),
        "export_model_type": model.config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "versions": package_versions(),
        "created_unix": time.time(),
    }
    (tmp / "export_info.json").write_text(json.dumps(info, indent=2) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    del model


def ensure_export(model_id: str) -> Path:
    """Export (or wait for another process exporting) before any process group or DeepSpeed
    ZeRO-3 init exists: under ZeRO-3 init, `from_pretrained` would shard the weights."""
    return export_text_only(model_id)


def text_only_check(model: torch.nn.Module) -> dict[str, Any]:
    """Raise unless `model` is a text-only causal LM with no vision tower."""
    vision = [n for n, _ in model.named_modules() if n.split(".")[-1] in ("visual", "vision_tower")]
    if vision:
        raise RuntimeError(f"model still has vision modules: {vision[:3]}")
    if not type(model).__name__.endswith("ForCausalLM"):
        raise RuntimeError(f"expected a *ForCausalLM class, got {type(model).__name__}")
    cfg = model.config
    layer_types = list(getattr(cfg, "layer_types", []) or [])
    return {
        "class": type(model).__name__,
        "model_type": cfg.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": getattr(cfg, "num_hidden_layers", None),
        "layer_types": dict(Counter(layer_types)),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
    }


def load_policy(export_dir: Path, dtype: str = "float32", device: str | None = None):
    """Load the text-only policy from an export. fp32 keeps master weights exact (bf16 compute
    comes from autocast via `bf16=True`); full-FT updates at lr 1e-6 vanish in pure bf16 weights."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(export_dir), dtype=getattr(torch, dtype), device_map=None)
    text_only_check(model)
    if device is not None:
        model.to(device)
    return model


def load_tokenizer(export_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(export_dir), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --------------------------------------------------------------------------------------------
# Hybrid-attention kernels
# --------------------------------------------------------------------------------------------


def _closure_vars(fn: Callable | None) -> dict[str, Any]:
    """Closure variables of transformers' `use_kernel_func_from_hub_with_fallback` wrapper."""
    for _ in range(6):
        if fn is None:
            break
        if getattr(fn, "__closure__", None):
            nonlocals = dict(inspect.getclosurevars(fn).nonlocals)
            if "implementation" in nonlocals:
                return nonlocals
        fn = getattr(fn, "__wrapped__", None)
    return {}


def kernel_report() -> dict[str, Any]:
    """Which hybrid-attention implementation each Qwen3.5 kernel resolved to.

    `slow_fallback` is True if any function uses the PyTorch reference path or its status cannot
    be determined.
    """
    report: dict[str, Any] = {"packages": {}, "functions": {}}
    for pkg in ("fla", "causal_conv1d"):
        try:
            module = importlib.import_module(pkg)
            report["packages"][pkg] = {
                "importable": True,
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            report["packages"][pkg] = {"importable": False, "error": repr(exc)}

    modeling = importlib.import_module(QWEN35_MODELING)
    for name, pkg in KERNEL_FUNCS.items():
        fn = getattr(modeling, name, None)
        closure = _closure_vars(fn) if fn is not None else {}
        impl = closure.get("implementation")
        fast = closure.get("is_new_implementation")
        report["functions"][name] = {
            "package": pkg,
            "fast": bool(fast) if fast is not None else None,
            "implementation": (
                f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__qualname__', '?')}" if impl is not None else None
            ),
        }
    report["slow_fallback"] = not all(f["fast"] is True for f in report["functions"].values())
    report["vllm_note"] = (
        "vLLM uses its own Triton gated-delta-net and conv kernels; fla/causal-conv1d only "
        "matter for the HF training forward/backward and HF generate."
    )
    return report


def require_fast_kernels(allow_slow: bool) -> dict[str, Any]:
    """Fail loudly on the slow path unless explicitly allowed (and then say so in the output)."""
    report = kernel_report()
    if report["slow_fallback"]:
        slow = [n for n, f in report["functions"].items() if f["fast"] is not True]
        msg = (
            f"Qwen3.5 hybrid-attention kernels fall back to slow PyTorch code for {slow}. "
            f"Packages: {report['packages']}. Install flash-linear-attention (gpu extra) and "
            f"build causal-conv1d: {CAUSAL_CONV1D_INSTALL}"
        )
        if not allow_slow:
            raise RuntimeError(msg + "  (pass --allow-slow-kernels to run anyway)")
        print(f"WARNING: {msg}  Continuing because --allow-slow-kernels was given.", flush=True)
    return report


@contextmanager
def count_kernel_calls() -> Iterator[Counter]:
    """Count calls to each Qwen3.5 kernel function while the context is active."""
    modeling = importlib.import_module(QWEN35_MODELING)
    counts: Counter = Counter()
    originals = {name: getattr(modeling, name) for name in KERNEL_FUNCS}

    def make(name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def counted(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)

        return counted

    for name, fn in originals.items():
        setattr(modeling, name, make(name, fn))
    try:
        yield counts
    finally:
        for name, fn in originals.items():
            setattr(modeling, name, fn)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_fallback_warnings() -> Iterator[list[str]]:
    """Collect transformers' "falling back to its reference PyTorch implementation" warnings."""
    handler = _ListHandler()
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    found: list[str] = []
    try:
        yield found
    finally:
        logger.removeHandler(handler)
        found.extend(m for m in handler.messages if FALLBACK_WARNING in m)


# --------------------------------------------------------------------------------------------
# Weight and state fingerprints
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def param_checksums(model: torch.nn.Module) -> dict[str, tuple[float, float]]:
    """(sum, sum of squares) per parameter in float64: cheap change detection without copies."""
    out = {}
    for name, p in model.named_parameters():
        d = p.detach().double()
        out[name] = (d.sum().item(), d.pow(2).sum().item())
    return out


def compare_checksums(before: dict[str, tuple[float, float]], after: dict[str, tuple[float, float]]) -> dict[str, Any]:
    changed = [n for n in before if n in after and before[n] != after[n]]
    return {
        "params_total": len(before),
        "params_changed": len(changed),
        "example_changed": changed[:5],
        "changed": bool(changed),
    }


@torch.no_grad()
def snapshot_params(model: torch.nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    params = dict(model.named_parameters())
    return {n: params[n].detach().float().cpu().clone() for n in names if n in params}


@torch.no_grad()
def relative_change(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> dict[str, float]:
    params = dict(model.named_parameters())
    out = {}
    for name, before in snapshot.items():
        now = params[name].detach().float().cpu()
        out[name] = float((now - before).norm() / before.norm().clamp_min(1e-12))
    return out


def model_fingerprint(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += p.detach().double().abs().sum().item()
    return total


def optimizer_fingerprint(optimizer: Any) -> dict[str, float]:
    if optimizer is None:
        return {}
    opt = getattr(optimizer, "optimizer", optimizer)  # accelerate wraps the optimizer
    sums: dict[str, float] = {}
    with torch.no_grad():
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    sums[key] = sums.get(key, 0.0) + value.detach().double().abs().sum().item()
    return sums


def rng_fingerprint() -> dict[str, str]:
    def h(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    out = {
        "python": h(pickle.dumps(random.getstate())),
        "numpy": h(pickle.dumps(np.random.get_state())),
        "torch_cpu": h(torch.random.get_rng_state().numpy().tobytes()),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = h(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return out


def dir_size_gb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / GB, 3)


# --------------------------------------------------------------------------------------------
# GRPO plumbing
# --------------------------------------------------------------------------------------------


def grpo_config(
    output_dir: Path,
    *,
    lr: float,
    max_steps: int,
    layout: dict[str, int],
    num_generations: int,
    max_completion_length: int,
    vllm_mode: str,
    vllm_server_url: str | None = None,
    vllm_group_port: int = 51216,
    vllm_gpu_memory_utilization: float = 0.3,
    vllm_max_model_length: int | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    lr_scheduler_type: str = "constant",
    temperature: float = 1.0,
    **overrides: Any,
):
    """GRPOConfig with the project defaults (PROGRESSION: beta 0, bf16, grad checkpointing,
    temp 1.0, thinking disabled through the chat template)."""
    from trl import GRPOConfig

    use_vllm = vllm_mode != "none"
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": lr,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": 0,
        "max_steps": max_steps,
        "per_device_train_batch_size": layout["per_device_train_batch_size"],
        "gradient_accumulation_steps": layout["gradient_accumulation_steps"],
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "beta": 0.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
        "disable_dropout": True,
        "log_completions": False,
        "use_vllm": use_vllm,
    }
    if use_vllm:
        kwargs["vllm_mode"] = vllm_mode
        if vllm_mode == "server":
            kwargs["vllm_server_base_url"] = vllm_server_url or "http://localhost:8000"
            kwargs["vllm_group_port"] = vllm_group_port
        else:
            kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_memory_utilization
            kwargs["vllm_max_model_length"] = vllm_max_model_length
    if generation_kwargs:
        kwargs["generation_kwargs"] = generation_kwargs
    kwargs.update(overrides)
    return GRPOConfig(**kwargs)


class RewardRecorder:
    """Wraps a TRL reward function and keeps the rewards of the current step, so the dead-group
    share can be recomputed independently of TRL's `frac_reward_zero_std` (single process)."""

    def __init__(self, fn: Callable[..., list[float]]):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "reward")
        self.rewards: list[float] = []
        self.group_ids: list[Any] = []

    def __call__(self, completions, **kwargs):
        rewards = self.fn(completions=completions, **kwargs)
        self.rewards.extend(float(r) for r in rewards)
        self.group_ids.extend(kwargs.get("group_id") or range(len(rewards)))
        return rewards

    def pop_dead_group_share(self, group_size: int) -> float | None:
        if not self.rewards:
            return None
        try:
            share = dead_group_share(self.rewards, group_size, group_ids=self.group_ids)
        except ValueError:
            share = None
        self.rewards, self.group_ids = [], []
        return share


def step_metrics_callback(
    metrics_path: Path,
    recorder: RewardRecorder | None = None,
    num_generations: int = 8,
):
    """TrainerCallback writing one metrics.jsonl row per optimizer step (main process only)."""
    from transformers import TrainerCallback

    class StepMetrics(TrainerCallback):
        def __init__(self) -> None:
            self.t0 = None
            self.wall: list[float] = []
            self.peaks: list[dict[str, float]] = []
            self.records: list[dict[str, Any]] = []
            self._last_wall = None
            self._last_peak: dict[str, float] = {}

        def on_step_begin(self, args, state, control, **kwargs):
            reset_peak_memory()
            self.t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            if self.t0 is not None:
                self._last_wall = time.perf_counter() - self.t0
                self.wall.append(self._last_wall)
            self._last_peak = peak_memory()
            self.peaks.append(self._last_peak)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "reward" not in logs:  # final summary log, eval logs, ...
                return
            extra: dict[str, Any] = {"wall_step_time": self._last_wall, **self._last_peak}
            if recorder is not None:
                share = recorder.pop_dead_group_share(num_generations)
                if share is not None and state.is_world_process_zero and args.world_size == 1:
                    extra["dead_group_share_check"] = share
            row = metrics_record(state.global_step, logs, extra)
            self.records.append(row)
            if state.is_world_process_zero:
                append_jsonl(metrics_path, row)

    return StepMetrics()


def print_json(title: str, data: Any) -> None:
    print(f"\n=== {title} ===\n{json.dumps(data, indent=2, default=str)}", flush=True)


def grpo_trainer_class(hf_generate: bool):
    """`GRPOTrainer`, or for the HF-generate fallback a subclass that samples under bf16 autocast.

    TRL 1.15 calls `model.generate` outside autocast, so with fp32 master weights the fallback would
    sample in fp32. `_generate_single_turn` is private TRL API (verified in 1.15.0).
    """
    from trl import GRPOTrainer

    if not hf_generate:
        return GRPOTrainer

    class AutocastGenerateGRPOTrainer(GRPOTrainer):
        def _generate_single_turn(self, *args: Any, **kwargs: Any):
            with torch.autocast(self.accelerator.device.type, dtype=torch.bfloat16):
                return super()._generate_single_turn(*args, **kwargs)

    return AutocastGenerateGRPOTrainer


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))
