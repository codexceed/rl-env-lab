"""Toy GRPO on two-number addition (PROGRESSION §0.4): does reward rise at this learning rate?

Full fine-tune with TRL `GRPOTrainer`: 8 prompts x 8 generations per step, beta 0, bf16 autocast
over fp32 master weights, gradient checkpointing, thinking disabled through the chat template.
Reward is 1 when the completion is exactly the sum, else 0.

    uv run python train/smoke_toy.py --model Qwen/Qwen3.5-2B --lr 3e-6 --steps 100
    uv run python train/smoke_toy.py --model Qwen/Qwen3.5-2B --lr 3e-6 --steps 100 --hf-generate

Writes to `runs/toy/<model>-lr<lr>/`:
    metrics.jsonl   one row per step: reward, dead_group_share, entropy, completion_length,
                    wall_step_time (+ TRL step_time, loss, grad_norm, memory)
    summary.json    verdict: reward rose (last 20 steps > first 20), no entropy collapse,
                    weights changed, step-time and memory summary

Exit code: 0 when the run completed with changed weights and no NaN reward (whether or not the
reward rose: a flat LR is a result, not a crash); 1 otherwise. Fast hybrid-attention kernels are
required unless `--allow-slow-kernels` is given (see train/compat_check.py for the
causal-conv1d build).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import common
from train.toy_utils import (
    addition_reward,
    grpo_batch_layout,
    make_addition_dataset,
    read_jsonl,
    summarize,
    toy_run_name,
    toy_verdict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy GRPO (addition) for an LR sweep")
    p.add_argument("--model", required=True, help="Hub id, e.g. Qwen/Qwen3.5-2B")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    p.add_argument("--hf-generate", action="store_true", help="fallback: HF generate instead of vLLM")
    p.add_argument("--vllm-server-url", default="http://localhost:8000")
    p.add_argument("--vllm-group-port", type=int, default=51216)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=32)
    p.add_argument("--digits", type=int, default=11, help="operand size; base 2B accuracy ~0.69 at 11 (0.92 at 5)")
    p.add_argument("--lr-scheduler", default="constant")
    p.add_argument("--param-dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=20, help="steps compared for 'reward rose'")
    p.add_argument("--allow-slow-kernels", action="store_true")
    p.add_argument("--out-root", type=Path, default=common.RUNS_DIR / "toy")
    return p.parse_args()


def hints(verdict: dict, weights: dict, args: argparse.Namespace) -> list[str]:
    """Plain-language next steps for the common failure shapes."""
    out = []
    dead = verdict.get("dead_group_share_last")
    first = verdict["reward_rose"].get("first_mean")
    if not weights["changed"]:
        out.append("Weights did not change: every prompt group had equal rewards (no GRPO signal) or LR is 0.")
    if first is not None and first < 0.05 and dead is not None and dead > 0.8:
        out.append(f"Reward starts near 0 with mostly dead groups: try --digits {max(1, args.digits - 2)}.")
    if first is not None and first > 0.9:
        out.append(f"Reward starts near 1 (saturated): try --digits {args.digits + 2}.")
    if verdict["entropy"]["collapsed"]:
        out.append("Entropy collapsed: this LR is too high for the sweep rule.")
    return out


def main() -> int:
    args = parse_args()
    common.setup_torch()
    export_dir = common.ensure_export(args.model)  # before any DeepSpeed / process-group init
    kernels = common.require_fast_kernels(args.allow_slow_kernels)

    from datasets import Dataset

    run_dir = args.out_root / toy_run_name(args.model, args.lr)
    metrics_path = run_dir / "metrics.jsonl"
    if common.is_main_rank():
        metrics_path.unlink(missing_ok=True)
        shutil.rmtree(run_dir / "trainer", ignore_errors=True)

    vllm_mode = "none" if args.hf_generate else args.vllm_mode
    layout = grpo_batch_layout(args.prompts_per_step, args.num_generations, common.world_size())
    config = common.grpo_config(
        run_dir / "trainer",
        lr=args.lr,
        max_steps=args.steps,
        layout=layout,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        vllm_mode=vllm_mode,
        vllm_server_url=args.vllm_server_url,
        vllm_group_port=args.vllm_group_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=512,
        seed=args.seed,
        lr_scheduler_type=args.lr_scheduler,
    )
    tokenizer = common.load_tokenizer(export_dir)
    model = common.load_policy(export_dir, args.param_dtype)

    rows = make_addition_dataset(args.prompts_per_step * args.steps + 64, seed=args.seed, digits=args.digits)
    dataset = Dataset.from_list(rows)
    recorder = common.RewardRecorder(addition_reward)
    callback = common.step_metrics_callback(metrics_path, recorder, args.num_generations)

    probe_names = [n for n, _ in model.named_parameters()]
    probe_names = [probe_names[0], probe_names[len(probe_names) // 2], probe_names[-1]]
    before = common.param_checksums(model)
    snapshot = common.snapshot_params(model, probe_names)

    trainer_cls = common.grpo_trainer_class(hf_generate=args.hf_generate)
    trainer = trainer_cls(
        model=model,
        reward_funcs=recorder,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    t0 = time.perf_counter()
    trainer.train()
    total_s = time.perf_counter() - t0

    if not trainer.accelerator.is_main_process:
        return 0

    trained = trainer.accelerator.unwrap_model(trainer.model)
    weights = common.compare_checksums(before, common.param_checksums(trained))
    weights["relative_change"] = common.relative_change(trained, snapshot)

    records = read_jsonl(metrics_path) if metrics_path.exists() else []
    verdict = toy_verdict(records, window=args.window)
    summary = {
        "model": args.model,
        "export_dir": str(export_dir),
        "lr": args.lr,
        "steps": args.steps,
        "backend": "hf-generate" if args.hf_generate else f"vllm-{args.vllm_mode}",
        "layout": layout,
        "digits": args.digits,
        "param_dtype": args.param_dtype,
        "slow_kernels": kernels["slow_fallback"],
        "verdict": verdict,
        "reward_rose": verdict["reward_rose"]["rose"],
        "entropy_collapsed": verdict["entropy"]["collapsed"],
        "passed": verdict["passed"] and weights["changed"],
        "weights": weights,
        "wall_step_time_s": summarize([r.get("wall_step_time") for r in records]),
        "trl_step_time_s": summarize([r.get("trl_step_time") for r in records]),
        "peak_allocated_gb": max((r.get("peak_allocated_gb") or 0.0 for r in records), default=None),
        "device_used_gb": max((r.get("device_used_gb") or 0.0 for r in records), default=None),
        "total_seconds": round(total_s, 1),
        "versions": common.package_versions(),
        "gpu": common.gpu_info(),
    }
    summary["hints"] = hints(verdict, weights, args)
    common.write_json(run_dir / "summary.json", summary)
    common.print_json(
        "toy summary",
        {
            k: summary[k]
            for k in ("model", "lr", "backend", "reward_rose", "entropy_collapsed", "passed", "wall_step_time_s")
        }
        | {
            "reward_first_last": [verdict["reward_rose"].get("first_mean"), verdict["reward_rose"].get("last_mean")],
            "weights_changed": weights["changed"],
        },
    )
    print(f"\nmetrics: {metrics_path}\nsummary: {run_dir / 'summary.json'}")
    return 0 if weights["changed"] and not verdict["nan_reward"] else 1


if __name__ == "__main__":
    sys.exit(main())
