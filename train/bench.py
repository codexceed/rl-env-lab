"""Step time, peak memory and eval throughput at real task lengths (PROGRESSION §0.4).

Real lengths: prompt 1,536 tokens, completion 128 tokens (forced), 8 prompts x 8 generations.

    uv run python train/bench.py              # everything below, one job at a time -> runs/bench/
    uv run python train/bench.py train ...    # one GRPO step-time benchmark (plain or under accelerate)
    uv run python train/bench.py eval ...     # vLLM eval throughput on N GPUs (samples/s)

The default run (`make bench`) needs >= 4 visible GPUs and does, sequentially:
    0.8B and 2B   1 GPU each, colocated vLLM
    4B            `trl vllm-serve` on 1 GPU + ZeRO-3 training (configs/accelerate/zero3_<n>gpu.yaml):
                  server-3train: the planned 1+3 layout. 64 completions cannot be split over 3 ranks
                                 (TRL needs completions % (per_device_batch x ranks) == 0), so it runs
                                 9 prompts x 8 = 72 and also reports time scaled to 64.
                  server-2train: 1+2, exactly 64 completions (budget-preserving alternative)
                  colocate-4train (opt-in): 4 ranks with colocated vLLM, exactly 64 completions
    eval          250 prompts x 4 samples, temp 0.7, on 2 GPUs (data parallel), per size
and writes runs/bench/summary.json plus a Markdown table for the PROGRESSION run matrix.

4B by hand (the same commands the default run uses):

    uv run python train/export_text.py --model Qwen/Qwen3.5-4B
    CUDA_VISIBLE_DEVICES=0 uv run trl vllm-serve --model runs/models/Qwen3.5-4B-text \\
        --port 8000 --gpu_memory_utilization 0.9 --max_model_len 2048
    CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file configs/accelerate/zero3_3gpu.yaml \\
        -m train.bench train --model Qwen/Qwen3.5-4B --vllm-mode server \\
        --vllm-server-url http://localhost:8000 --prompts-per-step 9
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import common
from train.toy_utils import (
    grpo_batch_layout,
    hash_reward,
    model_slug,
    smallest_valid_prompts_per_step,
    summarize,
)

SIZES = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B", "4B": "Qwen/Qwen3.5-4B"}
BENCH_DIR = common.RUNS_DIR / "bench"
ACCEL_DIR = common.REPO_ROOT / "configs" / "accelerate"
TARGET_COMPLETIONS = 64
# Concurrent runs per node that one 2-GPU eval pool must keep up with (PROGRESSION GPU allocation).
CONCURRENT_RUNS = {"0.8B": 6, "2B": 6, "4B": 2}
CKPT_EVERY = 50
EVAL_SAMPLES = 250 * 4


# --------------------------------------------------------------------------------------------
# Prompts at real length
# --------------------------------------------------------------------------------------------


def _records_text(rng: random.Random, n_chars: int) -> str:
    statuses = ["open", "paid", "void", "disputed", "partially_paid"]
    lines = []
    size = 0
    while size < n_chars:
        line = json.dumps(
            {
                "invoice_id": f"inv_{rng.randrange(10**5):05d}",
                "customer_id": f"cus_{rng.randrange(10**4):04d}",
                "amount_billed": rng.randrange(10**6),
                "amount_due": rng.randrange(10**6),
                "status": rng.choice(statuses),
            }
        )
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def long_prompts(tokenizer, n: int, target_tokens: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """`n` chat prompts whose templated length (thinking off) is at most, and close to, `target_tokens`."""
    rng = random.Random(seed)

    def templated_len(content: str) -> int:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    head = "Billing records (JSON lines). Find the overcharged invoice.\n"
    overhead = templated_len(head)
    rows, lengths = [], []
    for i in range(n):
        ids = tokenizer(_records_text(rng, target_tokens * 6), add_special_tokens=False)["input_ids"]
        ids = ids[: max(1, target_tokens - overhead)]
        content = head + tokenizer.decode(ids)
        length = templated_len(content)
        for _ in range(8):  # decode -> re-encode can shift a few tokens; trim until it fits
            if length <= target_tokens:
                break
            ids = ids[: len(ids) - (length - target_tokens)]
            content = head + tokenizer.decode(ids)
            length = templated_len(content)
        rows.append({"prompt": [{"role": "user", "content": content}], "group_id": i})
        lengths.append(length)
    return rows, summarize(lengths)


# --------------------------------------------------------------------------------------------
# `train`: one GRPO benchmark (run plainly for 1 GPU, under `accelerate launch` for ZeRO-3)
# --------------------------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    common.setup_torch()
    export_dir = common.ensure_export(args.model)  # before ZeRO-3 init
    kernels = common.require_fast_kernels(args.allow_slow_kernels)
    ranks = common.world_size()
    try:
        layout = grpo_batch_layout(args.prompts_per_step, args.num_generations, ranks)
    except ValueError as exc:
        hint = smallest_valid_prompts_per_step(args.prompts_per_step, args.num_generations, ranks)
        print(f"{exc}\nHint: --prompts-per-step {hint} works on {ranks} ranks.", file=sys.stderr)
        return 2

    from accelerate.utils import gather_object
    from datasets import Dataset

    backend = "hf" if args.vllm_mode == "none" else args.vllm_mode
    tag = args.tag or f"{model_slug(args.model)}-{ranks}train-{backend}"
    out_dir = BENCH_DIR / tag
    metrics_path = out_dir / "metrics.jsonl"
    if common.is_main_rank():
        metrics_path.unlink(missing_ok=True)

    steps = args.warmup_steps + args.measure_steps
    gen_kwargs = (
        {"min_new_tokens": args.completion_tokens}
        if args.vllm_mode == "none"
        else {"min_tokens": args.completion_tokens}
    )
    config = common.grpo_config(
        out_dir / "trainer",
        lr=1e-6,
        max_steps=steps,
        layout=layout,
        num_generations=args.num_generations,
        max_completion_length=args.completion_tokens,
        vllm_mode=args.vllm_mode,
        vllm_server_url=args.vllm_server_url,
        vllm_group_port=args.vllm_group_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.prompt_tokens + args.completion_tokens + 64,
        generation_kwargs=gen_kwargs,
    )
    tokenizer = common.load_tokenizer(export_dir)
    model = common.load_policy(export_dir, args.param_dtype)
    rows, prompt_stats = long_prompts(tokenizer, args.prompts_per_step * steps + 8, args.prompt_tokens, seed=0)
    callback = common.step_metrics_callback(metrics_path, None, args.num_generations)
    trainer = common.grpo_trainer_class(hf_generate=args.vllm_mode == "none")(
        model=model,
        reward_funcs=hash_reward,
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.train()

    peaks = callback.peaks or [{}]
    local = {
        "rank": int(os.environ.get("RANK", "0")),
        "peak_allocated_gb": max(p.get("peak_allocated_gb", 0.0) for p in peaks),
        "peak_reserved_gb": max(p.get("peak_reserved_gb", 0.0) for p in peaks),
        "device_used_gb": max(p.get("device_used_gb", 0.0) for p in peaks),
    }
    per_rank = gather_object([local])
    if not trainer.accelerator.is_main_process:
        return 0

    measured = callback.records[args.warmup_steps :]
    step_time = summarize([r.get("wall_step_time") for r in measured])
    completions = layout["completions_per_step"]
    scale = TARGET_COMPLETIONS / completions
    mean = step_time["mean"]
    result = {
        "kind": "train",
        "tag": tag,
        "model": args.model,
        "backend": backend,
        "vllm_server_url": args.vllm_server_url if args.vllm_mode == "server" else None,
        "train_ranks": ranks,
        "layout": layout,
        "prompts_per_step": args.prompts_per_step,
        "prompt_tokens": prompt_stats,
        "completion_tokens": summarize([r.get("completion_length") for r in measured]),
        "warmup_steps": args.warmup_steps,
        "measured_steps": len(measured),
        "step_time_s": step_time,
        "trl_step_time_s": summarize([r.get("trl_step_time") for r in measured]),
        "step_time_at_64_completions_s": mean * scale if mean else None,
        "est_hours_at_64": {str(n): round(mean * scale * n / 3600, 2) for n in (100, 200, 300)} if mean else None,
        "per_rank_memory": per_rank,
        "nvidia_smi_at_end": common.nvidia_smi(),
        "slow_kernels": kernels["slow_fallback"],
        "param_dtype": args.param_dtype,
        "versions": common.package_versions(),
    }
    common.write_json(BENCH_DIR / f"{tag}.json", result)
    common.print_json(f"bench {tag}", {k: result[k] for k in ("tag", "step_time_s", "est_hours_at_64")})
    return 0


# --------------------------------------------------------------------------------------------
# `eval`: vLLM throughput, one data-parallel worker per GPU
# --------------------------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    export_dir = common.ensure_export(args.model)
    tokenizer = common.load_tokenizer(export_dir)
    rows, prompt_stats = long_prompts(tokenizer, args.prompts, args.prompt_tokens, seed=7)
    prompt_ids = [
        tokenizer(
            tokenizer.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            add_special_tokens=False,
        )["input_ids"]
        for r in rows
    ]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    tag = args.tag or f"eval-{model_slug(args.model)}-{len(gpus)}gpu"
    work = BENCH_DIR / tag
    work.mkdir(parents=True, exist_ok=True)
    go = work / "GO"
    go.unlink(missing_ok=True)

    procs = []
    for i, gpu in enumerate(gpus):
        shard = work / f"shard{i}.json"
        shard.write_text(json.dumps(prompt_ids[i :: len(gpus)]))
        for name in (f"ready{i}", f"result{i}.json"):
            (work / name).unlink(missing_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "train.bench",
            "eval-worker",
            "--export",
            str(export_dir),
            "--shard",
            str(shard),
            "--ready",
            str(work / f"ready{i}"),
            "--go",
            str(go),
            "--result",
            str(work / f"result{i}.json"),
            "--n",
            str(args.samples_per_prompt),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--min-tokens",
            str(0 if args.natural_stop else args.max_tokens),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.prompt_tokens + args.max_tokens + 64),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        log = (work / f"worker{i}.log").open("w")
        procs.append((subprocess.Popen(cmd, env=env, cwd=common.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT), log))

    deadline = time.time() + args.timeout
    try:
        while not all((work / f"ready{i}").exists() for i in range(len(gpus))):
            if any(p.poll() not in (None, 0) for p, _ in procs) or time.time() > deadline:
                raise RuntimeError(f"an eval worker failed to start; see {work}/worker*.log")
            time.sleep(2)
        go.write_text(str(time.time()))
        for p, _ in procs:
            p.wait(timeout=max(1.0, deadline - time.time()))
    finally:
        for p, log in procs:
            if p.poll() is None:
                p.kill()
            log.close()
    if any(p.returncode != 0 for p, _ in procs):
        print(f"eval worker failed; see {work}/worker*.log", file=sys.stderr)
        return 1

    results = [json.loads((work / f"result{i}.json").read_text()) for i in range(len(gpus))]
    start = min(r["start"] for r in results)
    end = max(r["end"] for r in results)
    samples = sum(r["samples"] for r in results)
    rate = samples / (end - start)
    load_s = max(r["load_s"] for r in results)
    result = {
        "kind": "eval",
        "tag": tag,
        "model": args.model,
        "gpus": gpus,
        "mode": "data-parallel (one vLLM engine per GPU)",
        "prompts": args.prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "temperature": args.temperature,
        "prompt_tokens": prompt_stats,
        "max_tokens": args.max_tokens,
        "forced_full_length": not args.natural_stop,
        "samples_per_s": round(rate, 2),
        "generated_tokens_per_s": round(sum(r["tokens"] for r in results) / (end - start), 1),
        "engine_load_s": round(load_s, 1),
        "eval_seconds_per_checkpoint": round(load_s + EVAL_SAMPLES / rate, 1),
        "workers": results,
        "versions": common.package_versions(),
    }
    common.write_json(BENCH_DIR / f"{tag}.json", result)
    common.print_json(f"bench {tag}", {k: result[k] for k in ("samples_per_s", "eval_seconds_per_checkpoint")})
    return 0


def cmd_eval_worker(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    prompts = json.loads(Path(args.shard).read_text())
    t0 = time.perf_counter()
    llm = LLM(
        model=args.export,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=0,
    )
    load_s = time.perf_counter() - t0
    Path(args.ready).write_text("ok")
    deadline = time.time() + 3600
    while not Path(args.go).exists():
        if time.time() > deadline:
            return 1
        time.sleep(0.5)
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        seed=0,
    )
    start = time.time()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sampling_params=params, use_tqdm=False)
    end = time.time()
    result = {
        "start": start,
        "end": end,
        "load_s": load_s,
        "prompts": len(prompts),
        "samples": sum(len(o.outputs) for o in outputs),
        "tokens": sum(len(c.token_ids) for o in outputs for c in o.outputs),
    }
    Path(args.result).write_text(json.dumps(result))
    return 0


# --------------------------------------------------------------------------------------------
# Default: the whole §0.4 suite
# --------------------------------------------------------------------------------------------


def _bin(name: str) -> str:
    path = Path(sys.executable).with_name(name)
    return str(path) if path.exists() else name


def _run_job(name: str, cmd: list[str], gpus: list[str], timeout: float) -> dict[str, Any]:
    log_path = BENCH_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    print(f"\n>>> {name}: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                cmd, env=env, cwd=common.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    status = {"name": name, "exit": code, "seconds": round(time.perf_counter() - t0, 1), "log": str(log_path)}
    print(f"<<< {name}: exit={code} ({status['seconds']}s) log={log_path}", flush=True)
    if code != 0:
        tail = log_path.read_text().splitlines()[-25:]
        print("\n".join(tail), flush=True)
    return status


class VllmServer:
    """`trl vllm-serve` on one GPU, stopped (whole process group) on exit."""

    def __init__(self, export_dir: Path, gpu: str, port: int, timeout: float):
        self.url = f"http://localhost:{port}"
        self.log_path = BENCH_DIR / "logs" / f"vllm-server-{export_dir.name}.log"
        self.cmd = [
            _bin("trl"),
            "vllm-serve",
            "--model",
            str(export_dir),
            "--port",
            str(port),
            "--gpu_memory_utilization",
            "0.9",
            "--max_model_len",
            "2048",
        ]
        self.gpu = gpu
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> VllmServer:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu}
        print(f"\n>>> vLLM server: CUDA_VISIBLE_DEVICES={self.gpu} {' '.join(self.cmd)}", flush=True)
        self.proc = subprocess.Popen(
            self.cmd, env=env, cwd=common.REPO_ROOT, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True
        )
        deadline = time.time() + self.timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited early; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except OSError:
                pass
            if time.time() > deadline:
                raise TimeoutError(f"vLLM server not healthy after {self.timeout}s; see {self.log_path}")
            time.sleep(5)

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self._log.close()


def cmd_all(args: argparse.Namespace) -> int:
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [str(i) for i in range(_gpu_count())]
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    if len(gpus) < 4 and "4B" in sizes and not args.skip_train:
        print(f"need >= 4 GPUs for the 4B layouts, have {gpus}", file=sys.stderr)
        return 2
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for size in sizes:  # export once, up front, on CPU
        common.ensure_export(SIZES[size])

    me = [sys.executable, "-m", "train.bench"]
    common_flags = ["--warmup-steps", str(args.warmup_steps), "--measure-steps", str(args.measure_steps)]
    if args.allow_slow_kernels:
        common_flags.append("--allow-slow-kernels")
    jobs: list[dict[str, Any]] = []
    tags: list[str] = []

    if not args.skip_train:
        for size in (s for s in sizes if s != "4B"):
            tag = f"{model_slug(SIZES[size])}-1gpu-colocate"
            cmd = [*me, "train", "--model", SIZES[size], "--vllm-mode", "colocate", "--tag", tag, *common_flags]
            jobs.append(_run_job(tag, cmd, gpus[:1], args.job_timeout))
            tags.append(tag)
        if "4B" in sizes:
            model = SIZES["4B"]
            export_dir = common.export_dir_for(model)
            for layout in [x.strip() for x in args.layouts_4b.split(",") if x.strip()]:
                tag = f"{model_slug(model)}-{layout}"
                if layout == "colocate-4train":
                    cfg = ACCEL_DIR / "zero3_4gpu.yaml"
                    cmd = [_bin("accelerate"), "launch", "--config_file", str(cfg), "-m", "train.bench", "train"]
                    cmd += ["--model", model, "--vllm-mode", "colocate", "--vllm-gpu-memory-utilization", "0.25"]
                    jobs.append(_run_job(tag, [*cmd, "--tag", tag, *common_flags], gpus[:4], args.job_timeout))
                    tags.append(tag)
                    continue
                n_train = {"server-3train": 3, "server-2train": 2}[layout]
                prompts = smallest_valid_prompts_per_step(8, 8, n_train)
                cfg = ACCEL_DIR / f"zero3_{n_train}gpu.yaml"
                cmd = [_bin("accelerate"), "launch", "--config_file", str(cfg), "-m", "train.bench", "train"]
                cmd += ["--model", model, "--vllm-mode", "server", "--vllm-server-url", f"http://localhost:{args.port}"]
                cmd += ["--vllm-group-port", str(args.group_port), "--prompts-per-step", str(prompts)]
                try:
                    with VllmServer(export_dir, gpus[0], args.port, args.server_timeout):
                        jobs.append(
                            _run_job(tag, [*cmd, "--tag", tag, *common_flags], gpus[1 : 1 + n_train], args.job_timeout)
                        )
                except (RuntimeError, TimeoutError) as exc:
                    jobs.append({"name": tag, "exit": "server-failed", "error": str(exc)})
                tags.append(tag)

    if not args.skip_eval:
        for size in sizes:
            tag = f"eval-{model_slug(SIZES[size])}-2gpu"
            cmd = [*me, "eval", "--model", SIZES[size], "--gpus", ",".join(gpus[:2]), "--tag", tag]
            jobs.append(_run_job(tag, cmd, gpus[:2], args.job_timeout))
            tags.append(tag)

    summary = build_summary(tags, jobs, args.steps_for_estimate)
    common.write_json(BENCH_DIR / "summary.json", summary)
    print("\n" + summary["markdown"])
    print(f"\nsummary: {BENCH_DIR / 'summary.json'}")
    return 0 if all(j.get("exit") == 0 for j in jobs) else 1


def _gpu_count() -> int:
    rows = common.nvidia_smi()
    return len(rows) if rows else 0


def build_summary(tags: list[str], jobs: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    results = {}
    for tag in tags:
        path = BENCH_DIR / f"{tag}.json"
        if path.exists():
            results[tag] = json.loads(path.read_text())
    lines = [
        f"| Bench | Train GPUs | Completions/step | Step time (s) | Step time @64 (s) | Est. h @ {steps} steps "
        "| Peak alloc / GPU (GB) |",
        "|---|---|---|---|---|---|---|",
    ]
    step_by_size: dict[str, float] = {}
    for tag, r in results.items():
        if r["kind"] != "train":
            continue
        mean = r["step_time_s"]["mean"]
        at64 = r["step_time_at_64_completions_s"]
        peak = max(m["peak_allocated_gb"] for m in r["per_rank_memory"])
        hours = round(at64 * steps / 3600, 2) if at64 else None
        lines.append(
            f"| {tag} | {r['train_ranks']} | {r['layout']['completions_per_step']} | {mean:.1f} | {at64:.1f} "
            f"| {hours} | {peak:.1f} |"
        )
        for size, model in SIZES.items():
            if r["model"] == model and at64:
                step_by_size[size] = min(step_by_size.get(size, at64), at64)
    lines += ["", "| Eval (2 GPUs) | Samples/s | Engine load (s) | s per checkpoint (1,000 samples) | Keeps pace? |"]
    lines.append("|---|---|---|---|---|")
    pace = {}
    for r in results.values():
        if r["kind"] != "eval":
            continue
        size = next((s for s, m in SIZES.items() if m == r["model"]), None)
        verdict = "n/a"
        if size in step_by_size:
            needed_per_h = CONCURRENT_RUNS[size] * 3600 / (CKPT_EVERY * step_by_size[size])
            capacity_per_h = 3600 / r["eval_seconds_per_checkpoint"]
            pace[size] = {"needed_ckpts_per_h": needed_per_h, "capacity_ckpts_per_h": capacity_per_h}
            verdict = (
                f"{'yes' if capacity_per_h >= needed_per_h else 'NO'} ({capacity_per_h:.1f} vs {needed_per_h:.1f}/h)"
            )
        lines.append(
            f"| {r['model']} | {r['samples_per_s']} | {r['engine_load_s']} | {r['eval_seconds_per_checkpoint']} "
            f"| {verdict} |"
        )
    return {
        "jobs": jobs,
        "results": results,
        "eval_keeps_pace": pace,
        "assumptions": {
            "checkpoint_every_steps": CKPT_EVERY,
            "concurrent_runs_per_eval_pool": CONCURRENT_RUNS,
            "eval_samples_per_checkpoint": EVAL_SAMPLES,
        },
        "markdown": "\n".join(lines),
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="§0.4 benchmarks at real task lengths")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("all", help="the default: every size and layout, then eval throughput")
    for target in (p, a):
        target.add_argument("--gpus", default=None, help="physical GPU ids, e.g. 0,1,2,3 (default: all)")
        target.add_argument("--sizes", default="0.8B,2B,4B")
        target.add_argument("--layouts-4b", default="server-3train,server-2train")
        target.add_argument("--skip-train", action="store_true")
        target.add_argument("--skip-eval", action="store_true")
        target.add_argument("--warmup-steps", type=int, default=2)
        target.add_argument("--measure-steps", type=int, default=5)
        target.add_argument("--steps-for-estimate", type=int, default=300)
        target.add_argument("--port", type=int, default=8011)
        target.add_argument("--group-port", type=int, default=51311)
        target.add_argument("--server-timeout", type=float, default=900)
        target.add_argument("--job-timeout", type=float, default=3600)
        target.add_argument("--allow-slow-kernels", action="store_true")

    t = sub.add_parser("train", help="one GRPO step-time benchmark")
    t.add_argument("--model", required=True)
    t.add_argument("--vllm-mode", choices=["colocate", "server", "none"], default="colocate")
    t.add_argument("--vllm-server-url", default="http://localhost:8000")
    t.add_argument("--vllm-group-port", type=int, default=51216)
    t.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    t.add_argument("--prompts-per-step", type=int, default=8)
    t.add_argument("--num-generations", type=int, default=8)
    t.add_argument("--prompt-tokens", type=int, default=1536)
    t.add_argument("--completion-tokens", type=int, default=128)
    t.add_argument("--warmup-steps", type=int, default=2)
    t.add_argument("--measure-steps", type=int, default=5)
    t.add_argument("--param-dtype", choices=["float32", "bfloat16"], default="float32")
    t.add_argument("--allow-slow-kernels", action="store_true")
    t.add_argument("--tag", default=None)

    e = sub.add_parser("eval", help="vLLM eval throughput (samples/s), one engine per GPU")
    e.add_argument("--model", required=True)
    e.add_argument("--gpus", default="0,1", help="physical GPU ids")
    e.add_argument("--prompts", type=int, default=250)
    e.add_argument("--samples-per-prompt", type=int, default=4)
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--prompt-tokens", type=int, default=1536)
    e.add_argument("--max-tokens", type=int, default=128)
    e.add_argument("--natural-stop", action="store_true", help="let samples stop at EOS (default: force 128)")
    e.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    e.add_argument("--timeout", type=float, default=3600)
    e.add_argument("--tag", default=None)

    w = sub.add_parser("eval-worker", help=argparse.SUPPRESS)
    for name in ("--export", "--shard", "--ready", "--go", "--result"):
        w.add_argument(name, required=True)
    w.add_argument("--n", type=int, default=4)
    w.add_argument("--temperature", type=float, default=0.7)
    w.add_argument("--max-tokens", type=int, default=128)
    w.add_argument("--min-tokens", type=int, default=128)
    w.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    w.add_argument("--max-model-len", type=int, default=1728)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    handlers = {"train": cmd_train, "eval": cmd_eval, "eval-worker": cmd_eval_worker}
    return handlers.get(args.cmd, cmd_all)(args)


if __name__ == "__main__":
    sys.exit(main())
