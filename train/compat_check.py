"""Qwen3.5 compatibility check for the full training path (PROGRESSION §0.3).

Runs every §0.3 check on one GPU and writes a pass/fail JSON report to
`runs/compat/<model>.json`. Exit code 0 only if every required check passed.

    uv run python -m train.compat_check --model Qwen/Qwen3.5-2B                     # colocated vLLM
    uv run python -m train.compat_check --model Qwen/Qwen3.5-2B --vllm-mode none    # HF only
    # server mode: start `trl vllm-serve` on another GPU first (see train/README.md)
    CUDA_VISIBLE_DEVICES=1 uv run python -m train.compat_check --model Qwen/Qwen3.5-4B \\
        --vllm-mode server --vllm-server-url http://localhost:8000

Checks
    load               text-only Qwen3_5ForCausalLM from a clean export, no vision tower
    thinking_disabled  chat template with enable_thinking=False opens an empty <think></think>
    kernels            fla + causal-conv1d resolved and actually called; slow_fallback flagged
    forward_backward   bf16 autocast + gradient checkpointing, finite non-zero grads, tokens/s
    hf_generate        HF generate works (training/eval fallback), tokens/s
    tool_call_format   native tool-call format rendered by the template, plus a model attempt
    save_resume        TRL GRPOTrainer checkpoint -> resume restores model, optimizer,
                       scheduler, RNG, trainer state; loss/reward continue
    vllm_generate      vLLM generation through TRL's VLLMGeneration (colocate or server)
    weight_sync        TRL's sync_weights(): a tiny update changes vLLM's greedy output

Fast kernels: `flash-linear-attention` comes with the gpu extra. `causal-conv1d` has no PyPI
wheels and must be built on the node (needs a CUDA 13.x nvcc matching torch 2.13+cu130):

    uv pip install setuptools wheel ninja packaging
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 \\
        uv pip install --no-build-isolation --no-deps 'causal-conv1d==1.7.0'

`uv sync` removes packages that are not in the lock, so rerun that build after any `uv sync`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from train import common  # noqa: E402
from train.toy_utils import (  # noqa: E402
    extract_answer,
    make_addition_dataset,
    model_slug,
)

THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINK_ON_SUFFIX = "<|im_start|>assistant\n<think>\n"

SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_credit",
        "description": "Apply a credit to an invoice and post a ledger entry.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "Invoice id, e.g. inv_001"},
                "amount": {"type": "integer", "description": "Credit in integer cents"},
                "reason": {"type": "string"},
            },
            "required": ["invoice_id", "amount", "reason"],
        },
    },
}
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


# --------------------------------------------------------------------------------------------
# Report plumbing
# --------------------------------------------------------------------------------------------


class Report:
    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = path
        self.data: dict[str, Any] = {"meta": meta, "checks": {}}

    def run(self, name: str, fn: Callable[[], dict[str, Any]], required: bool = True) -> dict:
        print(f"\n>>> check: {name}", flush=True)
        t0 = time.perf_counter()
        try:
            details = fn()
            passed = bool(details.pop("passed"))
            entry = {"passed": passed, "required": required, "details": details}
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, then we move on
            entry = {
                "passed": False,
                "required": required,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        self.data["checks"][name] = entry
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"<<< {name}: {status} ({entry['seconds']}s)", flush=True)
        if not entry["passed"]:
            print(entry.get("error") or json.dumps(entry["details"], default=str)[:2000])
        self.save()
        return entry

    def skip(self, name: str, reason: str) -> None:
        self.data["checks"][name] = {"passed": None, "required": False, "skipped": reason}
        self.save()

    def finish(self) -> bool:
        checks = self.data["checks"]
        failed = [n for n, c in checks.items() if c.get("required") and not c.get("passed")]
        self.data["summary"] = {
            "all_required_passed": not failed,
            "failed": failed,
            "skipped": [n for n, c in checks.items() if "skipped" in c],
        }
        self.save()
        return not failed

    def save(self) -> None:
        common.write_json(self.path, self.data)


def chat_text(tokenizer, messages, *, tools=None, add_generation_prompt=True, **template_kwargs):
    # transformers 5 returns a dict by default when tokenize=True, so render text explicitly.
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **template_kwargs,
    )


def encode(tokenizer, texts: list[str], device: str = "cuda"):
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in batch.items()}


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def check_load(model, export_dir: Path) -> dict[str, Any]:
    info = common.text_only_check(model)
    export_info = json.loads((export_dir / "export_info.json").read_text())
    info.update(
        source_architectures=export_info.get("source_architectures"),
        export_architectures=export_info.get("export_architectures"),
        param_dtype=str(next(model.parameters()).dtype),
        passed=info["class"] == "Qwen3_5ForCausalLM",
    )
    return info


def check_thinking(tokenizer) -> dict[str, Any]:
    messages = [{"role": "user", "content": "What is 2 + 3?"}]
    off = chat_text(tokenizer, messages, enable_thinking=False)
    default = chat_text(tokenizer, messages)
    on = chat_text(tokenizer, messages, enable_thinking=True)
    if default.endswith(THINK_OFF_SUFFIX):
        default_mode = "non-thinking"
    elif default.endswith(THINK_ON_SUFFIX):
        default_mode = "thinking"
    else:
        default_mode = "unknown"
    return {
        "passed": off.endswith(THINK_OFF_SUFFIX) and on.endswith(THINK_ON_SUFFIX),
        "how": "apply_chat_template(..., enable_thinking=False); TRL: "
        "GRPOConfig(chat_template_kwargs={'enable_thinking': False})",
        "template_default_mode": default_mode,
        "generation_prompt_off": off[-60:],
        "generation_prompt_on": on[-60:],
    }


def check_kernels_static() -> dict[str, Any]:
    report = common.kernel_report()
    report["passed"] = not report["slow_fallback"]
    return report


def check_forward_backward(model, tokenizer, batch_size: int, seq_len: int) -> dict[str, Any]:
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Realistic chat-formatted text, half the rows shorter so left padding is exercised.
    rows = make_addition_dataset(4096, seed=1, digits=6)
    filler = " ".join(f"{r['a']} + {r['b']} = {r['answer']}." for r in rows)
    texts = []
    for i in range(batch_size):
        words = filler[i * 7 :]
        msg = [{"role": "user", "content": words}]
        texts.append(chat_text(tokenizer, msg, enable_thinking=False))
    full = tokenizer(texts, add_special_tokens=False)["input_ids"]
    ids = []
    for i, row in enumerate(full):
        length = seq_len if i % 2 == 0 else seq_len // 2
        if len(row) < length:
            raise RuntimeError(f"filler too short: {len(row)} < {length}")
        ids.append(row[:length])
    pad = tokenizer.pad_token_id
    input_ids = torch.full((batch_size, seq_len), pad, dtype=torch.long)
    attention = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i, row in enumerate(ids):
        input_ids[i, seq_len - len(row) :] = torch.tensor(row)
        attention[i, seq_len - len(row) :] = 1
    labels = input_ids.masked_fill(attention == 0, -100)
    input_ids, attention, labels = (t.cuda() for t in (input_ids, attention, labels))
    real_tokens = int(attention.sum())

    probes = {}
    for name, _ in model.named_parameters():
        if "linear_attn.in_proj_qkv.weight" in name and "gdn_qkv" not in probes:
            probes["gdn_qkv"] = name
        elif "linear_attn.conv1d.weight" in name and "gdn_conv" not in probes:
            probes["gdn_conv"] = name
        elif name.endswith("linear_attn.A_log") and "gdn_A_log" not in probes:
            probes["gdn_A_log"] = name
        elif "self_attn.q_proj.weight" in name and "attn_q" not in probes:
            probes["attn_q"] = name
    params = dict(model.named_parameters())

    def step() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
        out.loss.backward()
        return out.loss.detach()

    common.reset_peak_memory()
    with common.count_kernel_calls() as calls, common.capture_fallback_warnings() as warns:
        loss = step()  # warm-up (Triton autotuning) and gradient inspection
        grad_norms = {
            k: float(params[n].grad.float().norm()) if params[n].grad is not None else None
            for k, n in probes.items()
        }
        model.zero_grad(set_to_none=True)
        iters = 3
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses = []
        for _ in range(iters):
            losses.append(float(step()))
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    mem = common.peak_memory()
    grads_ok = all(g is not None and math.isfinite(g) and g > 0 for g in grad_norms.values())
    used = {k: calls.get(k, 0) for k in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")}
    return {
        "passed": math.isfinite(float(loss))
        and all(math.isfinite(x) for x in losses)
        and grads_ok
        and len(probes) == 4
        and all(v > 0 for v in used.values()),
        "precision": "fp32 master weights + bf16 autocast (what Trainer bf16=True does)",
        "gradient_checkpointing": model.is_gradient_checkpointing,
        "batch": [batch_size, seq_len],
        "first_loss": float(loss),
        "losses": losses,
        "grad_norms": grad_norms,
        "kernel_calls": dict(calls),
        "fallback_warnings": warns,
        "train_tokens_per_s": round(real_tokens * iters / elapsed, 1),
        "seconds_per_step": round(elapsed / iters, 3),
        **mem,
    }


def greedy(model, tokenizer, texts: list[str], max_new_tokens: int, **gen) -> list[list[int]]:
    batch = encode(tokenizer, texts)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=gen.pop("do_sample", False),
            pad_token_id=tokenizer.pad_token_id,
            **gen,
        )
    return [row.tolist() for row in out[:, batch["input_ids"].shape[1] :]]


def strip_pad(ids: list[int], tokenizer) -> list[int]:
    return [t for t in ids if t != tokenizer.pad_token_id]


def check_hf_generate(model, tokenizer) -> dict[str, Any]:
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    rows = make_addition_dataset(8, seed=3, digits=3)
    texts = [chat_text(tokenizer, r["prompt"], enable_thinking=False) for r in rows]
    with common.count_kernel_calls() as calls, common.capture_fallback_warnings() as warns:
        outs = greedy(model, tokenizer, texts, max_new_tokens=32)
        decoded = [tokenizer.decode(strip_pad(o, tokenizer)) for o in outs]
        answers = [extract_answer(d.replace("<|im_end|>", ""), strict=False) for d in decoded]
        correct = sum(int(a == int(r["answer"])) for a, r in zip(answers, rows, strict=True))

        # Throughput at task completion length: 16 prompts x 128 forced new tokens, sampled.
        big = make_addition_dataset(16, seed=4, digits=4)
        texts = [chat_text(tokenizer, r["prompt"], enable_thinking=False) for r in big]
        common.reset_peak_memory()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = greedy(
            model, tokenizer, texts, max_new_tokens=128, min_new_tokens=128, do_sample=True
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    new_tokens = sum(len(o) for o in outs)
    leaked_think = [d for d in decoded if "<think>" in d]
    return {
        "passed": all(decoded) and not leaked_think,
        "samples": decoded[:4],
        "addition_correct": f"{correct}/{len(rows)}",
        "kernel_calls": dict(calls),
        "fallback_warnings": warns,
        "generate_tokens_per_s": round(new_tokens / elapsed, 1),
        "generate_batch": [len(texts), 128],
        "note": "bf16 autocast over fp32 weights; train/smoke_toy.py --hf-generate does the same",
        **common.peak_memory(),
    }


def check_tool_call(model, tokenizer) -> dict[str, Any]:
    call = {"invoice_id": "inv_001", "amount": 1500, "reason": "overcharge"}
    conversation = [
        {"role": "system", "content": "You are a billing assistant."},
        {"role": "user", "content": "Invoice inv_001 was overcharged by 1500 cents. Fix it."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "apply_credit", "arguments": call}}
            ],
        },
    ]
    rendered = chat_text(
        tokenizer,
        conversation,
        tools=[SAMPLE_TOOL],
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assistant_turn = rendered.rsplit("<|im_start|>assistant\n", 1)[-1]
    system_block = rendered.split("<|im_end|>", 1)[0]
    print("\n--- rendered assistant tool-call turn ---\n" + assistant_turn, flush=True)

    prompt = chat_text(
        tokenizer,
        conversation[:2],
        tools=[SAMPLE_TOOL],
        enable_thinking=False,
    )
    out = greedy(model, tokenizer, [prompt], max_new_tokens=128)[0]
    text = tokenizer.decode(strip_pad(out, tokenizer))
    print("--- model output ---\n" + text, flush=True)
    match = TOOL_CALL_RE.search(text)
    parsed = None
    if match:
        parsed = {"name": match.group(1), "arguments": dict(PARAM_RE.findall(match.group(2)))}
    expected_head = (
        "<tool_call>\n<function=apply_credit>\n<parameter=invoice_id>\ninv_001\n</parameter>"
    )
    return {
        "passed": expected_head in assistant_turn,
        "format": "<tool_call>\\n<function=NAME>\\n<parameter=ARG>\\nVALUE\\n</parameter>\\n..."
        "</function>\\n</tool_call>  (XML-style, NOT <tool_call>{json}</tool_call>)",
        "rendered_assistant_turn": assistant_turn,
        "system_tools_block_excerpt": system_block[-900:],
        "notes": [
            "Non-string argument values are JSON-encoded; strings are inserted raw.",
            "Tool call arguments must be a dict (the template iterates arguments|items).",
            "The final assistant turn is prefixed with an empty <think></think> block.",
        ],
        "model_output": text,
        "model_called_tool": parsed is not None,
        "model_parsed_call": parsed,
    }


def token_hash_reward(completion_ids: list[list[int]], **kwargs: Any) -> list[float]:
    """Deterministic reward that varies within a group, so advantages and gradients are non-zero
    whatever the model writes. Only the checkpoint mechanics are under test here."""
    return [(sum(ids) % 17) / 16 for ids in completion_ids]


def check_save_resume(export_dir: Path, tokenizer, workdir: Path, keep: bool) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOTrainer

    shutil.rmtree(workdir, ignore_errors=True)
    dataset = Dataset.from_list(make_addition_dataset(64, seed=0, digits=3))
    layout = {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 2}
    total_steps, save_at = 6, 3

    class Probe(TrainerCallback):
        def __init__(self) -> None:
            self.at_start: float | None = None
            self.at_resume: dict[str, Any] = {}
            self.logs: dict[int, dict[str, float]] = {}

        def on_step_begin(self, args, state, control, model=None, optimizer=None, **kw):
            if state.global_step == 0 and self.at_start is None:
                self.at_start = common.model_fingerprint(model)
            if state.global_step == save_at and not self.at_resume:
                sched = kw.get("lr_scheduler")
                self.at_resume = {
                    "global_step": state.global_step,
                    "model": common.model_fingerprint(model),
                    "optimizer": common.optimizer_fingerprint(optimizer),
                    "lr": sched.get_last_lr()[0] if sched is not None else None,
                    "scheduler_last_epoch": getattr(sched, "last_epoch", None),
                    "rng": common.rng_fingerprint(),
                }

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.logs[state.global_step] = {
                    k: float(logs[k]) for k in ("loss", "reward", "entropy") if k in logs
                }

    def make(out: Path, probe: Probe):
        args = common.grpo_config(
            out,
            lr=1e-5,
            max_steps=total_steps,
            layout=layout,
            num_generations=4,
            max_completion_length=16,
            vllm_mode="none",
            lr_scheduler_type="linear",
            save_strategy="steps",
            save_steps=save_at,
            save_only_model=False,
        )
        model = common.load_policy(export_dir, "float32")
        return GRPOTrainer(
            model=model,
            reward_funcs=token_hash_reward,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[probe],
        )

    probe_a = Probe()
    trainer = make(workdir / "A", probe_a)
    trainer.train()
    ckpt = workdir / "A" / f"checkpoint-{save_at}"
    ckpt_files = sorted(p.name for p in ckpt.iterdir())
    ckpt_gb = common.dir_size_gb(ckpt)
    del trainer
    free_cuda()

    probe_b = Probe()
    trainer = make(workdir / "B", probe_b)
    trainer.train(resume_from_checkpoint=str(ckpt))
    b_state = trainer.state
    resumed_history = [e.get("step") for e in b_state.log_history if "loss" in e]
    del trainer
    free_cuda()

    a, b = probe_a.at_resume, probe_b.at_resume

    def close(x, y, rel=1e-7):
        return x is not None and y is not None and abs(x - y) <= rel * max(1.0, abs(x), abs(y))

    restored = {
        "model": close(a.get("model"), b.get("model")),
        "optimizer": bool(a.get("optimizer"))
        and a.get("optimizer", {}).keys() == b.get("optimizer", {}).keys()
        and all(close(a["optimizer"][k], b["optimizer"][k]) for k in a["optimizer"]),
        "scheduler": close(a.get("lr"), b.get("lr"))
        and a.get("scheduler_last_epoch") == b.get("scheduler_last_epoch"),
        "rng": bool(a.get("rng")) and a.get("rng") == b.get("rng"),
        # Guard against a vacuous test: the weights must have moved before the checkpoint.
        "weights_trained_before_save": probe_a.at_start is not None
        and not close(probe_a.at_start, a.get("model"), rel=1e-12),
        "trainer_state": b.get("global_step") == save_at
        and b_state.global_step == total_steps
        and all(s in resumed_history for s in range(1, save_at + 1)),
    }
    after = range(save_at + 1, total_steps + 1)
    diffs = {
        k: [
            abs(probe_a.logs.get(s, {}).get(k, math.nan) - probe_b.logs.get(s, {}).get(k, math.nan))
            for s in after
        ]
        for k in ("loss", "reward")
    }
    finite_b = all(
        math.isfinite(probe_b.logs.get(s, {}).get("loss", math.nan))
        and math.isfinite(probe_b.logs.get(s, {}).get("reward", math.nan))
        for s in after
    )
    first_reward_diff = diffs["reward"][0]
    continuity = finite_b and math.isfinite(first_reward_diff) and first_reward_diff <= 0.13
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "passed": all(restored.values()) and continuity,
        "restored": restored,
        "loss_continues": continuity,
        "logs_uninterrupted": {s: probe_a.logs.get(s) for s in range(1, total_steps + 1)},
        "logs_resumed": {s: probe_b.logs.get(s) for s in after},
        "abs_diff_after_resume": diffs,
        "checkpoint_files": ckpt_files,
        "checkpoint_size_gb": ckpt_gb,
        "fingerprints_uninterrupted": a,
        "fingerprints_resumed": b,
        "note": "Same RNG + weights => step-4 sampling should match; later steps may drift "
        "slightly because some backward kernels are non-deterministic.",
    }


def make_vllm(model, tokenizer, args):
    from accelerate import Accelerator
    from trl.generation.vllm_generation import VLLMGeneration

    return VLLMGeneration(
        model=model,
        accelerator=Accelerator(mixed_precision="bf16"),
        processing_class=tokenizer,
        mode=args.vllm_mode,
        server_base_url=args.vllm_server_url,
        group_port=args.vllm_group_port,
        server_timeout=args.vllm_server_timeout,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_length=2048,
        temperature=1.0,
        max_completion_length=128,
    )


def vllm_gen(vg, prompt_ids: list[list[int]], n: int, temperature: float, max_tokens: int, **kw):
    vg.temperature = temperature
    vg.max_completion_length = max_tokens
    vg.generation_kwargs = kw
    expanded = [p for p in prompt_ids for _ in range(n)]
    _, completion_ids, _, _ = vg.generate(prompts=expanded, images=None, num_generations=n)
    vg.generation_kwargs = {}
    return [list(c) for c in completion_ids]


def check_vllm_generate(vg, tokenizer) -> dict[str, Any]:
    rows = make_addition_dataset(8, seed=3, digits=3)
    texts = [chat_text(tokenizer, r["prompt"], enable_thinking=False) for r in rows]
    prompts = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    outs = vllm_gen(vg, prompts, n=1, temperature=0.0, max_tokens=32)
    decoded = [tokenizer.decode(o) for o in outs]
    correct = sum(
        int(extract_answer(d.replace("<|im_end|>", ""), strict=False) == int(r["answer"]))
        for d, r in zip(decoded, rows, strict=True)
    )
    # Throughput: 8 prompts x 8 generations x 128 forced tokens (the GRPO step shape).
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = vllm_gen(vg, prompts, n=8, temperature=1.0, max_tokens=128, min_tokens=128)
    elapsed = time.perf_counter() - t0
    arch = None
    try:
        arch = vg.llm.llm_engine.model_config.architectures  # colocate only
    except AttributeError:
        pass
    return {
        "passed": all(decoded) and len(outs) == 64,
        "mode": vg.mode,
        "vllm_architectures": arch,
        "samples": decoded[:4],
        "addition_correct": f"{correct}/{len(rows)}",
        "generate_tokens_per_s": round(sum(len(o) for o in outs) / elapsed, 1),
        "completions": len(outs),
        **common.peak_memory(),
    }


def check_weight_sync(vg, model, tokenizer) -> dict[str, Any]:
    target = "PINEAPPLE PINEAPPLE PINEAPPLE"
    messages = [{"role": "user", "content": "Say one word."}]
    prompt = chat_text(tokenizer, messages, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    before = vllm_gen(vg, [prompt_ids], n=1, temperature=0.0, max_tokens=12)[0]

    # Tiny update: fine-tune two small tensors so the greedy answer becomes the target.
    n_layers = model.config.num_hidden_layers
    names = [f"model.layers.{n_layers - 1}.mlp.down_proj.weight", "model.norm.weight"]
    params = dict(model.named_parameters())
    snapshot = common.snapshot_params(model, names)
    flags = {n: p.requires_grad for n, p in params.items()}
    for n, p in params.items():
        p.requires_grad_(n in names)
    model.train()
    model.gradient_checkpointing_disable()
    full = tokenizer(prompt + target + "<|im_end|>", add_special_tokens=False, return_tensors="pt")
    ids = full["input_ids"].cuda()
    labels = ids.clone()
    labels[:, : len(prompt_ids)] = -100
    opt = torch.optim.Adam([params[n] for n in names], lr=1e-3)
    losses = []
    for _ in range(60):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if losses[-1] < 0.05:
            break
    model.eval()
    hf_after = strip_pad(greedy(model, tokenizer, [prompt], max_new_tokens=12)[0], tokenizer)

    t0 = time.perf_counter()
    vg.sync_weights()  # TRL's own trainer -> vLLM path
    sync_s = time.perf_counter() - t0
    after = vllm_gen(vg, [prompt_ids], n=1, temperature=0.0, max_tokens=12)

    # Restore the original weights and sync again, so a shared server is left clean.
    with torch.no_grad():
        for n, t in snapshot.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    for n, p in params.items():
        p.requires_grad_(flags[n])
    vg.sync_weights()
    restored = vllm_gen(vg, [prompt_ids], n=1, temperature=0.0, max_tokens=12)[0]

    after = after[0]
    k = 4
    return {
        "passed": after != before and after[:k] == hf_after[:k],
        "how": "trl.generation.vllm_generation.VLLMGeneration.sync_weights()",
        "update": {"params": names, "sft_steps": len(losses), "final_loss": losses[-1]},
        "vllm_before": tokenizer.decode(before),
        "vllm_after": tokenizer.decode(after),
        "hf_after": tokenizer.decode(hf_after),
        "first_tokens_match_hf": after[:k] == hf_after[:k],
        "restored_matches_before": restored[:k] == before[:k],
        "sync_seconds": round(sync_s, 2),
    }


def record_kernel_usage(report: Report) -> None:
    """Attach runtime call counts to the kernels check: static fast + called = actually used."""
    checks = report.data["checks"]
    kern = checks.get("kernels")
    if not kern or "details" not in kern:
        return
    calls: dict[str, int] = {}
    warnings: list[str] = []
    for name in ("forward_backward", "hf_generate"):
        details = checks.get(name, {}).get("details", {})
        for k, v in (details.get("kernel_calls") or {}).items():
            calls[k] = calls.get(k, 0) + v
        warnings += details.get("fallback_warnings") or []
    functions = kern["details"].get("functions", {})
    used = {n: bool(f.get("fast")) and calls.get(n, 0) > 0 for n, f in functions.items()}
    kern["details"]["runtime_calls"] = calls
    kern["details"]["fast_and_called"] = used
    kern["details"]["fallback_warnings_seen"] = warnings
    if warnings:
        kern["passed"] = False
        kern["details"]["slow_fallback"] = True
    report.save()


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3.5 training-path compatibility check (§0.3)")
    p.add_argument("--model", required=True, help="Hub id, e.g. Qwen/Qwen3.5-2B")
    p.add_argument("--vllm-mode", choices=["colocate", "server", "none"], default="colocate")
    p.add_argument("--vllm-server-url", default="http://localhost:8000")
    p.add_argument("--vllm-group-port", type=int, default=51216)
    p.add_argument("--vllm-server-timeout", type=float, default=240.0)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=4, help="forward/backward rows")
    p.add_argument("--seq-len", type=int, default=1664, help="prompt 1536 + completion 128")
    p.add_argument("--skip-resume", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    common.setup_torch()
    if not torch.cuda.is_available():
        print("CUDA is not available; this check must run on a GPU node.", file=sys.stderr)
        return 2
    slug = model_slug(args.model)
    out = args.out or common.RUNS_DIR / "compat" / f"{slug}.json"
    export_dir = common.ensure_export(args.model)
    tokenizer = common.load_tokenizer(export_dir)
    report = Report(
        out,
        {
            "model": args.model,
            "export_dir": str(export_dir),
            "vllm_mode": args.vllm_mode,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": common.package_versions(),
            "gpu": common.gpu_info(),
        },
    )

    model = common.load_policy(export_dir, "float32", device="cuda")
    report.run("load", lambda: check_load(model, export_dir))
    report.run("thinking_disabled", lambda: check_thinking(tokenizer))
    report.run("kernels", check_kernels_static)
    report.run(
        "forward_backward",
        lambda: check_forward_backward(model, tokenizer, args.batch_size, args.seq_len),
    )
    free_cuda()
    report.run("hf_generate", lambda: check_hf_generate(model, tokenizer))
    report.run("tool_call_format", lambda: check_tool_call(model, tokenizer))
    record_kernel_usage(report)

    if args.skip_resume:
        report.skip("save_resume", "--skip-resume")
    else:
        model.to("cpu")
        free_cuda()
        workdir = common.RUNS_DIR / "compat" / f"{slug}-resume"
        report.run(
            "save_resume",
            lambda: check_save_resume(export_dir, tokenizer, workdir, args.keep_checkpoints),
        )
        model.to("cuda")
        free_cuda()

    if args.vllm_mode == "none":
        report.skip("vllm_generate", "--vllm-mode none")
        report.skip("weight_sync", "--vllm-mode none")
    else:
        holder: dict[str, Any] = {}

        def init_and_generate() -> dict[str, Any]:
            holder["vg"] = make_vllm(model, tokenizer, args)
            return check_vllm_generate(holder["vg"], tokenizer)

        entry = report.run("vllm_generate", init_and_generate)
        if "vg" in holder:
            report.run("weight_sync", lambda: check_weight_sync(holder["vg"], model, tokenizer))
        else:
            report.skip("weight_sync", f"vLLM did not start: {entry.get('error')}")

    # One-line answers for PROGRESSION §0.3 / §0.4.
    checks = report.data["checks"]
    kern = checks.get("kernels", {}).get("details", {})
    fb = checks.get("forward_backward", {}).get("details", {})
    report.data["for_progression"] = {
        "kernels": {n: f.get("implementation") for n, f in (kern.get("functions") or {}).items()},
        "slow_fallback": kern.get("slow_fallback", True),
        "tool_call_format": checks.get("tool_call_format", {}).get("details", {}).get("format"),
        "train_tokens_per_s": fb.get("train_tokens_per_s"),
        "train_peak_allocated_gb": fb.get("peak_allocated_gb"),
        "hf_generate_tokens_per_s": checks.get("hf_generate", {})
        .get("details", {})
        .get("generate_tokens_per_s"),
        "vllm_generate_tokens_per_s": checks.get("vllm_generate", {})
        .get("details", {})
        .get("generate_tokens_per_s"),
        "checkpoint_size_gb": checks.get("save_resume", {})
        .get("details", {})
        .get("checkpoint_size_gb"),
    }
    ok = report.finish()
    common.print_json("summary", {**report.data["summary"], **report.data["for_progression"]})
    print(f"\nreport: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
