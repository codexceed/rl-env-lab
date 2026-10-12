# GPU checks and benchmarks (PROGRESSION §0.3, §0.4)

Three scripts, run on a Linux H200 node in this order:

```
make setup-gpu  ──▶  make compat (×3 sizes)  ──▶  make sweep (toy LR sweep)  ──▶  make bench
                       runs/compat/*.json           runs/toy/*/summary.json          runs/bench/summary.json
```

| Script | Make target | What it answers |
|---|---|---|
| `compat_check.py` | `make compat MODEL=…` | Does the full training path work for this size? (§0.3) |
| `smoke_toy.py` | `make toy` / `make sweep` | Does reward rise at this LR without entropy collapse? (§0.4) |
| `bench.py` | `make bench` | Step time and memory at real lengths; can eval keep up? (§0.4) |
| `export_text.py` | (called by the others) | Saves a clean text-only copy of the model under `runs/models/` |

## 1. Setup on a fresh node

```bash
git clone <repo> && cd rl-env-lab
nvcc --version            # must be CUDA 13.x (torch 2.13 is built for cu130); set CUDA_HOME if needed
make setup-gpu            # gpu extra + causal-conv1d build (~15 min on 16 CPUs; no GPU needed)
```

`make setup-gpu` runs `scripts/setup_gpu.sh`. It works on a CPU machine, since compiling needs no GPU, so run it before switching a studio to H200s:

1. `uv sync --group dev --extra gpu --inexact`
2. Install a pip CUDA compiler (`nvidia-cuda-nvcc`) whose version **exactly matches torch's CUDA** (13.0)
3. Build `causal-conv1d==1.7.0` for `TORCH_CUDA_ARCH_LIST=9.0` (H100/H200). This takes about 12 minutes on 16 CPUs.
4. Import check, then the test suite

- **Why the source build:** `causal-conv1d` ships no PyPI wheels, and its GitHub release wheels don't cover torch 2.13 on Python 3.14. `flash-linear-attention` is pure Python/Triton and already comes with the `gpu` extra.
- **Rerun the build after any `uv sync`:** `uv sync` removes packages that aren't in the lock. `uv run` doesn't remove them, so `make` targets are safe.
- **If the build fails:** PROGRESSION §0.3 says to retry on Python 3.12 before switching model family.

## 2. Compat checks (§0.3)

```bash
make compat MODEL=Qwen/Qwen3.5-0.8B
make compat MODEL=Qwen/Qwen3.5-2B
make compat MODEL=Qwen/Qwen3.5-4B
```

These use one GPU each, with colocated vLLM. Run them on different GPUs in parallel by prefixing `CUDA_VISIBLE_DEVICES=<i>`. Add `--vllm-mode none` to test the HF-only path. Each run writes `runs/compat/<model>.json`, and exits 0 only if every required check passed.

**4B in server mode** (the planned 4B layout), with the server on GPU 0 and the check on GPU 1:

```bash
uv run python train/export_text.py --model Qwen/Qwen3.5-4B
CUDA_VISIBLE_DEVICES=0 uv run trl vllm-serve --model runs/models/Qwen3.5-4B-text \
    --port 8000 --gpu_memory_utilization 0.9 --max_model_len 2048 &
CUDA_VISIBLE_DEVICES=1 uv run python train/compat_check.py --model Qwen/Qwen3.5-4B \
    --vllm-mode server --vllm-server-url http://localhost:8000 --out runs/compat/Qwen3.5-4B-server.json
```

`trl vllm-serve` is deprecated in TRL 1.15. It prints the equivalent raw command: `VLLM_SERVER_DEV_MODE=1 vllm serve … --weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1 --enable-scale-out`.

Checks in the report (`checks.<name>.passed`, with details):

| Check | Pass means |
|---|---|
| `load` | `Qwen3_5ForCausalLM` from the text-only export, with no vision modules |
| `thinking_disabled` | `enable_thinking=False` renders `<think>\n\n</think>\n\n`. 0.8B and 2B already default to no thinking; **4B defaults to thinking on**, so always pass the flag |
| `kernels` | fla and causal-conv1d are resolved and called. `slow_fallback: true` fails the check |
| `forward_backward` | bf16 autocast, gradient checkpointing, 4×1,664 tokens, finite non-zero gradients |
| `hf_generate` | HF generate works (the fallback path); reports tokens/s |
| `tool_call_format` | Records the template's native tool-call format, plus one model attempt |
| `save_resume` | TRL checkpoint → resume restores model, optimizer, scheduler, RNG and trainer state; step-4 reward matches |
| `vllm_generate` | vLLM through TRL's `VLLMGeneration` (colocate or server) |
| `weight_sync` | TRL's `sync_weights()`: a tiny update changes vLLM's greedy output, which then matches HF |

## 3. Toy LR sweep (§0.4)

```bash
make sweep MODEL=Qwen/Qwen3.5-2B      # LRS="1e-6 3e-6 1e-5", 100 steps each, 1 GPU, colocated vLLM
make toy MODEL=Qwen/Qwen3.5-2B LR=3e-6 STEPS=100
```

To run the three LRs in parallel on three GPUs:

```bash
for i in 0 1 2; do lr=$(echo 1e-6 3e-6 1e-5 | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$i uv run python train/smoke_toy.py --model Qwen/Qwen3.5-2B --lr $lr --steps 100 &
done; wait
```

- **Fallback:** if vLLM fails, add `--hf-generate`.
- **Starting difficulty:** each run writes a `hints` list in `summary.json`. If reward starts near 0 or near 1, change `--digits` (default 5).
- **Output:** `runs/toy/<model>-lr<lr>/metrics.jsonl` has one row per step: `reward`, `dead_group_share`, `entropy`, `completion_length`, `wall_step_time`, plus TRL's `trl_step_time`, `loss`, `grad_norm` and memory.
- **Verdict:** `summary.json` holds `reward_rose` (mean of the last 20 steps > first 20), `entropy_collapsed`, `weights.changed` and `passed`.

## 4. Benchmarks (§0.4)

```bash
make bench        # needs >= 4 GPUs; ~30-60 min; sequential, one job at a time
```

What it runs:

| Job | GPUs | Notes |
|---|---|---|
| 0.8B, 2B | 1 each, colocated vLLM | Prompt 1,536 + completion 128 tokens (forced), 8×8 |
| 4B `server-3train` | 1 server + 3 ZeRO-3 | The planned layout. **64 completions can't split over 3 ranks**, so it runs 9×8 = 72 and also reports time scaled to 64 |
| 4B `server-2train` | 1 server + 2 ZeRO-3 | Exactly 64 completions |
| 4B `colocate-4train` | 4 ZeRO-3, colocated vLLM | Default (with `server-2train`); `server-3train` is opt-in only (DECISIONS #9) |
| Eval | 2 (one vLLM engine each) | 250 prompts × 4 samples, temperature 0.7, per size |

Configs are `configs/accelerate/zero3_{2,3,4}gpu.yaml`. Each job's log is in `runs/bench/logs/`, its result in `runs/bench/<tag>.json`, and the combined result plus a Markdown table in `runs/bench/summary.json` (also printed).

To run a single benchmark by hand, see the docstring of `train/bench.py`. When two accelerate jobs share a node, give each its own `--main_process_port`, `--vllm-server-url` port and `--vllm-group-port`.

## 5. What to paste back into PROGRESSION.md

Don't edit PROGRESSION.md from a workstream. Paste these into the PR or a message for the integrator.

| PROGRESSION item | Where to read it |
|---|---|
| §0.3 "Loads as text-only…" | `checks.load.passed` for each size |
| §0.3 "Forward + backward…" | `checks.forward_backward.passed` |
| §0.3 "Optimized kernels… record which: `____`" | `for_progression.kernels` (e.g. `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`, `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn`) and `slow_fallback: false` |
| §0.3 "Generation works: vLLM colocated… and vLLM server" | `checks.vllm_generate` from the default run and from the 4B server run |
| §0.3 "Weight sync…" | `checks.weight_sync.passed` (`vllm_before` vs `vllm_after`) |
| §0.3 "Save → resume…" | `checks.save_resume.restored` (all true) and `loss_continues` |
| §0.3 "Thinking mode disabled…" | `checks.thinking_disabled` |
| §0.3 "Native tool-call format identified: `____`" | `for_progression.tool_call_format`: `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>` |
| §0.3 "Memory and throughput…" | `for_progression.*_tokens_per_s`, `train_peak_allocated_gb` |
| §0.1 "Storage measured after first checkpoint save" | `for_progression.checkpoint_size_gb`. This is the 2B full checkpoint: fp32 weights plus Adam state |
| §0.4 toy rows and "Done when: reward rises…" | `runs/toy/*/summary.json`: `reward_rose`, `entropy_collapsed`, `weights.changed`, `wall_step_time_s` |
| §0.4 "Step time + peak memory… → update Run matrix" | The first table printed by `make bench`. The Run matrix "Est. time" is `Est. h @ 300 steps` |
| §0.4 "Eval throughput… keeps pace" | The second table printed by `make bench` ("Keeps pace?") |
| §0.8 / DECISIONS rows 6–9 | LR from the sweep; backend from compat; 4B layout from bench (see the 3-rank note above) |

## Notes on the locked stack

These were verified against transformers 5.17, TRL 1.15 and vLLM 0.31.

- **Text-only loading:**
  - The Qwen3.5 Hub repos are vision-language checkpoints (`Qwen3_5ForConditionalGeneration`). `AutoModelForCausalLM` gives the text-only `Qwen3_5ForCausalLM`.
  - `GRPOTrainer(model="<id>")` would build the VLM, so the scripts pass a model instance and a plain tokenizer.
- **Clean export:**
  - TRL's colocated vLLM is built from `model.name_or_path`. Pointing it at the Hub id loads the VLM, whose weight names don't match the text-only trainer during weight sync.
  - A model loaded straight from the Hub id is re-saved by `Trainer` checkpoints with the old `model.language_model.*` prefix. Resume (`load_state_dict(strict=False)`) then silently loads nothing.
  - So everything trains and serves from `runs/models/<name>-text`, which has clean `model.layers.*` names.
- **Silent kernel fallback:** transformers silently uses slow PyTorch code if `fla` or `causal_conv1d` is missing, and logs one warning. The scripts refuse to run on that path unless you pass `--allow-slow-kernels`.
- **fp32 master weights:** the scripts load fp32 weights and get bf16 compute from `bf16=True` autocast. In pure-bf16 weights, full-fine-tune updates at LR 1e-6 round away.
- **HF fallback generation:** TRL calls `generate` outside autocast. With `--hf-generate`, a small subclass wraps it in bf16 autocast.
- **TRL fused LM head:** TRL 1.15 always computes training log-probs with a Triton fused LM-head kernel, so training needs Linux and CUDA.
- **Batch layout:** TRL requires completions per step to be divisible by per-device batch × training ranks. 64 completions works on 1, 2, 4 or 8 ranks, never on 3.
