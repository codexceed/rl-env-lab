# GRPO runs and held-out eval (PROGRESSION §1.4)

```
configs/grpo.yaml (+ sizes/<size>.yaml: layout only)
        │
train/grpo.py ──▶ runs/<id>/eval/step-NNNNN/ + COMPLETE ──▶ eval/watcher.py ──▶ eval/heldout.py ──▶ evals/<id>/step-NNNNN.jsonl
        │        runs/<id>/recovery/checkpoint-N/ + COMPLETE  (resume, rolling prune when enabled)
        └──────▶ runs/<id>/train_metrics.jsonl  (one row per step: reward, robust_reward_mean, leaky_reward_mean, ...)
```

| File | Role |
|---|---|
| `train/grpo.py` | One run: `--run-id --size --reward --seed --gpus --output-dir` (+ `--resume-from`, `--vllm-server-gpus --train-gpus`, `--max-steps`, `--hf-generate`) |
| `train/rewards.py` | `robust_reward`, `leaky_reward`: completion + serialized `TaskInstance` → 1.0 / 0.0 |
| `train/callbacks.py` | Eval weights, `COMPLETE` markers, rolling prune, `train_metrics.jsonl` |
| `eval/heldout.py` | Formal eval (250 × 4, temp 0.7) or `--mode diagnostic` (50 per train template × 8, temp 1.0) |

## Commands (on a GPU node, after `make setup-gpu` and `make data`)

```bash
# 10-step dry run per reward on 2B (GPU 0 and 1)
uv run python train/grpo.py --run-id dry-2b-rob  --size 2b --reward robust --seed 1 --gpus 0 --output-dir runs/dry-2b-rob  --max-steps 10 --eval-every 5 --recovery-every 5
uv run python train/grpo.py --run-id dry-2b-leak --size 2b --reward leaky  --seed 1 --gpus 1 --output-dir runs/dry-2b-leak --max-steps 10 --eval-every 5 --recovery-every 5

# resume test: continue from checkpoint-5 in a copy and compare train_metrics.jsonl steps 6-10
cp -r runs/dry-2b-rob runs/dry-2b-rob-resume && rm -rf runs/dry-2b-rob-resume/{eval/step-00010,recovery/checkpoint-10,train_metrics.jsonl}
uv run python train/grpo.py --run-id dry-2b-rob-resume --size 2b --reward robust --seed 1 --gpus 0 \
    --output-dir runs/dry-2b-rob-resume --max-steps 10 --eval-every 5 --recovery-every 5 \
    --resume-from runs/dry-2b-rob-resume/recovery/checkpoint-5

# eval a dry-run checkpoint, the base model, and the diagnostic baseline
uv run python eval/heldout.py --run-id dry-2b-rob --size 2b --reward robust --seed 1 --checkpoint runs/dry-2b-rob --step final
make eval-base SIZE=2b          # -> evals/base-2b/step-00000.jsonl
make diagnostic SIZE=2b         # -> evals/diagnostic/base-2b_step-00000.jsonl + .summary.json
```

Only after the resume test passes: set `checkpoints.pruning_enabled: true` in `configs/grpo.yaml`.

## Notes

- **Text-only export:** trains and serves from `runs/models/<name>-text` (`train/common.py`).
- **Prompts:** rendered once with `envlab.prompts.render_prompt` (thinking off) and given to TRL as plain text. A prompt over 1,536 tokens stops the run (TRL 1.15 has no prompt truncation).
- **Both graders every step:** TRL gets both reward functions with weights `[1, 0]` (robust run) or `[0, 1]` (leaky run). The advantages equal those of the own reward alone; TRL logs both means.
- **Decoding:** rewards and eval decode completion token ids with special tokens kept, so a `<tool_call>` tag is never stripped before parsing.
- **4B:** default layout is 1 vLLM server + 2 ZeRO-3 ranks (64 completions can't split over 3 ranks); the 4th GPU idles. `configs/sizes/4b.yaml` documents the 4-GPU colocated alternative. DECISIONS #9 picks one.
- **Ports:** offset by the run's lowest GPU index (accelerate `29500+i`, vLLM `8000+i`, weight sync `51216+i`).
- **Placeholders:** learning rate 3e-6 (DECISIONS #6) and 300 steps (DECISIONS #7).
