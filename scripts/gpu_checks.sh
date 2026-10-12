#!/usr/bin/env bash
# §0.3 + §0.4 on one 8-GPU node, all at once:
#   1. export the text-only checkpoints for 0.8B / 2B / 4B (sequential, shared cache)
#   2. in parallel, one GPU each: compat for 3 sizes (GPUs 0-2) and the 2B toy LR sweep (GPUs 3-5)
# Each job logs to runs/gpu_checks/<job>.log, ending with `PROGRESS_EXIT <code>` (lightning-jobs progress.py).
# Usage: bash scripts/gpu_checks.sh            (TIMEOUT=seconds per job, default 7200)
set -u
unset UV_LIGHTNING_VIRTUALENV_ROOT
cd "$(dirname "$0")/.."
OUT=runs/gpu_checks
mkdir -p "$OUT"
TIMEOUT=${TIMEOUT:-7200}
SIZES=("0.8B" "2B" "4B")
LRS=("1e-6" "3e-6" "1e-5")

launch() {  # launch NAME GPU ARGS...: one detached job on one GPU, own rendezvous port
  local name=$1 gpu=$2
  shift 2
  (
    CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((29500 + gpu)) timeout "$TIMEOUT" .venv/bin/python -u "$@" \
      > "$OUT/$name.log" 2>&1
    echo "PROGRESS_EXIT $?" >> "$OUT/$name.log"
  ) &
}

{
  for i in "${!SIZES[@]}"; do
    echo "PROGRESS_PHASE export-${SIZES[$i]} $((i + 1))/${#SIZES[@]}"
    timeout 3600 .venv/bin/python -u train/export_text.py --model "Qwen/Qwen3.5-${SIZES[$i]}" || exit 1
  done
  echo "PROGRESS_EXIT 0"
} > "$OUT/export.log" 2>&1 || { echo "PROGRESS_EXIT 1" >> "$OUT/export.log"; exit 1; }

for i in "${!SIZES[@]}"; do
  launch "compat-${SIZES[$i]}" "$i" train/compat_check.py --model "Qwen/Qwen3.5-${SIZES[$i]}"
done
for i in "${!LRS[@]}"; do
  launch "toy-2B-lr${LRS[$i]}" "$((i + 3))" train/smoke_toy.py --model Qwen/Qwen3.5-2B --lr "${LRS[$i]}" --steps 100 \
    --vllm-group-port $((51216 + i + 3))
done
wait
