#!/usr/bin/env bash
# §2.2 diagnostic baseline + §2.3 formal base eval for 0.8B / 2B / 4B, in parallel, one GPU each (GPUs 0-5).
# Logs: runs/base_evals/<job>.log, each ending with `PROGRESS_EXIT <code>` (lightning-jobs progress.py).
# Usage: bash scripts/base_evals.sh    (TIMEOUT=seconds per job, default 3600)
set -u
unset UV_LIGHTNING_VIRTUALENV_ROOT
cd "$(dirname "$0")/.."
OUT=runs/base_evals
mkdir -p "$OUT"
TIMEOUT=${TIMEOUT:-3600}

launch() {  # launch NAME GPU ARGS...
  local name=$1 gpu=$2
  shift 2
  (
    CUDA_VISIBLE_DEVICES=$gpu timeout "$TIMEOUT" .venv/bin/python -u eval/heldout.py --gpus "$gpu" "$@" > "$OUT/$name.log" 2>&1
    echo "PROGRESS_EXIT $?" >> "$OUT/$name.log"
  ) &
}

gpu=0
for size in 0.8b 2b 4b; do
  launch "diagnostic-$size" "$gpu" --size "$size" --checkpoint base --mode diagnostic
  launch "formal-$size" "$((gpu + 1))" --size "$size" --checkpoint base
  gpu=$((gpu + 2))
done
wait
