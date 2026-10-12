#!/usr/bin/env bash
# §1.4 dry runs on one 8-GPU node: the real train/grpo.py, 10 steps, checkpoints every 5.
#   GPU 0     2B robust  -> runs/dry-2b-rob
#   GPU 1     2B leaky   -> runs/dry-2b-leak
#   GPU 4-7   4B robust  -> runs/dry-4b-rob   (4 colocated ZeRO-3 ranks, DECISIONS #9)
#   then GPU 0: resume dry-2b-rob from recovery/checkpoint-5 into runs/dry-2b-rob-resume (steps 6-10)
# Each job logs to runs/dry/<name>.log, ending with `PROGRESS_EXIT <code>` (lightning-jobs progress.py).
# Usage: bash scripts/dry_runs.sh    (TIMEOUT=seconds per job, default 3600)
set -u
unset UV_LIGHTNING_VIRTUALENV_ROOT
cd "$(dirname "$0")/.."
OUT=runs/dry
mkdir -p "$OUT"
TIMEOUT=${TIMEOUT:-3600}
DRY=(--max-steps 10 --eval-every 5 --recovery-every 5)

job() {  # job NAME SIZE REWARD GPUS [EXTRA...]: one grpo.py run, logged
  local name=$1 size=$2 reward=$3 gpus=$4
  shift 4
  rm -rf "runs/$name"
  timeout "$TIMEOUT" .venv/bin/python -u train/grpo.py --run-id "$name" --size "$size" --reward "$reward" \
    --seed 1 --gpus "$gpus" --output-dir "runs/$name" "${DRY[@]}" "$@" > "$OUT/$name.log" 2>&1
  local code=$?
  echo "PROGRESS_EXIT $code" >> "$OUT/$name.log"
  return $code
}

( job dry-2b-rob 2b robust 0 && job dry-2b-rob-resume 2b robust 0 --resume-from runs/dry-2b-rob/recovery/checkpoint-5 ) &
job dry-2b-leak 2b leaky 1 &
job dry-4b-rob 4b robust 4,5,6,7 &
wait
