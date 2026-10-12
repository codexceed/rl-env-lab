#!/usr/bin/env bash
# §1.4 dry runs on one 8-GPU node: the real train/grpo.py, 12 steps, eval and recovery every 5, with
# asynchronous recovery checkpoints (checkpoints.async_recovery in configs/grpo.yaml).
#   GPU 0     2B robust  -> runs/dry-2b-rob,  then resume from its async checkpoint-5 -> runs/dry-2b-rob-resume
#   GPU 1     2B leaky   -> runs/dry-2b-leak
#   GPU 4-7   4B robust  -> runs/dry-4b-rob   (4 colocated ZeRO-3 ranks, DECISIONS #9),
#                         then resume from its async checkpoint-5 -> runs/dry-4b-rob-resume
# Then, per run: the recovery events (snapshot / write seconds, skipped steps, no staging dir left),
# and per resume: steps 6-12 of train_metrics.jsonl against the original (step 6 must match).
# Optional, DRY_V2=1 (ADR 0005 v2):
#   GPU 2     2B robust --variant v2-think -> runs/dry-v2-2b-rob (12 steps, one eval step at 12), then a
#             formal eval of that step on 8 instances; checks thinking on and the 512 cap in training and eval.
# Each job logs to runs/dry/<name>.log, ending with `PROGRESS_EXIT <code>` (lightning-jobs progress.py).
# Usage: bash scripts/dry_runs.sh    (TIMEOUT=seconds per job, default 3600; DRY_V2=1 adds the v2 run).
# Exit 0 only if all pass.
set -u
unset UV_LIGHTNING_VIRTUALENV_ROOT
cd "$(dirname "$0")/.."
OUT=runs/dry
mkdir -p "$OUT"
TIMEOUT=${TIMEOUT:-3600}
export DRY_V2=${DRY_V2:-0}
DRY=(--max-steps 12 --eval-every 5 --recovery-every 5)

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

pair() {  # pair NAME SIZE GPUS: a run, then a resume from its async recovery checkpoint-5
  local name=$1 size=$2 gpus=$3
  job "$name" "$size" robust "$gpus" && \
    job "$name-resume" "$size" robust "$gpus" --resume-from "runs/$name/recovery/checkpoint-5"
}

v2_eval() {  # formal eval of the v2 dry run's final step on 8 instances: thinking + cap come from the run
  CUDA_VISIBLE_DEVICES=2 timeout "$TIMEOUT" .venv/bin/python -u eval/heldout.py --run-id dry-v2-2b-rob --size 2b \
    --reward robust --seed 1 --checkpoint runs/dry-v2-2b-rob --step final --limit 8 --gpus 2 \
    --output "$OUT/dry-v2-2b-rob.eval.jsonl" > "$OUT/dry-v2-2b-rob-eval.log" 2>&1
  local code=$?
  echo "PROGRESS_EXIT $code" >> "$OUT/dry-v2-2b-rob-eval.log"
  return $code
}

pair dry-2b-rob 2b 0 &
p1=$!
job dry-2b-leak 2b leaky 1 &
p2=$!
pair dry-4b-rob 4b 4,5,6,7 &
p3=$!
pids="$p1 $p2 $p3"
if [ "$DRY_V2" = 1 ]; then
  ( job dry-v2-2b-rob 2b robust 2 --variant v2-think --eval-every 12 && v2_eval ) &
  pids="$pids $!"
fi
status=0
for p in $pids; do wait "$p" || status=1; done

.venv/bin/python - <<'EOF' || status=1
import json, math, os, sys
from pathlib import Path

ok = True

def rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

for run in ("dry-2b-rob", "dry-2b-leak", "dry-4b-rob", "dry-2b-rob-resume", "dry-4b-rob-resume"):
    rdir = Path("runs") / run
    events = rows(rdir / "recovery_events.jsonl")
    written = [e for e in events if e["event"] == "recovery_written"]
    print(f"\n== {run}: recovery events")
    for e in events:
        extra = {k: e[k] for k in ("snapshot_s", "write_s", "max_rank_write_s", "total_s", "bytes", "reason") if k in e}
        print(f"  step {e['step']:>3} {e['event']:<18} {extra}")
    staging = sorted(p.name for p in (rdir / "recovery").glob(".tmp-checkpoint-*"))
    complete = sorted(int(p.parent.name.split("-")[1]) for p in (rdir / "recovery").glob("checkpoint-*/COMPLETE"))
    print(f"  complete checkpoints {complete}; staging dirs left {staging}")
    if not written or staging or any(e["event"] == "recovery_failed" for e in events):
        print(f"  FAIL: expected async writes, no failures and no staging dir left")
        ok = False

def close(a, b, rel):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)

for size in ("2b", "4b"):
    orig = {r["step"]: r for r in rows(f"runs/dry-{size}-rob/train_metrics.jsonl")}
    res = {r["step"]: r for r in rows(f"runs/dry-{size}-rob-resume/train_metrics.jsonl")}
    print(f"\n== {size}: resume from async checkpoint-5 vs original (reward, loss)")
    for step in range(6, 13):
        a, b = orig.get(step, {}), res.get(step, {})
        same = close(a.get("reward"), b.get("reward"), 1e-6) and close(a.get("loss"), b.get("loss"), 1e-3)
        print(f"  step {step:>2}  orig {a.get('reward')}, {a.get('loss')}  resumed {b.get('reward')}, {b.get('loss')}"
              f"  {'match' if same else 'DIFF'}")
        if step == 6 and not same:  # the first resumed step sees exactly the saved state
            print(f"  FAIL: {size} step 6 differs after resume")
            ok = False

if os.environ.get("DRY_V2") == "1":  # ADR 0005 v2: thinking on and the 512 cap, in training and in the eval
    rdir = Path("runs/dry-v2-2b-rob")
    cfg = json.loads((rdir / "train_config.json").read_text()) if (rdir / "train_config.json").exists() else {}
    man = json.loads((rdir / "run_manifest.json").read_text()) if (rdir / "run_manifest.json").exists() else {}
    budget = cfg.get("budget", {})
    steps = sorted(p.parent.name for p in (rdir / "eval").glob("step-*/COMPLETE"))
    elog = Path("runs/dry/dry-v2-2b-rob-eval.log")
    elog = elog.read_text() if elog.exists() else ""
    traces = rows("runs/dry/dry-v2-2b-rob.eval.traces.jsonl")
    checks = {
        "train_config variant v2-think": cfg.get("variant") == "v2-think",
        "training thinking on, cap 512": budget.get("enable_thinking") is True and budget.get("max_completion_length") == 512,
        "run_manifest variant v2-think": man.get("variant") == "v2-think",
        "one eval step (12)": steps == ["step-00012"],
        "eval: thinking on, cap 512": "run variant v2-think: thinking on, cap 512" in elog,
        "eval: 8 instances x 4 samples graded as thinking-on": len(traces) == 32 and all("graded_text" in t for t in traces),
    }
    print("\n== v2 dry run (DRY_V2=1)")
    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok = ok and passed
    if traces:
        parsed = sum(t["parse_ok"] for t in traces)
        longest = max(len(t["completion"]) for t in traces)
        print(f"  parse ok {parsed}/{len(traces)}; longest completion {longest} chars")
        print(f"  sample: {traces[0]['completion'][:300]!r}")
sys.exit(0 if ok else 1)
EOF
echo "DRY_RUNS_EXIT $status"
exit $status
