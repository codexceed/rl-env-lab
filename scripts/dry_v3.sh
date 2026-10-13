#!/usr/bin/env bash
# ADR 0007 v3 dry run on one 8-GPU node: the real train/grpo.py --variant v3, 12 steps, eval weights at step 12,
# recovery every 100 (so only the final-step one is written), then a formal eval of each final step on 8 held-out instances.
#   GPU 0-3   4B v3 robust -> runs/dry-v3-4b-rob   (4 colocated ZeRO-3 ranks), then its eval on GPU 0
#   GPU 4     2B v3 leaky  -> runs/dry-v3-2b-leak, then its eval on GPU 4
# The eval reads runs/dry/v3-heldout8/: 4 near + 4 far instances from data/v3/heldout.jsonl (one per template,
# alternating) plus data/v3/manifest.json, so both held-out kinds are present.
# Then the checks: variant v3 in train_config.json and run_manifest.json, data/v3 (spec_version v3.0) in the manifest
# identity, the 2,048-token prompt budget and 3,072-token context in effect in training and eval, and eval records
# with v3 templates of both kinds.
# Each job logs to runs/dry/<name>.log, ending with `PROGRESS_EXIT <code>` (lightning-jobs progress.py).
# Usage: bash scripts/dry_v3.sh   (`make dry-v3`; TIMEOUT=seconds per job, default 3600). Needs `make data-v3`.
# Exit 0 only if all pass.
set -u
unset UV_LIGHTNING_VIRTUALENV_ROOT
cd "$(dirname "$0")/.."
OUT=runs/dry
mkdir -p "$OUT"
TIMEOUT=${TIMEOUT:-3600}
DATA=data/v3
EVAL_DATA="$OUT/v3-heldout8"
DRY=(--max-steps 12 --eval-every 12 --recovery-every 100 --variant v3)

for f in train.jsonl heldout.jsonl manifest.json; do
  if [ ! -f "$DATA/$f" ]; then
    echo "$DATA/$f not found: run \`make data-v3\` first" >&2
    echo "DRY_V3_EXIT 2"
    exit 2
  fi
done

# 8 held-out instances, near and far alternating, one per template (F1N F1X F2N F2X ...)
.venv/bin/python - "$DATA" "$EVAL_DATA" <<'EOF' || { echo "DRY_V3_EXIT 2"; exit 2; }
import shutil, sys
from pathlib import Path
from envlab.tasks.instance import read_jsonl, write_jsonl
from envlab.tasks.v3 import FAR_TEMPLATES, NEAR_TEMPLATES

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
first = {}
for inst in read_jsonl(src / "heldout.jsonl"):
    first.setdefault(inst.template, inst)
order = [t for pair in zip(NEAR_TEMPLATES, FAR_TEMPLATES) for t in pair][:8]
missing = [t for t in order if t not in first]
if missing:
    sys.exit(f"{src}/heldout.jsonl has no instance of {missing}")
dst.mkdir(parents=True, exist_ok=True)
write_jsonl(dst / "heldout.jsonl", [first[t] for t in order])
shutil.copy(src / "manifest.json", dst / "manifest.json")
print(f"[dry-v3] eval subset: {order} -> {dst}")
EOF

job() {  # job NAME SIZE REWARD GPUS [EXTRA...]: one grpo.py --variant v3 run, logged
  local name=$1 size=$2 reward=$3 gpus=$4
  shift 4
  rm -rf "runs/$name"
  timeout "$TIMEOUT" .venv/bin/python -u train/grpo.py --run-id "$name" --size "$size" --reward "$reward" \
    --seed 1 --gpus "$gpus" --output-dir "runs/$name" "${DRY[@]}" "$@" > "$OUT/$name.log" 2>&1
  local code=$?
  echo "PROGRESS_EXIT $code" >> "$OUT/$name.log"
  return $code
}

evaluate() {  # evaluate NAME SIZE REWARD GPU: formal eval of the run's final step; v3 settings come from the run
  local name=$1 size=$2 reward=$3 gpu=$4
  rm -f "$OUT/$name.eval.jsonl" "$OUT/$name.eval.traces.jsonl"
  CUDA_VISIBLE_DEVICES=$gpu timeout "$TIMEOUT" .venv/bin/python -u eval/heldout.py --run-id "$name" --size "$size" \
    --reward "$reward" --seed 1 --checkpoint "runs/$name" --step final --gpus "$gpu" --data-dir "$EVAL_DATA" \
    --output "$OUT/$name.eval.jsonl" > "$OUT/$name-eval.log" 2>&1
  local code=$?
  echo "PROGRESS_EXIT $code" >> "$OUT/$name-eval.log"
  return $code
}

( job dry-v3-4b-rob 4b robust 0,1,2,3 && evaluate dry-v3-4b-rob 4b robust 0 ) &
p1=$!
( job dry-v3-2b-leak 2b leaky 4 && evaluate dry-v3-2b-leak 2b leaky 4 ) &
p2=$!
status=0
for p in $p1 $p2; do wait "$p" || status=1; done

.venv/bin/python - <<'EOF' || status=1
import json, re, sys
from pathlib import Path
from envlab.tasks.v3 import HELDOUT_TEMPLATES, SPEC_VERSION, heldout_kind

def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

def rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

def text(path):
    p = Path(path)
    return p.read_text() if p.exists() else ""

ok = True
for run in ("dry-v3-4b-rob", "dry-v3-2b-leak"):
    rdir = Path("runs") / run
    cfg, man = load(rdir / "train_config.json"), load(rdir / "run_manifest.json")
    budget, layout, data = cfg.get("budget", {}), cfg.get("layout", {}), man.get("data") or {}
    steps = sorted(p.parent.name for p in (rdir / "eval").glob("step-*/COMPLETE"))
    marker = load(rdir / "eval" / "step-00012" / "COMPLETE")
    tlog, elog = text(f"runs/dry/{run}.log"), text(f"runs/dry/{run}-eval.log")
    lengths = re.search(r"train prompts from (\S+); tokens min/max (\d+)/(\d+)", tlog)
    records = rows(f"runs/dry/{run}.eval.jsonl")
    kinds = {heldout_kind(r["template"]) for r in records}
    checks = {
        "train_config variant v3": cfg.get("variant") == "v3",
        "run_manifest variant v3": man.get("variant") == "v3",
        "COMPLETE of step 12 records variant v3": marker.get("variant") == "v3",
        "manifest identity: data dir data/v3": man.get("data_dir") == "data/v3" and cfg.get("data_dir") == "data/v3",
        f"manifest identity: spec_version {SPEC_VERSION}": data.get("spec_version") == SPEC_VERSION,
        "training: thinking on, cap 1024": budget.get("enable_thinking") is True and budget.get("max_completion_length") == 1024,
        "training: prompt budget 2048": budget.get("max_prompt_length") == 2048,
        "training: context 3072": layout.get("vllm_max_model_length") == 3072,
        "training: prompts read from data/v3/train.jsonl": bool(lengths) and lengths[1] == "data/v3/train.jsonl",
        # the trainer always saves recovery at the final step; recovery_every 100 means no other one
        "one eval step (12), recovery only at the final step": steps == ["step-00012"]
        and sorted(p.name for p in (rdir / "recovery").glob("checkpoint-*")) == ["checkpoint-12"],
        "eval: run variant v3, thinking on, cap 1024": "run variant v3: thinking on, cap 1024" in elog,
        "eval: prompt budget 2048, context 3072": "prompts <= 2048 tokens, context 3072" in elog,
        "eval: 8 instances x 4 samples": len(records) == 32 and len({r["instance_id"] for r in records}) == 8,
        "eval: v3 held-out templates only": bool(records) and all(r["template"] in HELDOUT_TEMPLATES for r in records),
        "eval: near and far both present": kinds == {"near", "far"},
    }
    print(f"\n== {run}")
    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok = ok and passed
    print(f"  info: train prompt tokens min/max {lengths[2] + '/' + lengths[3] if lengths else '?'}; "
          f"train split matches its manifest: {data.get('matches_manifest')}")
    if records:
        for kind in ("near", "far"):
            sub = [r for r in records if heldout_kind(r["template"]) == kind]
            if sub:
                print(f"  info: {kind}: parse ok {sum(r['parse_ok'] for r in sub)}/{len(sub)}, "
                      f"robust {sum(r['robust_pass'] for r in sub)}, leaky {sum(r['leaky_pass'] for r in sub)}")
sys.exit(0 if ok else 1)
EOF
echo "DRY_V3_EXIT $status"
exit $status
