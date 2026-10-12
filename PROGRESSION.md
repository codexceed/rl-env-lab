# PROGRESSION — execution log ($100 budget plan)

**This file is the authoritative execution plan.** `INSTRUCTIONS.md` is the original plan, kept unchanged. Anything not overridden in **Plan deltas** is inherited from it as-is (see **Inherited unchanged**). This branch holds only these two files; decisions and agent working rules live in this file.

**Update rules**

- Tick a box the moment its *done-when* is met; append `— YYYY-MM-DD HH:MM, note` if useful.
- Never delete a task: mark it `[~]` (cut) or `[>]` (moved/replaced) with a one-line reason.
- Any change to scope, model, config, schedule, budget or statistics → row in **Change log**; if it touches a decision, also a new row in **Decisions** (old row marked superseded).
- After every GPU session, add a row to the **Spend ledger** and check the **Stop rules**.
- Keep **Status** current.

**Legend:** `[ ]` todo · `[x]` done · `[~]` cut · `[>]` moved/replaced · `[!]` blocked · 🚦 gate (must pass before the next phase) · ⚠ unvalidated estimate

**Naming:** **main model** = the model H1 and H2 run on — currently **Qwen3.5-0.8B**. Run IDs use the actual size (`0.8b-…`); if the main model changes, IDs change with it.

---

## Status

| | |
|---|---|
| **Today** | Mon 12 Oct 2026 |
| **Phase** | P0 — Setup |
| **Deadline** | **Fri 16 Oct, 12:00** (training cutoff Fri 06:00) |
| **Compute** | 1 × A100 80GB, RunPod, per-second billing |
| **Budget** | **$100 hard cap** · $73.61 planned · $26.39 contingency · spent **$0.00** |
| **Next gate** | 🚦 G-Mon (end of today) |
| **Blockers** | — |

---

## Plan deltas vs `INSTRUCTIONS.md`

| Area | Original | Now |
|---|---|---|
| Deadline | Sun 18 Oct | **Fri 16 Oct 12:00** |
| Compute | Scenario A (3 nodes) or B (one 24 GB GPU) | **Scenario B, $100 total, one A100 80GB** (see §Cost plan for why not 24 GB) |
| Main model | Qwen3-0.6B (LoRA) | **Qwen3.5-0.8B**; fallback Qwen3-0.6B |
| Size ladder | 0.6B / 1.7B | **Cut.** H5 cut; optional 2B leaky run only if money remains |
| Weights updated | LoRA | **Full fine-tune** (LoRA if the benchmark says it is ≥30% cheaper per step) |
| LR sweep | 1e-5 / 3e-5 / 1e-4 (LoRA range) | **1e-6 / 3e-6 / 1e-5** (full-FT range) |
| Seeds | 1 | **2 committed, 3rd if budget allows** |
| Steps | 300, fit in 8 h | **≤ 300, fit in 4.5 h** on the A100 |
| Checkpoints | Every 50 steps, all evaluated | **Saved every 50; formally evaluated at 100, 200, final** (others only if money remains) |
| Statistics | Per-run bootstrap CI | **Paired instance bootstrap + per-seed agreement**, rules in §0.7 |
| Hypotheses | H1–H5 | **H1, H2 tested; H3 exploratory; H4, H5 cut** |
| Triage classifier + API spend | Sat, ~$15 API | **Cut** ($0 API) |
| Multi-turn, ablations | Stretch | **Cut** |
| Shutdown | Each job stops its instance | **Same, enforced 3 ways:** per-job timeout, pod dead-man switch, prepaid credit cap |
| Python | 3.11 | **3.14**; retry on **3.12** if kernels fail (§0.3) |
| GPU deps | Always installed | **`gpu` extra, Linux only**; CI and local dev run without it |
| Reporting | Sat–Sun | **Skeleton Tue, live tables Thu, final review Fri** |

## Inherited unchanged from `INSTRUCTIONS.md`

Environment design (5 entities, 6 tools, 5 families × 3 templates) · T1/T2 train, T3 held out, frozen Monday · one tool call per task in native format · binary reward from simulator state · robust/leaky grader definitions · exploit definition · quality gates G1–G3 and acceptance bar · 2,000 / 250 / 100 data split and broken-variant types · GRPO config (8 gens × 8 prompts, prompt 1,536 / completion 128 tokens, temp 1.0, beta 0, bf16, grad checkpointing) · step-100 health definition · `uv`, `pytest`, CI · MIT, synthetic data · guardrails (personal accounts, no confidential material, `.env` never committed, timeouts and spend caps).

---

## Decisions

Fixed unless a fallback trigger fires. A change gets a new row; the old row is marked superseded.

| # | Area | Decision | Why | Status |
|---|---|---|---|---|
| 1 | Deadline | Fri 16 Oct 12:00; training cutoff Fri 06:00 | Availability | Locked |
| 2 | Budget | $100 cloud compute, hard cap; ≥25% held as contingency | Owner | Locked |
| 3 | Compute | 1 × A100 80GB SXM, RunPod Secure Cloud (Community if cheaper and available) | Cheapest per useful step that fits the deadline (§Cost plan) | Locked |
| 4 | Model family | Qwen3.5; fallback Qwen3 | Same family as the main plan | Pending §0.3 compat |
| 5 | Main model | Qwen3.5-0.8B | 2B costs ~2× per run; only 0.8B fits 2–3 seed pairs in budget | Pending §2.2 band check |
| 6 | Weights updated | Full fine-tune | Generation dominates step time, so LoRA saves little; avoids adapter sync to vLLM | Pending §0.4 benchmark |
| 7 | Learning rate | `____` (from 1e-6 / 3e-6 / 1e-5) | | Pending §0.4 |
| 8 | Step count | `____` (≤ 300; run ≤ 4.5 h) | | Pending §0.4 |
| 9 | Generation backend | `____` (vLLM colocated preferred; HF generate fallback for train + eval) | | Pending §0.3 |
| 10 | Seeds | 1, 2 committed; 3 if Stop rules allow | Budget | Locked |
| 11 | Hypotheses | H1, H2 tested; H3 exploratory; H4, H5 cut | Budget, deadline | Locked |
| 12 | Scope cuts | Size ladder, triage classifier, multi-turn, ablations | Budget, deadline | Locked |
| 13 | Python | 3.14; kernel failure → source build → 3.12 → model switch | `causal-conv1d` has no PyPI wheels | Locked |
| 14 | Shutdown | Every GPU job stops its pod on exit; pod has a dead-man switch | Idle GPUs are the main budget risk | Locked |

## Agent working rules

- `PROGRESSION.md` is the plan; `INSTRUCTIONS.md` is **never edited**. A planned `SPEC.md` (environment contract) wins over code once signed off; `PREREG.md` is frozen before task training.
- Branch `<type>/<issue-id>-<description>`; Conventional Commits `<type>(<scope>): <subject>`; PR sections **Description, Motivation, Changes, Testing**. **No AI or assistant attribution** anywhere (no `Co-Authored-By` trailers, no "generated with" footers). Commit as the repo-local git user.
- Parallel workstreams don't edit this file; they report which boxes are met, and the integrator ticks them.
- **GPU only for:** compat, toy sweep, benchmarks, dry runs, baseline, training, checkpoint eval. Everything else (simulator, graders, data, metrics, reports) is built and tested locally on CPU at $0.
- CI has no GPU: code imported by tests never imports torch, transformers or vllm at module top level.
- Code: explicit `seed` everywhere (no global RNG); pure simulator (no I/O, no wall-clock); graders never mutate state; money in integer cents; grader tests cover correct, wrong amount, wrong record, side effect per family.
- Never launch a GPU job without a timeout, and never leave a pod running without its dead-man switch.

---

## GPU prices (checked 2026-10-12)

On-demand, single GPU, USD per hour. Marketplace prices move daily: re-check before renting.

| GPU | RunPod (Community / Secure) | Lambda | Modal (per-second) | Lightning AI | Vast.ai (verified, 1 GPU) |
|---|---|---|---|---|---|
| H200 141GB | 3.59 / 4.59 | — | 4.54 | 4.50 (3.82 interruptible) | ~5.00 (1 offer) |
| H100 80GB | SXM 2.69 / 3.49 · PCIe 1.99 / 2.89 | SXM 4.29 · PCIe 3.29 | 3.95 | 4.68 (via Lambda) – 5.68 (via Nebius) | PCIe ~2.14 · SXM 2.07–6.01 (few offers) |
| **A100 80GB** | **SXM 1.39 / 1.59** · PCIe 1.19 / 1.59 | 1× not listed (40GB: 1.99) | 2.50 | 1× 80GB not listed (40GB: 2.19) | PCIe ~0.94 (1 offer) |
| L40S 48GB | 0.79 / 1.09 | — | 1.95 | 2.14 (via Nebius) – 3.54 (via AWS) | ~0.80 |
| RTX 5090 32GB | 0.69 / 0.99 | — | — | — | ~0.47 |
| RTX 4090 24GB | 0.34 / 0.74 | — | — | — | ~0.38–0.43 (30 offers) |

Sources: [RunPod pricing](https://www.runpod.io/pricing) (page "updated 27 Sep 2026"; its rendered table shows higher rates for some GPUs, e.g. A100 SXM $1.79, so the plan budgets at $1.79) · [Lambda pricing](https://lambda.ai/pricing) · [Modal pricing](https://modal.com/pricing) (Starter plan includes $30/month free compute) · Lightning AI public accelerator catalog (`lightning.ai/v1/core/accelerators?cloudProvider=MACHINE|NEBIUS|LAMBDA_LABS|AWS`) · Vast.ai public offer search (`console.vast.ai/api/v0/bundles`, verified on-demand hosts).

## Cost plan

**Choice: RunPod A100 80GB SXM, budgeted at $1.79/h** (highest RunPod figure seen; Secure is listed at $1.59, Community at $1.39).

Why this GPU:
- **Debugging is billed by the hour, not by speed.** Compat and setup favour a low hourly rate.
- **Training cost per step is about even with H100.** H100 is ~1.7–2× faster for small-model GRPO and ~1.9× the price. H100 stays the fallback when the deadline needs speed.
- **24 GB cards (4090, Scenario B's original pick) don't fit the deadline.** They run at about half A100 speed: 6 runs ≈ 54 h, more than the 40 h training window. Qwen3.5's ~248k vocabulary also makes logits memory tight for full FT plus colocated vLLM at 24 GB. RTX 5090 (Blackwell) adds kernel-build risk for `causal-conv1d` and vLLM. L40S is ~0.45× A100 speed for similar cost per step.
- **80 GB of headroom** removes most out-of-memory fallbacks.

**Step-time basis ⚠:** the main plan estimated ~4 h per 2B run and ~2 h per 0.8B run (300 steps) on one H200, unvalidated. A100 has about 1/3 of H200's bf16 compute and ~0.4× its memory bandwidth. Small-model GRPO is mostly generation, so assume **~2.25× slower**: **0.8B ≈ 4.5 h per 300-step run** (≈ 54 s/step); 2B ≈ 9 h. Replace with measured numbers after §0.4.

| Phase | Assumption ⚠ | GPU-h | Cost @ $1.79 |
|---|---|---|---|
| Setup, compat, dry runs | Kernel build, vLLM, resume test, 10-step dry runs; debugging burns time (capped) | 5.0 | $8.95 |
| Toy LR sweep + benchmarks | 3 LRs × 100 toy steps; 20 real-length steps; eval throughput; LoRA timing | 2.5 | $4.48 |
| Baseline eval | Diagnostic (10 train templates × 50 × 8) + formal base (250 × 4), 0.8B only | 1.0 | $1.79 |
| Training, seeds 1–2 | 4 runs × 4.5 h | 18.0 | $32.22 |
| Checkpoint evals, seeds 1–2 | 4 runs × 3 checkpoints (100, 200, final) × 0.25 h | 3.0 | $5.37 |
| Storage | 50 GB network volume for ~1 week + stopped-pod disk | — | $2.00 |
| **Committed subtotal** | | **29.5** | **$54.81** |
| Seed 3 pair (conditional) | 2 runs × 4.5 h + 6 evals × 0.25 h | 10.5 | $18.80 |
| **Planned total** | | **40.0** | **$73.61** |
| **Contingency (not allocated)** | 26% of cap | — | **$26.39** |
| **Hard cap** | | | **$100.00** |

At the listed Secure rate ($1.59) the planned total is ~$65.60; at the Community rate ($1.39) it is ~$57.60. The extra goes to contingency, not to scope.

**Optional lever:** Modal's Starter plan includes $30/month of free compute (≈12 A100-80GB hours). If the account has it, checkpoint evals can run there at $0 net. Not counted in this plan.

## Spend controls

- **Prepaid credit is the hard cap.** Disable auto-pay. Load $60 on Monday; top up only at a gate and log it; total loaded never exceeds $100.
- **Every job has a `timeout`** of about 1.5× its estimate, then stops the pod: `timeout <T> <cmd>; runpodctl stop pod $RUNPOD_POD_ID`.
- **Every pod has a dead-man switch** started at boot: `(sleep <session max>; runpodctl stop pod $RUNPOD_POD_ID) &`.
- **Training runs as one queue on one pod** (runs and their evals back to back), which stops the pod when the queue ends. No idle GPU while I sleep.
- **Terminate, don't just stop**, a pod after each session. Keep only the network volume (env, `causal-conv1d` wheel, HF cache). Also copy the built wheel to the Mac so a new datacenter doesn't mean a rebuild.
- **Artifacts leave the pod as soon as they exist.** Eval weights and eval JSONL go to a private Hugging Face Hub repo, then to the Mac.
- **Recompute the cost table** with measured step time after §0.4. If the planned total is over $75, cut steps first (min 200), then seed 3, before touching seeds 1–2.

**Stop rules (cumulative actual spend, from the Spend ledger)**

| When | If spend ≥ | Then |
|---|---|---|
| Compat not passed (§0.3) | $9 | Stop debugging Qwen3.5 → switch to Qwen3-0.6B (no custom kernels) |
| End of P0 + P1 GPU work | $15 | Skip the LoRA timing and extra benchmarks; go with what is measured |
| Training launch (G-Wed) | $25 | Seed 3 off; cut steps so a run ≤ 3.5 h |
| Before G-core | $55 | Intermediate evals off (final checkpoint only); seed 3 off |
| Before G-core | $65 | Finish the pair that is running; start nothing new; H1/H2 on the seeds available |
| Any time | $75 | No new training launches; only evals of finished final checkpoints |
| Any time | $90 | All GPU work stops; download artifacts; terminate pods |

**Pair rule:** start a seed's robust run only if the remaining planned budget covers both members of the pair and their evals.

## Spend ledger

| Date | Session | GPU | GPU-h | $ | Cumulative $ | Note |
|---|---|---|---|---|---|---|
| | | | | | 0.00 | |

---

## Run matrix

All on one A100 80GB, run in sequence (or two at once, see §0.4 pair packing). Each robust/leaky pair shares config, seed, start weights and step count.

| Tier | IDs | Reward | Seed | Est. time ⚠ | Est. cost (train + 3 evals) |
|---|---|---|---|---|---|
| 1 — never cut | `0.8b-rob-s1`, `0.8b-leak-s1` | robust, leaky | 1 | 2 × 4.5 h | $18.80 |
| 2 — committed | `0.8b-rob-s2`, `0.8b-leak-s2` | robust, leaky | 2 | 2 × 4.5 h | $18.80 |
| 3 — if Stop rules allow | `0.8b-rob-s3`, `0.8b-leak-s3` | robust, leaky | 3 | 2 × 4.5 h | $18.80 |
| Exploratory (H5-lite) | `2b-leak-s1` | leaky | 1 | ~9 h | ~$17; only from money below the $75 line after tier 3, and if wall-clock remains |

## Timeline

```
Mon 12      CPU: scaffold · SPEC · metrics · PREREG       GPU: compat · toy sweep · benchmarks   🚦 G-Mon
Tue 13      CPU: simulator · graders · tasks · data       GPU (evening, ≤1 h): dry runs + resume 🚦 G-Tue
Wed 14 AM   CPU: acceptance                                GPU (≤1 h): baseline + formal base     🚦 G-Wed
            ── training queue on one pod (stops itself) ──────────────────────────────
Wed 14:00   rob-s1 → eval → leak-s1 → eval                 ~10.5 h  → Thu ~00:30
Thu 00:30   rob-s2 → eval → leak-s2 → eval                 ~10.5 h  → Thu ~11:00   🚦 G-core
Thu 11:00   [seed 3 if allowed]                            ~10.5 h  → Thu ~21:30
Thu 21:30   buffer: one rerun, or H5-lite if money and time remain   → Fri 06:00
Thu         compare.py fills tables, provisional verdicts as results land
Fri 16      06:00 training stop · final eval · review · clean-clone · 12:00 done  🚦 G-Fri
```

---

## P0 — Monday 12 Oct: setup

### 0.1 Compute and account (me)
- [ ] RunPod account (personal), auto-pay off, $60 prepaid
- [ ] 50 GB network volume in a datacenter with A100 80GB stock
- [ ] Pod check prints GPU name, memory, CUDA available; dead-man switch tested once (pod stops itself)
- [ ] Private Hugging Face Hub repo for eval weights and eval outputs
- [ ] Spend ledger row added after each session

### 0.2 Repo scaffold (agent, CPU)
- [ ] `git init`, MIT `LICENSE`, `.gitignore` (`data/`, `runs/`, `.env`; keep manifest + split IDs)
- [ ] Layout from `INSTRUCTIONS.md` (`src/envlab/{sim,tasks,verify}`, `train/`, `eval/`, `reports/`, `tests/`, `configs/`)
- [ ] `uv` project, Python 3.14, pinned via `uv.lock`; torch, transformers, trl, datasets, vllm in a Linux-only `gpu` extra
- [ ] `ruff` + `pytest`; one trivial passing test; GitHub Actions CI runs both
- [ ] Done when: CI green on `main`

(A scaffold for this already exists on the repo's `main` branch and may be copied over.)

### 0.3 Qwen3.5 compatibility — full training path (agent, GPU, ≤ $9)
- [ ] Qwen3.5-0.8B loads as a text-only causal LM (no vision tower)
- [ ] Forward + backward, bf16, gradient checkpointing on
- [ ] Hybrid-attention kernels active (record which: `____`); no silent slow fallback (a slow fallback inflates every GPU hour)
- [ ] vLLM colocated generation works; weight sync trainer → vLLM works (generations change after an update)
- [ ] Save → resume restores model, optimizer, scheduler, RNG, trainer state
- [ ] Thinking mode disabled cleanly via chat template
- [ ] Native tool-call format identified: `____` (feeds SPEC prompt format and `parse.py`)
- [ ] Decision rule:
  - vLLM fails → HF generate for **training and eval**; model unchanged; eval estimates ×2–3 (recompute cost table)
  - Kernels fail or fall back: `causal-conv1d` ships **no PyPI wheels** → build from source on the pod (≤ 45 min; save the wheel to the volume and the Mac) → **retry on Python 3.12** (upstream prebuilt wheels are more likely to exist for 3.12) → only then switch model. `flash-linear-attention` is pure Python/Triton and fine on 3.14
  - Still failing, or spend hits $9 → **Qwen3-0.6B** (dense attention, no custom kernels)
  - Log any switch in Change log + Decisions

### 0.4 Toy GRPO + benchmarks (agent, GPU) — `train/smoke_toy.py`
- [ ] Task: two-number addition, reward = exact answer
- [ ] Main model, full FT, 100 steps at each LR: `1e-6`, `3e-6`, `1e-5`; each with a 45 min timeout
- [ ] Logged per step: reward, dead-group share, entropy, completion length, step time
- [ ] Weights confirmed changed
- [ ] Step time + peak memory at real task lengths (20 steps) → update Cost plan and Run matrix
- [ ] Eval throughput (samples/s) at 250 × 4 → update eval estimate
- [ ] LoRA timing (r 16, 20 steps): switch to LoRA only if ≥30% cheaper per step (then LR range 1e-5 / 3e-5 / 1e-4)
- [ ] Optional **pair packing**: robust + leaky of one seed as two processes on the one GPU. Adopt if combined throughput ≥ 1.4× sequential and memory fits; it cuts training cost ~30%
- [ ] Done when: reward rises for ≥1 LR with no entropy collapse

### 0.5 Spec (agent drafts, I review, CPU) — `SPEC.md`
- [ ] Entities and fields; 6 tools (arguments, state effects, error cases, ledger postings); ledger balance rule
- [ ] 15 templates with oracle rules; a pre-specified harder variant per train template (too-easy branch only)
- [ ] Robust and leaky grader definitions; gates G1–G3; 4 broken-variant types
- [ ] Prompt format using the tool-call format from §0.3
- [ ] Done when: I sign off, no open TODOs

### 0.6 Metrics (agent, CPU) — `eval/metrics.py`
- [ ] True success, exploit rate, format-failure rate per sample
- [ ] Paired instance bootstrap: resample held-out instances, keeping compared conditions on the same draws; effect averaged over available seeds
- [ ] Per-seed effects; seed-pair + instance hierarchical bootstrap when n = 3 (secondary)
- [ ] H1, H2 verdict functions implementing §0.7 exactly; training-reward window helper
- [ ] Done when: unit tests pass on fake data with known answers

### 0.7 Pre-registration (me) — `PREREG.md`, committed before any task training
- [ ] Held-out template IDs frozen (T3 per family)
- [ ] **Two separate evals:**
  - *Diagnostic baseline*: train templates, 50 instances × 8 samples, difficulty band only; never used in verdicts
  - *Formal eval*: 250 held-out × 4 samples, temp 0.7, both graders; base model, steps 100, 200, final
- [ ] **Statistics (scaled down for 2–3 seeds):** effect = mean across seeds; primary 95% CI from the paired instance bootstrap; **plus every seed must show the effect in the predicted direction.** State in the report that the instance CI does not capture seed-to-seed variance
- [ ] **H1** passes iff robust gain over base ≥ 10 pts **and** CI lower bound > 0 **and** gain > 0 in every seed
- [ ] **H2** passes iff all three hold:
  - training reward: leaky ≥ robust, mean over the **final 50 steps**, averaged over seeds
  - true success: leaky ≥ 10 pts below robust, CI excludes 0, same direction in every seed
  - exploit rate: leaky ≥ 10% **and** ≥ 2× base
- [ ] **H3** exploratory; **H4, H5** cut (H5-lite, if run, is descriptive only, n = 1)
- [ ] **Difficulty rule (frozen):** see §2.2; train-template diagnostics only; held-out never adapted
- [ ] **SFT rule:** if SFT warm-up is used, both pair members start from the same SFT checkpoint

### 0.8 End-of-day decisions (me)
- [ ] Fill Decisions rows 4–9 (model, FT vs LoRA, LR, step count, backend)
- [ ] Cost table recomputed with measured step time; Stop rules still hold

### 🚦 G-Mon
- [ ] CI green · compat passes (or fallback logged) · toy reward rises · benchmarks recorded · SPEC signed off · PREREG committed · Decisions filled · spend ≤ $15

---

## P1 — Tuesday 13 Oct: environment build (CPU, $0)

Three agents in parallel; I integrate and review.

### 1.1 Simulator and tools — `src/envlab/sim/`
- [ ] Entity dataclasses, in-memory store; 6 tools per SPEC, ledger postings on money moves
- [ ] Ledger balance check; deterministic state hash
- [ ] `parse.py`: one tool call; bad JSON / unknown tool → parse failure
- [ ] Unit tests: every tool, error cases, ledger, hash stability

### 1.2 Graders and gates — `src/envlab/verify/`
- [ ] Robust grader; leaky grader; G1, G2, G3
- [ ] Unit tests per family: correct, wrong amount, wrong record, side effect

### 1.3 Tasks and data — `src/envlab/tasks/`
- [ ] F1–F5 generators, 3 templates each, seeded; harder variants for train templates
- [ ] Distractors (2–4 per prompt); broken variants (25 × 4) + `manifest.json`
- [ ] Data build: 2,000 train, 250 held-out, 100 broken; `prompts.py` in native format
- [ ] Unit tests: seed reproducibility, no T3 in train

### 1.4 Training and eval plumbing
- [ ] TRL reward wrappers (robust, leaky; same interface); `configs/grpo.yaml`
- [ ] `train/grpo.py`: config + reward + seed; eval weights every 50 steps; rolling full recovery checkpoint (keep last 1)
- [ ] Resume tested from a recovery checkpoint **before** pruning is enabled
- [ ] `eval/heldout.py`: vLLM path + HF fallback; both graders; per-sample JSONL
- [ ] **Queue script**: runs the matrix in order, evals each run's checkpoints after it finishes, pushes artifacts to the Hub, applies the step-100 health check (below), per-job timeouts, stops the pod at the end
- [ ] GPU (evening, ≤ 1 h): 10-step dry run with each reward + resume test
- [ ] CI green

### 1.5 Reporting skeleton (agent, low priority)
- [ ] README: setup, data build, one training run, eval, how to rent and stop the GPU
- [ ] `reports/results.md` skeleton with H1/H2/H3 sections, a cost section and empty tables
- [ ] `eval/compare.py` stub that runs on fake eval JSONL

### 🚦 G-Tue
- [ ] Tests pass · data built · dry runs + resume work for both rewards · queue script tested (including self-stop) · if slipping, run §2 overnight

---

## P2 — Wednesday 14 Oct AM: acceptance and baseline

### 2.1 Acceptance (CPU) — `reports/acceptance.md`
- [ ] G1–G3 on every instance; all 100 broken variants rejected; ≥ 99% of valid instances pass
- [ ] Fix and regenerate if not; record failures

### 2.2 Diagnostic baseline (GPU, train templates only) — `reports/baseline.md`
- [ ] 0.8B: 50 instances × 8 samples per train template; per-template pass rate; count of the 10 in the 10–90% band
- [ ] **Apply frozen rule:**
  - ≥ 6 of 10 in band → proceed
  - **Too hard** → SFT warm-up (500 oracle completions, 1 epoch) → drop hardest template per family → Qwen3.5-2B as main (seed 3 off; recompute cost)
  - **Too easy** → pre-specified harder variants → Qwen3-0.6B
- [ ] Any adaptation logged; held-out unchanged

### 2.3 Formal base eval (GPU)
- [ ] 250 held-out × 4 samples, temp 0.7, both graders
- [ ] Base true success, exploit rate, format failures recorded (H1 and H2 baselines)

### 🚦 G-Wed (by 13:30)
- [ ] Acceptance passed · main model in band (after any adaptation) · formal base eval done · spend ≤ $25

---

## P3 — Wed 14:00 → Fri 06:00: training

### 3.1 Launch
- [ ] Prepaid balance covers tiers 1–2 plus evals; queue started at `____`; dead-man switch set to `____`
- [ ] Spot check: first eval checkpoint evaluated end to end and visible on the Hub

### 3.2 Health check at step 100 (automatic, in the queue script)
Healthy = mean reward steps 80–100 > steps 0–20 · dead-group share < 80% · no NaN/crash.
Unhealthy → halve LR, restart **both members of the pair** from step 0 (once); second failure → stop the pod and wait for me (SFT warm-up decision). Crash → resume from the recovery checkpoint.

| Run | Step-100 healthy? | Finished | Ckpts evaluated (100 / 200 / final) | Cost |
|---|---|---|---|---|
| 0.8b-rob-s1 | [ ] | [ ] | [ ] / [ ] / [ ] | |
| 0.8b-leak-s1 | [ ] | [ ] | [ ] / [ ] / [ ] | |
| 0.8b-rob-s2 | [ ] | [ ] | [ ] / [ ] / [ ] | |
| 0.8b-leak-s2 | [ ] | [ ] | [ ] / [ ] / [ ] | |
| 0.8b-rob-s3 | [ ] | [ ] | [ ] / [ ] / [ ] | |
| 0.8b-leak-s3 | [ ] | [ ] | [ ] / [ ] / [ ] | |

- [ ] 🚦 G-core (~Thu 11:00): seeds 1–2 finished and evaluated; spend logged
- [ ] Seed 3 go/no-go: go only if spend ≤ $55, measured run time ≤ 4.5 h and ≥ 12 h remain before Fri 06:00
- [ ] Recovery checkpoints pruned only after the run's final artifacts are on the Hub and checked

### 3.3 Live analysis (Thursday, CPU, as results land)
- [ ] `eval/compare.py` fills results tables (reward × seed, CIs) from eval JSONL
- [ ] Plot: training reward vs. true success over steps, per reward
- [ ] Per-family gain vs. base pass rate (H3)
- [ ] **Provisional verdicts** for H1, H2 after G-core, in `reports/results.md`
- [ ] Exploit gallery: 10–20 real leaky-pass / robust-fail samples, grouped by type
- [ ] Limits section drafted (seed count, instance-only CI, unvalidated estimates)

### 3.4 Buffer (Thu ~21:30 → Fri 06:00)
- [ ] Priority 1: rerun of any unhealthy/failed run
- [ ] Optional: evaluate checkpoints 50 / 150 / 250 (≈ $0.45 per checkpoint)
- [ ] Optional: H5-lite `2b-leak-s1` (only per Run matrix rule)

---

## P4 — Friday 16 Oct 06:00 → 12:00: final review and publish

- [ ] 06:00: training stopped; all artifacts on the Hub and the Mac; **pods terminated**; network volume deleted once artifacts are checked
- [ ] Final verdicts from complete data (H1, H2; H3 exploratory); sample sizes stated
- [ ] Cost section: GPU-hours and dollars per phase vs. this plan
- [ ] Field notes: step times, memory, what broke, grader vs. sim effort, early signs of reward hacking
- [ ] Clean-clone check: fresh clone → `uv sync` → tests → data build (GPU run not repeated)
- [ ] No secrets, no confidential material, all data synthetic
- [ ] Commits dated before deadline; repo made public (my call)

### 🚦 G-Fri (12:00)
- [ ] `reports/results.md` final · repo public-ready · field notes done · total spend ≤ $100

---

## Never cut / cut order

- **Never cut:** seed 1 robust + leaky pair, formal eval of base and final checkpoints, reproducible repo.
- **Cut order if behind (time or money):**
  1. H5-lite and optional checkpoint evals
  2. Seed 3
  3. Intermediate checkpoint evals (final only)
  4. Steps per run (min 200, same for every run)
  5. Seed 2 → H1/H2 reported with n = 1, stated as a limit
- Every cut updates the sample sizes in `reports/results.md`.

## Fallbacks (quick reference)

| Trigger | Response |
|---|---|
| A100 80GB unavailable | RunPod Community → another RunPod datacenter → Vast.ai verified A100 80GB → H100 (fewer hours, similar cost per step; recompute cost table) |
| vLLM fails | HF generate for training **and** eval; same model; evals final-only if eval cost doubles |
| Qwen3.5 kernels missing/slow | Build `causal-conv1d` from source → Python 3.12 → Qwen3-0.6B |
| Compat spend hits $9 | Qwen3-0.6B |
| Toy reward flat at all LRs | Check reward + parser → widen LR range → LoRA |
| Main model too hard | SFT warm-up → drop hardest template per family → Qwen3.5-2B (seed 3 off) |
| Main model too easy | Harder train variants → Qwen3-0.6B |
| Measured run time > 4.5 h | Cut steps (min 200) → try pair packing → seed 3 off |
| Out of memory | Prompts/step 8 → 4 · completion 128 → 96 |
| Unhealthy at step 100 | Halve LR, restart both pair members; second failure → SFT warm-up |
| Run crashes | Resume from the recovery checkpoint |
| Pod preempted or lost | New pod from the network volume; resume from the latest checkpoint on the volume or Hub |
| No exploit in leaky runs | Report as finding |
| Tuesday slips | Run §2 overnight; slip eats the Thursday buffer, then seed 3 |

---

## Change log

| Date | Change | Why | In Decisions? |
|---|---|---|---|
| 2026-10-12 | Forked from 2-node plan for $100 budget | Owner: run on a personal $100 cloud budget | [x] 2 |
| 2026-10-12 | Compute → 1 × A100 80GB on RunPod; prices checked across 5 providers | Cheapest per useful step that meets the deadline | [x] 3 |
| 2026-10-12 | Main model → Qwen3.5-0.8B, full FT; size ladder and H5 cut | 2B costs ~2× per run; ladder doesn't fit | [x] 5, 6, 11 |
| 2026-10-12 | Seeds → 2 committed + 1 conditional; paired instance bootstrap + per-seed agreement | Budget | [x] 10 |
| 2026-10-12 | Formal eval at steps 100, 200, final only; spend caps, timeouts, dead-man switch, stop rules added | Budget | [x] 14 |
| 2026-10-12 | Decisions and agent rules folded into this file | Branch holds only this file and `INSTRUCTIONS.md` | — |
