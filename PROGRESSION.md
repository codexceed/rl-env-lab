# PROGRESSION — execution log

**This file is the authoritative execution plan.** `INSTRUCTIONS.md` is the original plan, kept unchanged. Anything not overridden in **Plan deltas** is inherited from it as-is (see **Inherited unchanged**).

**Update rules**

- Tick a box the moment its *done-when* is met; append `— YYYY-MM-DD, note` if useful.
- Never delete a task: mark it `[~]` (cut) or `[>]` (moved/replaced) with a one-line reason.
- Any change to scope, model, config, schedule or statistics → row in **Change log**; if it touches a locked decision, also in `DECISIONS.md`. Major architecture/planning/experiment-design changes get an ADR (`docs/adr/`, ask the owner first).
- Keep **Status** current.

**Legend:** `[ ]` todo · `[x]` done · `[~]` cut · `[>]` moved/replaced · `[!]` blocked · 🚦 gate (must pass before the next phase)

**Naming:** **main model** = the model the core experiment (H1, H2) runs on — currently **Qwen3.5-2B**. Run IDs use the actual size (`2b-…`); if the main model changes, IDs change with it.

---

## Status

| | |
|---|---|
| **Phase** | P3 — v1 matrix (18 runs) trained and evaluated; v2 2B done, v2 0.8B training on N1. Next: verdicts (`make report`), **project version v3** build (ADR 0007, §3.6) |
| **Deadline** | **Fri 16 Oct, 12:00** (training cutoff Fri 06:00) |
| **Compute** | 2 × 8 H200 (Lightning AI, `[teamspace]`, `bare-metal`) · node 1 `rl-env-lab-n1` on 8× H200 · node 2 `rl-env-lab-n2` created before training |
| **Next gate** | 🚦 v3 calibration gate (§3.6) |
| **Blockers** | — |

---

## Plan deltas vs `INSTRUCTIONS.md`

| Area | Original | Now |
|---|---|---|
| Deadline | Sun 18 Oct | **Fri 16 Oct 12:00** |
| Compute | Scenario A (3 nodes) or B (1 GPU) | **2 H200 nodes** |
| Main model | Qwen3-0.6B (LoRA) | **Qwen3.5-2B**; fallbacks in §Fallbacks |
| Size ladder | 0.6B / 1.7B / 4B | **Qwen3.5 0.8B / 2B / 4B** |
| Weights updated | LoRA (full allowed for 4B+) | **Full fine-tune, all sizes** |
| LR sweep | 1e-5 / 3e-5 / 1e-4 (LoRA range) | **1e-6 / 3e-6 / 1e-5** (full-FT range) |
| Seeds | 1 (B) / 3 (A) | **3 per condition** |
| Generation backend | vLLM colocated, else HF | **vLLM preferred** (colocated for 1-GPU runs, server for 4B); **HF generate fallback for training and eval** |
| Difficulty fallback | Too-hard branch only | **Too-hard and too-easy branches**, rule frozen in PREREG |
| Statistics | Per-run bootstrap CI | **Seed-pair + instance paired bootstrap**, rules in §0.7 |
| Checkpoints | Every 50 steps | **Eval weights every 50 + rolling full recovery checkpoint** |
| Node shutdown | Each job stops its instance | **No auto-release.** Coordinator stops *runs* at cutoff and reports "idle, safe to release"; machines stay attached through debugging/plan/code dips; **owner releases** only when certain they're no longer needed |
| Hypotheses | H1–H5 | **H1, H2, H5 tested; H3 exploratory; H4 cut** |
| Triage classifier | Sat | **Cut** |
| Multi-turn tasks | Stretch | **Cut** |
| Ablations | Stretch | **Optional, spare blocks only** |
| Reporting | Sat–Sun | **Skeleton during build, live tables as results land, final review before the deadline** |
| Python | 3.11 | **3.14** (Linux wheels confirmed for torch 2.13, vLLM 0.31, DeepSpeed 0.19, TRL 1.15) |
| GPU deps | Always installed | **`gpu` extra, Linux only**; CI and local dev run without it |

## Inherited unchanged from `INSTRUCTIONS.md`

Environment design (5 entities, 6 tools, 5 families × 3 templates) · T1/T2 train, T3 held out, frozen before training · one tool call per task in native format · binary reward from simulator state · robust/leaky grader definitions · exploit definition · quality gates G1–G3 and acceptance bar · 2,000 / 250 / 100 data split and broken-variant types · GRPO config (8 gens × 8 prompts, prompt 1,536 / completion 128 tokens, temp 1.0, beta 0, bf16, grad checkpointing, seed list) · step-100 health definition · `uv`, `pytest`, CI · MIT, synthetic data · guardrails (personal accounts, no confidential material, `.env` never committed, timeouts and spend caps).

---

## GPU allocation

Each training run does its own generation (colocated vLLM) unless noted.

| Block | Node 1 | Node 2 |
|---|---|---|
| Before training | Toy runs, compat checks, benchmarks, baseline eval | Same (split work across nodes) |
| 1 (~2 h) | GPU 0–5: six 2B runs (1 GPU each) · GPU 6–7: eval | GPU 0–5: six 0.8B runs · GPU 6–7: eval overflow |
| 2 (~1.5 h) | GPU 0–3: **4B P2 robust** · GPU 4–5: spare · GPU 6–7: eval | GPU 0–3: 4B P1 robust · GPU 4–7: 4B P1 leaky |
| 3 (~1.5 h) | GPU 0–3: **4B P2 leaky** · GPU 4–5: spare · GPU 6–7: eval | GPU 0–3: 4B P3 robust · GPU 4–7: 4B P3 leaky |
| Spare (until the cutoff) | Reruns · optional ablations · eval | Reruns · optional ablations |

**4B run layout** (DECISIONS #9): 4 ZeRO-3 ranks, each with colocated vLLM. 1 server + 3 training is infeasible (64 completions don't split over 3 ranks); 1 server + 2 training was 1.8× slower. Under HF fallback: 4 GPUs training, in-process generation.

**4B P2** runs robust (block 2) then leaky (block 3) on Node 1. It is still a matched pair: same config, seed, start weights and step count.

## Run matrix (18 runs)

| Group | Size | Reward | Seeds | GPUs/run | Time at 1,000 steps | Node | IDs |
|---|---|---|---|---|---|---|---|
| Core | 2B | robust + leaky | 1, 2, 3 | 1 | ~1.7 h | 1 | `2b-{rob,leak}-s{1,2,3}` |
| Ladder | 0.8B | robust + leaky | 1, 2, 3 | 1 | ~1.2 h | 2 | `0.8b-{rob,leak}-s{1,2,3}` |
| Ladder | 4B | robust + leaky | 1, 2, 3 | 4 | ~1.2 h | 2 (P1, P3), 1 (P2) | `4b-{rob,leak}-s{1,2,3}` |

Times are from the §0.4 benchmark at worst-case lengths (full 1,536-token prompts, full 128-token answers): 0.8B 4.4 s/step, 2B 6.2 s, 4B 4.4 s on 4 GPUs. Real runs should be a little faster.

**Common budget across sizes (for H5):** same step count, prompts per step, generations per prompt, completion length and LR schedule shape. GPU count per run may differ; the per-step data does not.

## Timeline

```
P0 setup        compat checks · toy GRPO + benchmarks · SPEC · PREREG          🚦 G-Setup
P1 build        simulator · graders · tasks · integration · README + report    🚦 G-Build
P2 baseline     acceptance · diagnostic baseline · formal base eval            🚦 G-Baseline
                ── training ──────────────────────────────────────────────────
                  block 1 (~2 h)   block 2 (~1.5 h)    block 3 (~1.5 h)   spare → cutoff
Node 1           [2B ×6]          [4B P2 rob]         [4B P2 leak]       ( reruns / ablations )
                 [ eval ═══════════════ every checkpoint as it lands ══════════════ ]
Node 2           [0.8B ×6]        [4B P1 rob+leak]    [4B P3 rob+leak]   ( reruns / ablations )
                      ▲                                      ▲
              🚦 G-core (end of block 1)        🚦 G-ladder (end of block 3) → large buffer
Live            compare.py fills tables, plots, provisional verdicts as results land
P4 final        Fri 06:00 cutoff · final eval · review · clean-clone · Fri 12:00 deadline 🚦 G-Final
```

---

## P0 — Setup

### 0.1 Compute and access (me)
- [ ] Written confirmation of H200 access (2 nodes, through Fri 16 Oct)
- [ ] Written confirmation: personal grant, no Lightning IP claim
- [x] Lightning studio `rl-env-lab` created in `[teamspace]`, started on CPU — 2026-10-12
- [x] Owner: `gh auth login` (codexceed) on the studio; clone repo; `uv sync --group dev` — 2026-10-12; agent drives studio via `lightning_sdk` `Studio.run`
- [x] Studio auto-sleep disabled (machines stay attached through dips) — 2026-10-12, verified via SDK
- [x] Switch studio to H200 (1× for §0.3/§0.4, then ×8) when GPU scripts land — 2026-10-12, n1 on 8× H200
- [x] Node 1 = studio `rl-env-lab-n1` on `bare-metal` (8× H200 `8x-H200`, [rate]); set up on CPU: gpu extra + causal-conv1d 1.7.0 built (pip CUDA 13.0 nvcc), 1,231 tests pass on node — 2026-10-12. Old `rl-env-lab` studio (public-prod, [rate] for 8× H200) stopped
- [ ] Node 2 = `rl-env-lab-n2` on `bare-metal`: create + CPU setup before training; H200 at training start
- [ ] Both nodes reachable; GPU check prints GPU name, memory, CUDA available on each
- [ ] Shared storage for checkpoints/eval outputs set up; path: `____`
- [ ] Storage measured after first checkpoint save (per-checkpoint size × planned count recorded here: `____`)

### 0.2 Repo scaffold (agent)
- [x] `git init`, MIT `LICENSE`, `.gitignore` (`data/`, `runs/`, `.env`; keep manifest + split IDs) — 2026-10-12, repo-local personal git identity
- [x] Layout from `INSTRUCTIONS.md` (`src/envlab/{sim,tasks,verify}`, `train/`, `eval/`, `reports/`, `tests/`, `configs/`) — 2026-10-12
- [x] `uv` project, Python 3.14, pinned via `uv.lock`: torch, transformers, trl, datasets, vllm, accelerate/deepspeed in `gpu` extra (peft only if LoRA kept) — 2026-10-12
- [x] `ruff` + `pytest`; one trivial passing test — 2026-10-12
- [ ] GitHub repo (private until Fri); Actions CI runs ruff + pytest
- [x] `DECISIONS.md` created (skeleton exists; fill as decisions land) — 2026-10-12
- [x] `AGENTS.md` (agent working rules) + `CLAUDE.md` pointer — 2026-10-12
- [ ] Done when: CI green on `main`

### 0.3 Qwen3.5 compatibility — full training path (agent, on GPU, pinned stack)
- [x] Loads as text-only causal LM (no vision tower) for 0.8B / 2B / 4B — 2026-10-12, compat passes on all 3 sizes
- [x] Forward + backward pass, bf16, gradient checkpointing on — 2026-10-12, 2B 15.2k train tok/s, 31 GB peak; 4B 7.6k tok/s, 41 GB
- [x] Optimized kernels for hybrid attention active (record which: `causal_conv1d` 1.7.0 source build + `fla` 0.5.2 chunk/fused-recurrent gated delta rule); no silent slow fallback — 2026-10-12
- [x] Generation works: vLLM colocated (1 GPU) and vLLM server (4B layout) — 2026-10-12, both modes in compat + bench
- [x] Weight sync trainer → vLLM works (generations change after an update) — 2026-10-12, colocated; logprob gap now logged per step (toy collapses at LR ≥ 3e-6 followed 10+ nat gaps)
- [x] Save → resume restores model, optimizer, scheduler, RNG, trainer state; loss continues smoothly — 2026-10-12, exact fingerprints after resume, all sizes (after fixing a vacuous test reward)
- [x] Thinking mode disabled cleanly via chat template — 2026-10-12, all sizes
- [x] Native tool-call format identified: **XML** `<tool_call><function=NAME><parameter=ARG>…` from HF chat templates (owner approved 2026-10-12); confirm with real tokenizer on GPU — 2026-10-12, confirmed with real tokenizers
- [x] Memory and throughput recorded per size (feeds §0.4) — 2026-10-12, `logs/checks/n1/compat/`
- [x] Decision rule: — 2026-10-12, no fallback needed
  - vLLM fails → HF generate for **training and eval**; model family unchanged
  - Hybrid-attention kernels fail or fall back (e.g. `causal-conv1d`, which ships no PyPI wheels; `flash-linear-attention` is pure-Python and fine on 3.14) → build from source on the node, then **retry on Python 3.12** before any model switch
  - Training path still fails after the kernel steps above (load/backward/resume/kernels) → switch to Qwen3 family (1.7B main; 0.6B / 4B ladder)
  - Log either in Change log + `DECISIONS.md`

### 0.4 Toy GRPO + benchmarks (agent) — `train/smoke_toy.py`
- [x] Task: two-number addition, reward = exact answer — 2026-10-12, 11-digit operands (5 digits too easy: base 0.92)
- [x] Main model full FT, 100 steps at each LR: `1e-6`, `3e-6`, `1e-5` — 2026-10-12, 2B
- [x] Logged per step: reward, dead-group share, entropy, completion length, step time — 2026-10-12, plus vLLM-vs-trainer logprob gap
- [x] Weights confirmed changed — 2026-10-12, 320/320 params at 1e-6
- [x] Step time + peak memory benchmarked at real task lengths for 0.8B (1 GPU), 2B (1 GPU), 4B (4 GPU layout) → update Run matrix — 2026-10-12, 4.4 / 6.2 / 4.4 s per step; 63 / 83 / 59 GB peak
- [x] Eval throughput benchmarked (samples/s on 2 GPUs) → confirm eval keeps pace with checkpoint arrival — 2026-10-12, 160 / 176 / 98 samples/s (~62–72 s per checkpoint); keeps pace for 4B, lags in block 1 (58 vs 70–97 ckpts/h) and catches up after; final checkpoints first
- [x] Done when: reward rises for ≥1 LR with no entropy collapse — 2026-10-12, 1e-6 passes (0.82→0.87); DECISIONS #6 = 1e-6

### 0.5 Spec (agent drafts, I review) — `SPEC.md`
- [x] Entities and fields
- [x] 6 tools: arguments, state effects, error cases, ledger postings
- [x] Ledger balance rule
- [x] 15 templates with oracle rules
- [x] **Pre-specified harder variant per train template** (e.g. more distractors, multi-line arithmetic), used only by the too-easy branch
- [x] Robust and leaky grader definitions
- [x] Quality gates G1–G3; 4 broken-variant types
- [x] Prompt format using the tool-call format from §0.3
- [x] Done when: I sign off, no open TODOs — 2026-10-12, PR #12 merged

- [x] SPEC follow-ups from implementation (2026-10-12, owner approved: 0 adjustment always allowed; chars/2.5 everywhere; ToolCall unhashable documented): `resolve_case` adjustment=0 on overpaid invoice rejected (literal rule); token estimate chars/3 (§14) vs chars/2.5 (§10); `ToolCall` unhashable (dict field)

### 0.6 Metrics (agent) — `eval/metrics.py`
- [x] True success, exploit rate, format-failure rate per sample
- [x] Paired, hierarchical bootstrap: resample **seed pairs**, then **held-out instances**, keeping compared conditions on the same draws
- [x] Effects reported as mean across seeds with 95% CI
- [x] H1, H2, H5 verdict functions implementing §0.7 rules exactly
- [x] Training-reward window helper (for H2)
- [x] Done when: unit tests pass on fake data with known answers — 2026-10-12, 33 tests, merged #11

### 0.7 Pre-registration (me) — `PREREG.md`, committed before any task training
- [x] Held-out template IDs frozen (T3 per family) — 2026-10-12, PREREG signed off with defaults; diagnostic temp 1.0
- [x] **Two separate evals:**
  - *Diagnostic baseline* — train templates, 50 instances × 8 samples, for difficulty band only; never used in verdicts
  - *Formal eval* — 250 held-out × 4 samples, temp 0.7, both graders; base model, every 50 steps, final
- [x] **Statistics:** effect = mean across 3 seeds; 95% CI from the §0.6 bootstrap
- [x] **H1** passes iff main-model robust gain over base ≥ 10 pts **and** CI lower bound > 0
- [x] **H2** passes iff all three hold:
  - training reward: leaky ≥ robust, mean over the **final 50 steps**, averaged over seeds
  - true success: leaky ≥ 10 pts below robust, CI excludes 0
  - exploit rate: leaky ≥ 10% **and** ≥ 2× base
- [x] **H5** passes iff final leaky exploit rate is ordered 0.8B < 2B < 4B **and** 4B − 0.8B CI excludes 0; requires 3 seeds per size, else H5 is reported as **exploratory/incomplete** with actual n
- [x] **H3** exploratory; **H4** cut
- [x] **Difficulty adaptation rule (frozen):** see §2.2; decided from train-template diagnostics only; one task distribution for all sizes; held-out templates never adapted
- [x] **SFT rule:** if SFT warm-up is used, the same recipe is applied to all sizes and both reward members start from the same SFT checkpoint; otherwise H5 is exploratory

### 0.8 Setup decisions (me) — `DECISIONS.md`
- [x] Model family + main model — 2026-10-12, Qwen3.5-2B (ADR 0001)
- [x] Full FT vs LoRA — 2026-10-12, full FT (ADR 0001)
- [x] Learning rate (frozen) — 2026-10-12, 1e-6
- [x] Step count (common across sizes; fits 4B estimate) — 2026-10-12, 1,000 (ADR 0003)
- [x] Generation backend for training and eval — 2026-10-12, vLLM colocated (ADR 0003)
- [x] 4B GPU layout (keep 1 gen + 3 train unless benchmark says otherwise) — 2026-10-12, 4 colocated (ADR 0003)

### 🚦 G-Setup
- [ ] CI green · compat checks pass (or fallback logged) · toy reward rises · benchmarks recorded · SPEC signed off · PREREG committed · DECISIONS filled

---

## P1 — Environment build

Three agents in parallel; I integrate and review.

### 1.1 Simulator and tools — `src/envlab/sim/`
- [x] Entity dataclasses, in-memory store
- [x] 6 tools per SPEC, ledger postings on money moves
- [x] Ledger balance check; deterministic state hash
- [x] `parse.py`: one tool call; bad JSON / unknown tool → parse failure
- [x] Unit tests: every tool, error cases, ledger, hash stability — 2026-10-12, 525 sim tests, merged (#5)

### 1.2 Graders and gates — `src/envlab/verify/`
- [x] Robust grader; leaky grader
- [x] G1, G2, G3
- [x] Unit tests per family: correct, wrong amount, wrong record, side effect — 2026-10-12, 196 verify tests on the real sim, merged (#6)

### 1.3 Tasks and data — `src/envlab/tasks/`
- [x] F1–F5 generators, 3 templates each, seeded; harder variants for train templates
- [x] Distractors (2–4 per prompt)
- [x] Broken variants (25 × 4) + `manifest.json`
- [x] Data build: 2,000 train, 250 held-out, 100 broken
- [x] `prompts.py` in native format
- [x] Unit tests: seed reproducibility, no T3 in train — 2026-10-12, merged (#7)
- [x] Real token counts checked with the Qwen3.5 tokenizer — 2026-10-12, all splits fit: max real 1,463 (train_hard), 1,430 (train), 1,462 (heldout) vs limit 1,536; 0 over. The chars/2.5 estimate overcounts (real = 0.92× estimate on average, ≤ 0.98× worst case), so data needs no rebuild
- [ ] Known limit: prompt budget (≤ 1,500 est. tokens, ≈ 1,465 real) leaves little headroom; most *hard* variants fit only 2–3 distractors (not 5–6). Only matters if the too-easy branch fires — then raise hard-variant budget or revise k before using them

### 1.4 Training and eval plumbing
- [x] TRL reward wrappers (robust, leaky; same interface) — 2026-10-12, both scored every step, weight 1/0 by run (#9)
- [x] `configs/grpo.yaml` + per-size overrides for GPU layout only — 2026-10-12 (#9)
- [x] `train/grpo.py`: config + reward + seed; saves **eval weights every 50 steps** and a **rolling full recovery checkpoint** (model, optimizer, scheduler, RNG, trainer state; keep last 2)
- [x] Resume tested from a recovery checkpoint **before** any pruning is enabled — 2026-10-12, 2B resumed from step 5: step 6 identical, rewards identical through 10
- [x] Rolling recovery pruning **during training** (keep last 2; delete an older one only once a newer one has `COMPLETE`; never touch `eval/`), via `train/checkpoints.py`; enabled only after resume is verified. Remaining recovery checkpoints deleted after the final-artifact integrity check — 2026-10-12, `pruning_enabled: true`
- [x] `eval/heldout.py`: vLLM path + HF fallback; both graders; per-sample JSONL — 2026-10-12, code only; GPU run pending (#9)
- [x] Eval watcher: picks up new eval checkpoints, evaluates, marks done; never deletes final artifacts — 2026-10-12, `eval/watcher.py`, priority final → every 100 → rest; layout in `train/LAYOUT.md`
- [x] **Coordinator** (`train/coordinator.py` or job script): — 2026-10-12, `configs/allocation.yaml`, 38 tests
  - launches runs per GPU allocation
  - per-run timeout (run ends, node stays up)
  - never releases a machine by default: when all training, eval and transfers are done it writes an idle marker ("safe to release"); at the hard deadline it stops runs only
  - actual shutdown only with explicit opt-in flag + configured command (owner's call)
- [x] 10-step dry run of `train/grpo.py` with each reward on 2B; resume tested — 2026-10-12, 2B robust + leaky and 4B robust (4 colocated GPUs), 0 grader errors, ~6 s/step; `make dry-run`
- [ ] CI green

### 1.5 Reporting skeleton (agent, low priority)
- [x] README: setup, data build, one training run, eval
- [x] `reports/results.md` skeleton with H1/H2/H5/H3 sections and empty tables
- [x] `eval/compare.py` stub that runs on fake eval JSONL — 2026-10-12, full tables/verdicts/plots; `make report-fake` passes H1/H2/H5 on fake data

### 🚦 G-Build
- [ ] Tests pass · data built · dry runs + resume work for both rewards · coordinator tested · if slipping, run §2 in parallel

---

## P2 — Acceptance and baseline

### 2.1 Acceptance — `reports/acceptance.md`
- [x] G1–G3 on every instance — 2026-10-12, 4,250 valid (train, train_hard, held-out) all pass
- [x] All 100 broken variants rejected; ≥99% of valid instances pass — 100/100, 100%
- [ ] Fix and regenerate if not; record failures

### 2.2 Diagnostic baseline (train templates only) — `reports/baseline.md`
- [ ] 0.8B, 2B, 4B: 50 instances × 8 samples per train template
- [ ] Per-template pass rate; count of the 10 train templates in the 10–90% band, per size
- [ ] **Apply frozen adaptation rule (main model):**
  - ≥6 of 10 in band → proceed
  - **Too hard** (most below 10%) → SFT warm-up (500 oracle completions, 1 epoch, all sizes) → drop hardest template per family → Qwen3-1.7B / larger
  - **Too easy** (most above 90%) → switch train templates to pre-specified harder variants (all sizes) → else make **0.8B the main model**
- [ ] Any adaptation logged; same task distribution for all sizes; held-out unchanged

### 2.3 Formal base eval
- [ ] 250 held-out × 4 samples, temp 0.7, both graders, for 0.8B, 2B, 4B
- [ ] Base true success, exploit rate, format failures recorded (H1 and H2 baselines)

### 🚦 G-Baseline
- [ ] Acceptance passed · main model in band (after any adaptation) · formal base eval done

---

## P3 — Training (until the Fri 06:00 cutoff)

### 3.1 Launch
- [ ] Coordinator launches per GPU allocation; start time `____`
- [ ] Eval watcher running on Node 1 GPU 6–7 (+ Node 2 GPU 6–7 during 0.8B)
- [ ] Spot check: first eval checkpoint of one run evaluated end-to-end

### 3.2 Health check at step 100 (each run)
Healthy = mean reward steps 80–100 > steps 0–20 · dead-group share < 80% · no NaN/crash.

| Run | Step-100 healthy? | Note |
|---|---|---|
| 2b-rob-s1 | [~] reward ↑, dead ≥ 80% | 0.46→0.54, dead 0.97; continued (PREREG #13) |
| 2b-rob-s2 | [~] reward ↑, dead ≥ 80% | 0.42→0.57, dead 0.94; continued (PREREG #13) |
| 2b-rob-s3 | [~] reward ↑, dead ≥ 80% | 0.48→0.62, dead 0.93; continued (PREREG #13) |
| 2b-leak-s1 | [~] reward ↑, dead ≥ 80% | 0.87→0.99, dead 0.98; continued (PREREG #13) |
| 2b-leak-s2 | [~] reward ↑, dead ≥ 80% | 0.90→0.98, dead 0.99; continued (PREREG #13) |
| 2b-leak-s3 | [~] reward ↑, dead ≥ 80% | 0.88→0.98, dead 0.97; continued (PREREG #13) |
| 0.8b-rob-s1 | [~] reward ↑, dead ≥ 80% | 0.27→0.59, dead 0.91; continued (PREREG #13) |
| 0.8b-rob-s2 | [~] reward ↑, dead ≥ 80% | 0.23→0.65, dead 0.93; continued (PREREG #13) |
| 0.8b-rob-s3 | [~] reward ↑, dead ≥ 80% | 0.29→0.33, dead 0.96 (slow learner); continued (PREREG #13) |
| 0.8b-leak-s1 | [~] reward ↑, dead ≥ 80% | 0.80→0.93, dead 0.99; continued (PREREG #13) |
| 0.8b-leak-s2 | [~] reward ↑, dead ≥ 80% | 0.80→0.98, dead 0.98; continued (PREREG #13) |
| 0.8b-leak-s3 | [~] reward ↑, dead ≥ 80% | 0.85→0.97, dead 0.97; continued (PREREG #13) |
| 4b-rob-s1 | [ ] | |
| 4b-leak-s1 | [ ] | |
| 4b-rob-s2 | [ ] | |
| 4b-leak-s2 | [ ] | |
| 4b-rob-s3 | [ ] | |
| 4b-leak-s3 | [ ] | |

**2026-10-12: not applied to block 1 — every run failed only on dead-group share (task saturation), see PREREG #13.** Unhealthy → halve LR, restart from step 0, **both members of the pair**; second failure → SFT warm-up (per §0.7 SFT rule). Crash → resume from latest recovery checkpoint.

### 3.3 Completion

| Pair | Node / block | Finished (rob / leak) | All ckpts evaluated (rob / leak) |
|---|---|---|---|
| 2B P1 | N1 · 1 | [x] / [x] | [x] / [x] (20 / 20) |
| 2B P2 | N1 · 1 | [x] / [x] | [x] / [x] (20 / 20) |
| 2B P3 | N1 · 1 | [x] / [x] | [x] / [x] (20 / 20) |
| 0.8B P1 | N2 · 1 | [x] / [x] | [x] / [x] (20 / 20) |
| 0.8B P2 | N2 · 1 | [x] / [x] | [x] / [x] (20 / 20) |
| 0.8B P3 | N2 · 1 | [x] / [x] | [x] / [x] (20 / 20) |
| 4B P1 | N2 · 2 | [x] / [x] | [x] / [x] (20 / 20) |
| 4B P2 | N2 · 4 (moved from N1, Change log 2026-10-12) | [x] / [x] | [x] / [x] (20 / 20) |
| 4B P3 | N2 · 3 | [x] / [x] | [x] / [x] (20 / 20) |

- [x] 🚦 G-core (end of block 1): all six main-model runs finished and evaluated — 2026-10-12 20:45, 1,000 steps each, 20 eval checkpoints each
- [x] 🚦 G-ladder (end of block 3): all 0.8B and 4B runs finished and evaluated — 2026-10-12 22:40, 4B evals 20 / 20 each (caught up on N2 after training)
- [ ] Recovery checkpoints pruned only after the run's final artifacts pass integrity check

### 3.4 Live analysis (as results land)
- [x] `eval/compare.py` fills results tables (size × reward × seed, CIs) from eval JSONL — 2026-10-13 00:00, data merged from both nodes; fixed a claim-marker bug that hid every eval
- [x] Plots: training reward vs. true success over steps (per size, per reward) — 2026-10-13, `reports/figures/`
- [x] Plot: exploit rate vs. size (H5) — 2026-10-13
- [x] Per-family gain vs. base pass rate (H3) — 2026-10-13, table in results.md
- [x] **Provisional verdicts** for H1, H2 (after G-core) and H5 (after G-ladder) written into `reports/results.md` — 2026-10-13 00:00, on complete data: H1 PASS, H2 FAIL (drop +9.8 vs 10 pts), H5 FAIL
- [ ] Exploit gallery: 10–20 real leaky-pass / robust-fail samples, grouped by type
- [ ] Limits section drafted

### 3.6 Project version v3: harder, larger task set (ADR 0007, exploratory)
- [x] ADR 0007 accepted by the owner — 2026-10-13 01:15, SPEC_v3.md signed off too
- [x] `SPEC_v3.md`: 15 train templates (A–C per family), 5 near + 5 far held-out, counts, gates — 2026-10-13 01:00, draft for owner sign-off
- [x] v3 generators, solvers, wrong-value calls, broken variants; `data/v3/` + its manifest; v1 `data/manifest.json` byte-identical — 2026-10-13 01:00, v1 rebuild sha256-identical (both manifests)
- [x] v3 acceptance: ≥ 99% of valid instances pass G1–G3, every broken instance fails a gate — 2026-10-13 01:00, 7,500 / 7,500 valid pass; 100 / 100 broken rejected
- [x] `--variant v3` (thinking on, cap 1,024, `data/v3`, prompt budget 2,048, context 3,072); eval reports near and far separately — 2026-10-13 00:40, tested on synthetic data; v1/v2 identity hashes pinned unchanged
- [x] 🚦 v3 calibration gate: thinking-on base diagnostic on v3 train split: 4B ≥ 9 / 15 in band and leaky pass ≤ 90%; 2B ≥ 6 / 15 in band (one tuning pass allowed) — 2026-10-13 01:27 UTC, on N1, no tuning: 4B **go** (11 / 15, true 0.58, leaky 0.75); 2B **go** (9 / 15, true 0.27, leaky 0.43, 41% cut off at the cap)
- [x] v3 dry run (4B, 12 steps): memory and step time at the longer prompts — 2026-10-13 01:33 UTC, 4B 23.6 s/step, 2B 15.0 s/step, no OOM, prompts 1,146–1,928 tokens; all checks ok (one over-strict check fixed)
- [ ] v3 runs: 4B × 6 and 2B × 6 (start by Wed 14 Oct evening) — started 2026-10-13 01:35 UTC on N1: 4B pairs side by side (rob GPU 0-3, leak GPU 4-7, seeds in sequence, ~6.6 h each), then 2B × 6; training ends ~Wed 03:00 UTC
- [ ] `reports/v3.md`: near vs far, robust vs leaky, labelled v3 exploratory

### 3.5 Spare blocks (after block 3 until the cutoff, only after G-core)
- [ ] Priority 1: reruns of any unhealthy/failed run
- [ ] Optional: SFT vs. GRPO at matched budget (main model)
- [ ] Optional: LoRA vs. full (main model)
- [ ] Optional: widen the leak if no exploit appeared

---

## P4 — Final review and publish (Fri 06:00 cutoff → 12:00 deadline)

- [ ] Cutoff: training stopped; coordinator confirms all eval + transfers done (idle marker); **owner releases machines** once certain they're not needed
- [ ] Final verdicts confirmed from complete data (H1, H2, H5; H3 exploratory); sample sizes stated
- [ ] Results, plots, gallery, limits reviewed
- [ ] Field notes: step times, memory, what broke, grader vs. sim effort, early signals of reward hacking
- [ ] Clean-clone check: fresh clone → `uv sync` → tests → data build → 10-step run
- [ ] No secrets, no confidential material, all data synthetic
- [ ] Commits dated before deadline; repo made public (my call)

### 🚦 G-Final (Fri 12:00 deadline)
- [ ] `reports/results.md` final · repo public-ready · field notes done

---

## Never cut / cut order

- **Never cut:** main-model robust and leaky runs (3 seeds), formal held-out eval, reproducible repo.
- **Cut order if behind:**
  1. Optional ablations
  2. 4B P3 → **H5 becomes exploratory** (n = 2 for 4B)
  3. 4B P2 → H5 exploratory (n = 1 for 4B)
  4. 0.8B seeds 2–3 → H5 exploratory
  5. Plots beyond the core three
- Every cut updates PREREG-facing sample sizes in `reports/results.md`.

## Fallbacks (quick reference)

| Trigger | Response |
|---|---|
| vLLM fails (either mode) | HF generate for training **and** eval; same model family |
| Qwen3.5 kernels missing/slow | Build `causal-conv1d` from source → Python 3.12 → then model switch |
| Qwen3.5 training path fails | Qwen3 family (1.7B main; 0.6B / 4B ladder) |
| Toy reward flat at all LRs | Check reward + parser → widen LR range → LoRA |
| Main model too hard | SFT warm-up (all sizes) → drop hardest template per family → larger model |
| Main model too easy | Harder train variants (all sizes) → 0.8B as main |
| 4B benchmark far over 8 h | Fewer common steps (all sizes) or move 4B to more GPUs; re-plan schedule |
| Out of memory | Prompts/step 8→4 · completion 128→96 · more GPUs per run (applied to all sizes for budget parity) |
| Unhealthy at step 100 | Halve LR, restart both pair members; second failure → SFT warm-up |
| Run crashes | Resume from latest recovery checkpoint |
| Eval falls behind | Add spare GPUs (N1 GPU 4–5) to eval; evaluate final + every 100 steps first |
| No exploit in leaky runs | Report as finding; optionally widen the leak in a spare block |
| Build slips | Run §2 in parallel; any slip consumes the buffer first |

---

## Change log

| Date | Change | Why | In `DECISIONS.md`? |
|---|---|---|---|
| 2026-10-12 | Deadline → Fri 16 Oct 12:00; compute → 2 H200 nodes | Availability | [ ] |
| 2026-10-12 | Main model → Qwen3.5-2B; ladder 0.8B/2B/4B; full fine-tune | Fits 1 GPU/run; same-family ladder | [ ] |
| 2026-10-12 | LR sweep → 1e-6 / 3e-6 / 1e-5 | Full-FT range, not LoRA | [ ] |
| 2026-10-12 | H4 triage, multi-turn cut; ablations optional | Deadline | [ ] |
| 2026-10-12 | Coordinator owns node shutdown; per-run timeouts only | Runs share nodes | [ ] |
| 2026-10-12 | Explicit GPU allocation; vLLM preferred, HF fallback for train + eval; vLLM failure ≠ model switch | Review | [ ] |
| 2026-10-12 | 4B P2 moved to Node 1 after core; ladder done ~Thu 10:00 | Node 1 idle after core | [ ] |
| 2026-10-12 | Too-easy branch added; adaptation rule frozen in PREREG, train-template diagnostics only | Review | [ ] |
| 2026-10-12 | Common per-step budget across sizes; SFT applied to all sizes; seed cuts downgrade H5 | H5 validity | [ ] |
| 2026-10-12 | Diagnostic vs. formal eval separated; paired seed+instance bootstrap; H1/H2/H5 rules defined | Review | [ ] |
| 2026-10-12 | Compat checks cover full training path; rolling recovery checkpoints; resume verified before pruning | Review | [ ] |
| 2026-10-12 | Reporting moved earlier: skeleton Tue, live tables + provisional verdicts Thu | Friday overload | [ ] |
| 2026-10-12 | Python 3.11 → 3.14; GPU stack moved to Linux-only `gpu` extra | Owner request; wheels verified | [ ] |
| 2026-10-12 | Kernel failure → source build, then Python 3.12, before switching model family | `causal-conv1d` has no PyPI wheels | [ ] |
| 2026-10-12 | Compute on Lightning studio `rl-env-lab` (`[teamspace]`); repo cloned there | Owner request | [ ] |
| 2026-10-12 | No automatic node release; machines stay attached through dips; owner releases | Owner request | [x] #13 |
| 2026-10-12 | PREREG signed off with default interpretations; diagnostic baseline at temp 1.0, robust grader decides band | Owner | [ ] |
| 2026-10-12 | Recovery checkpoints: rolling keep-last-2 during training (not only after the run), gated on verified resume | ~25–30 GB per 2B full-state ckpt; post-run-only pruning would need TBs | [ ] |
| 2026-10-12 | Tool calls in Qwen3.5 native XML format (INSTRUCTIONS assumed JSON); parser also accepts JSON; `enable_thinking=False` always | Native chat template; owner approved | [ ] |
| 2026-10-12 | 4B layout 1 vLLM server + 3 train is infeasible (64 completions/step not divisible by 3 train ranks); benchmark 1+2 and 4-GPU colocated instead | TRL batch constraint, found in #3 | [ ] |
| 2026-10-12 | Train from a clean text-only export of each Qwen3.5 checkpoint; fp32 master weights + bf16 autocast | Hub repos are VL checkpoints (weight-sync/resume break); bf16 weights lose 1e-6 updates | [ ] |
| 2026-10-12 | Code + deps kept Python 3.12-compatible (runtime 3.14); CI tests both | Kernel fallback stays a one-line switch | [ ] |
| 2026-10-12 | Nodes on `bare-metal` ([rate] per 8× H200) instead of `public-cloud` ([rate], likely unavailable); setup on CPU before switching | Price + availability; owner approved | [ ] |
| 2026-10-12 | ARCHITECTURE.md + docs/adr/ added; ADR 0001 records all decisions to date; major changes now get ADRs | Owner request (progressive disclosure) | [x] #14 |
| 2026-10-12 | Forked $100-budget plan to orphan branch `docs/10-budget-100-plan` (1× A100, 0.8B, 2–3 seeds, H5 cut) | Owner hand-off to another machine | [ ] |
| 2026-10-12 | Run logs committed under `logs/` and mirrored to Lightning Experiments by `train/logsync.py` (ADR 0002) | Owner: live tracking + public evidence | [x] #19 |
| 2026-10-12 | Toy GRPO task 5 → 11 digits: base 2B scored 0.92 at 5 digits (83% groups dead), 0.69 at 11 (38%) | 5-digit sweep could not show reward rising | [ ] |
| 2026-10-12 | LR = 1e-6 (DECISIONS #6); pair git-commit differences are noted, not excluded | Toy sweep; owner | [x] #6 |
| 2026-10-12 | Day/time labels removed from the plan; phases, gates (G-Setup, G-Build, G-Baseline, G-Final) and training blocks are sequence-based. Only the Fri 06:00 cutoff and 12:00 deadline stay fixed | Ahead of schedule; labels were stale | [ ] |
| 2026-10-12 | Steps 300 → 1,000 (all sizes); backend vLLM colocated; 4B on 4 colocated GPUs; run timeouts and block lengths cut to match the benchmark (ADR 0003) | Benchmark: runs 5–20× faster than estimated | [x] #7–#9 |
| 2026-10-12 | Step-100 health rule not applied to block 1 (all runs: reward rising, dead groups 91–99%); runs continue (PREREG #13) | Owner: dead groups are task saturation, not instability | [ ] |
| 2026-10-12 | Exploratory follow-up proposed: same-type eval + thinking-on v2 (ADR 0005, Proposed) | Held-out failures mix hacking with missing arithmetic | [x] #20 |
| 2026-10-12 | Recovery checkpoints asynchronous (snapshot to CPU, background write, skip a save while one is in flight); 4B recovery every 100 steps (eval stays 50). Block 1 runs keep the old code | Synchronous saves paused training ~10 min per 50 steps; 4B saves are 55 GB | [ ] |
| 2026-10-12 | ADR 0005 accepted (PR #16) | Owner merged | [x] #20 |
| 2026-10-12 | v2 moved up: n1 runs the v2 2B round (ADR 0005) right after its 2B block; n1's 4B pair (P2) moved to the end of n2's 4B queues. On a no-go the pair returns to n1 | v2 results ~3 h sooner; 4B still ends well before the cutoff | [ ] |
| 2026-10-12 | Async recovery verified on GPU (dry run: 2B + 4B resume match; save pause 9–16 s vs ~3.5 min); 4B runs started on n2 | Gate for 4B | [x] |
| 2026-10-12 | v2 go with answer cap 1,024 (512 failed the gate); v2 2B ×6 added to node1 after its 2B block (ADR 0006) | Thinking-on base check | [x] #22 |
| 2026-10-12 | v2 extended to 4B and 0.8B (ADR 0005 "extend if the effect holds": v2 2B robust held-out true ~0.8 vs v1 ~0.4). n1: v2 4B seeds 1, 3 after v2 2B; n2: v2 4B seed 2 + v2 0.8B ×6 after the 4B v1 runs and their evals. ~10 h, both nodes | Owner chose the full v2 ladder | [ ] #20 |
| 2026-10-12 | v2 4B dropped: thinking-on base diagnostic no-go (4B base true 0.87 on train types, 4 of 11 in band, leaky pass 0.98; 12-step 4B v2 dry run fit, ~20 s/step). v2 0.8B passed (9 of 11 in band, after fixing the go/no-go to skip templates absent from the thinking-on run) and moves to n1 GPU 0-5 now; n2 keeps only 4B v1 evals. Owner: build a harder, larger task set as **project version v3** (ADR 0007, Accepted 2026-10-13), trained with v2 settings at 4B and 2B | Gate result + owner | [ ] #23 |
| 2026-10-13 | v3 on N1 only: N2 restarted for the v3 gate but its network disk served the venv too slowly after the restart (bare `import torch` > 400 s); gate and dry run moved to N1 and passed; N2 stopped. All 12 v3 runs queued on N1 (~25 h). Owner pre-approved launch on a passing gate | v3 gate go | [x] |
| 2026-10-13 | v1 verdicts on complete data (18 runs, 363 checkpoints, pairs matched): H1 PASS (+14.8 pts), H2 FAIL (true-success drop +9.8 vs ≥ 10 pts; parts 1 and 3 met), H5 FAIL (leaky exploit 72.7 / 63.9 / 64.5% by size) | `make report` | [x] |
