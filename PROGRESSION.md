# PROGRESSION — execution log

**This file is the authoritative execution plan.** `INSTRUCTIONS.md` is the original plan, kept unchanged. Anything not overridden in **Plan deltas** is inherited from it as-is (see **Inherited unchanged**).

**Update rules**

- Tick a box the moment its *done-when* is met; append `— YYYY-MM-DD HH:MM, note` if useful.
- Never delete a task: mark it `[~]` (cut) or `[>]` (moved/replaced) with a one-line reason.
- Any change to scope, model, config, schedule or statistics → row in **Change log**; if it touches a locked decision, also in `DECISIONS.md`.
- Keep **Status** current.

**Legend:** `[ ]` todo · `[x]` done · `[~]` cut · `[>]` moved/replaced · `[!]` blocked · 🚦 gate (must pass before the next phase)

**Naming:** **main model** = the model the core experiment (H1, H2) runs on — currently **Qwen3.5-2B**. Run IDs use the actual size (`2b-…`); if the main model changes, IDs change with it.

---

## Status

| | |
|---|---|
| **Today** | Mon 12 Oct 2026 |
| **Phase** | P0 — Setup |
| **Deadline** | **Fri 16 Oct, 12:00** (training cutoff Fri 06:00) |
| **Compute** | 2 × 8 H200 (Lightning AI) · studio `rl-env-lab` in `[teamspace]` (CPU now → H200 when GPU scripts land) |
| **Next gate** | 🚦 G-Mon (end of today) |
| **Blockers** | — |

---

## Plan deltas vs `INSTRUCTIONS.md`

| Area | Original | Now |
|---|---|---|
| Deadline | Sun 18 Oct | **Fri 16 Oct 12:00** |
| Compute | Scenario A (3 nodes) or B (1 GPU) | **2 H200 nodes** |
| Main model | Qwen3-0.6B (LoRA) | **Qwen3.5-2B**; fallbacks in §Fallbacks |
| Size ladder | 0.6B / 1.7B / 4B | **Qwen3.5 0.8B / 2B / 4B** |
| Weights updated | LoRA (full allowed for 4B+) | **Full fine-tune, all sizes** (confirm Mon) |
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
| Reporting | Sat–Sun | **Skeleton Tue–Wed, live tables Thu, final review Fri** |
| Python | 3.11 | **3.14** (Linux wheels confirmed for torch 2.13, vLLM 0.31, DeepSpeed 0.19, TRL 1.15) |
| GPU deps | Always installed | **`gpu` extra, Linux only**; CI and local dev run without it |

## Inherited unchanged from `INSTRUCTIONS.md`

Environment design (5 entities, 6 tools, 5 families × 3 templates) · T1/T2 train, T3 held out, frozen Monday · one tool call per task in native format · binary reward from simulator state · robust/leaky grader definitions · exploit definition · quality gates G1–G3 and acceptance bar · 2,000 / 250 / 100 data split and broken-variant types · GRPO config (8 gens × 8 prompts, prompt 1,536 / completion 128 tokens, temp 1.0, beta 0, bf16, grad checkpointing, seed list) · step-100 health definition · `uv`, `pytest`, CI · MIT, synthetic data · guardrails (personal accounts, no confidential material, `.env` never committed, timeouts and spend caps).

---

## GPU allocation

Each training run does its own generation (colocated vLLM) unless noted.

| Window | Node 1 | Node 2 |
|---|---|---|
| Mon–Wed AM | Toy runs, compat checks, benchmarks, baseline eval | Same (split work across nodes) |
| Wed 14 → 18 | GPU 0–5: six 2B runs (1 GPU each) · GPU 6–7: eval | GPU 0–5: six 0.8B runs · GPU 6–7: eval overflow |
| Wed 18 → Thu 02 | GPU 0–3: **4B P2 robust** · GPU 4–5: spare · GPU 6–7: eval | GPU 0–3: 4B P1 robust · GPU 4–7: 4B P1 leaky |
| Thu 02 → 10 | GPU 0–3: **4B P2 leaky** · GPU 4–5: spare · GPU 6–7: eval | GPU 0–3: 4B P3 robust · GPU 4–7: 4B P3 leaky |
| Thu 10 → Fri 06 | Reruns · optional ablations · eval | Reruns · optional ablations |

**4B run layout:** 4 GPUs = 1 vLLM server + 3 training (ZeRO-3). Keep this until Monday's benchmark justifies a change. Under HF fallback: 4 GPUs training, in-process generation.

**4B P2** runs robust then leaky in sequence on Node 1. It is still a matched pair: same config, seed, start weights and step count.

## Run matrix (18 runs)

| Group | Size | Reward | Seeds | GPUs/run | Est. time ⚠ | Node | IDs |
|---|---|---|---|---|---|---|---|
| Core | 2B | robust + leaky | 1, 2, 3 | 1 | ~4 h | 1 | `2b-{rob,leak}-s{1,2,3}` |
| Ladder | 0.8B | robust + leaky | 1, 2, 3 | 1 | ~2 h | 2 | `0.8b-{rob,leak}-s{1,2,3}` |
| Ladder | 4B | robust + leaky | 1, 2, 3 | 4 | ~8 h | 2 (P1, P3), 1 (P2) | `4b-{rob,leak}-s{1,2,3}` |

⚠ All times are unvalidated estimates; the 4B estimate in particular is benchmarked Monday (§0.4). Parameter count alone doesn't set throughput.

**Common budget across sizes (for H5):** same step count, prompts per step, generations per prompt, completion length and LR schedule shape. GPU count per run may differ; the per-step data does not.

## Timeline

```
Mon 12      setup · compat checks · toy GRPO + benchmarks · SPEC · PREREG       🚦 G-Mon
Tue 13      simulator · graders · tasks · integration · README + report skeleton 🚦 G-Tue
Wed 14 AM   acceptance · diagnostic baseline · formal base eval                  🚦 G-Wed
            ── training ─────────────────────────────────────────────────────
              Wed 14-18     Wed 18-02         Thu 02-10         Thu 10 → Fri 06
Node 1       [2B ×6]       [4B P2 rob]       [4B P2 leak]      ( reruns / ablations )
             [ eval ═══════════════ every checkpoint as it lands ════════════ ]
Node 2       [0.8B ×6]     [4B P1 rob+leak]  [4B P3 rob+leak]  ( reruns / ablations )
                  ▲                                ▲
          🚦 G-core ~Wed 18:00         🚦 G-ladder ~Thu 10:00 → ~20 h buffer
Thu 16      compare.py fills tables, plots, provisional verdicts as results land
Fri 16      06:00 training stop · final eval · review · clean-clone · 12:00 done 🚦 G-Fri
```

---

## P0 — Monday 12 Oct: setup

### 0.1 Compute and access (me)
- [ ] Written confirmation of H200 access (2 nodes, through Fri 16 Oct)
- [ ] Written confirmation: personal grant, no Lightning IP claim
- [x] Lightning studio `rl-env-lab` created in `[teamspace]`, started on CPU — 2026-10-12
- [x] Owner: `gh auth login` (codexceed) on the studio; clone repo; `uv sync --group dev` — 2026-10-12; agent drives studio via `lightning_sdk` `Studio.run`
- [x] Studio auto-sleep disabled (machines stay attached through dips) — 2026-10-12, verified via SDK
- [ ] Switch studio to H200 (1× for §0.3/§0.4, then ×8) when GPU scripts land
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
- [ ] Loads as text-only causal LM (no vision tower) for 0.8B / 2B / 4B
- [ ] Forward + backward pass, bf16, gradient checkpointing on
- [ ] Optimized kernels for hybrid attention active (record which: `____`); no silent slow fallback
- [ ] Generation works: vLLM colocated (1 GPU) and vLLM server (4B layout)
- [ ] Weight sync trainer → vLLM works (generations change after an update)
- [ ] Save → resume restores model, optimizer, scheduler, RNG, trainer state; loss continues smoothly
- [ ] Thinking mode disabled cleanly via chat template
- [ ] Native tool-call format identified: **XML** `<tool_call><function=NAME><parameter=ARG>…` from HF chat templates (owner approved 2026-10-12); confirm with real tokenizer on GPU
- [ ] Memory and throughput recorded per size (feeds §0.4)
- [ ] Decision rule:
  - vLLM fails → HF generate for **training and eval**; model family unchanged
  - Hybrid-attention kernels fail or fall back (e.g. `causal-conv1d`, which ships no PyPI wheels; `flash-linear-attention` is pure-Python and fine on 3.14) → build from source on the node, then **retry on Python 3.12** before any model switch
  - Training path still fails after the kernel steps above (load/backward/resume/kernels) → switch to Qwen3 family (1.7B main; 0.6B / 4B ladder)
  - Log either in Change log + `DECISIONS.md`

### 0.4 Toy GRPO + benchmarks (agent) — `train/smoke_toy.py`
- [ ] Task: two-number addition, reward = exact answer
- [ ] Main model full FT, 100 steps at each LR: `1e-6`, `3e-6`, `1e-5`
- [ ] Logged per step: reward, dead-group share, entropy, completion length, step time
- [ ] Weights confirmed changed
- [ ] Step time + peak memory benchmarked at real task lengths for 0.8B (1 GPU), 2B (1 GPU), 4B (4 GPU layout) → update Run matrix
- [ ] Eval throughput benchmarked (samples/s on 2 GPUs) → confirm eval keeps pace with checkpoint arrival
- [ ] Done when: reward rises for ≥1 LR with no entropy collapse

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

### 0.8 End-of-day decisions (me) — `DECISIONS.md`
- [ ] Model family + main model
- [ ] Full FT vs LoRA
- [ ] Learning rate (frozen)
- [ ] Step count (common across sizes; fits 4B estimate)
- [ ] Generation backend for training and eval
- [ ] 4B GPU layout (keep 1 gen + 3 train unless benchmark says otherwise)

### 🚦 G-Mon
- [ ] CI green · compat checks pass (or fallback logged) · toy reward rises · benchmarks recorded · SPEC signed off · PREREG committed · DECISIONS filled

---

## P1 — Tuesday 13 Oct: environment build

Three agents in parallel; I integrate and review.

### 1.1 Simulator and tools — `src/envlab/sim/`
- [x] Entity dataclasses, in-memory store
- [x] 6 tools per SPEC, ledger postings on money moves
- [x] Ledger balance check; deterministic state hash
- [x] `parse.py`: one tool call; bad JSON / unknown tool → parse failure
- [x] Unit tests: every tool, error cases, ledger, hash stability — 2026-10-12, 525 sim tests, merged (#5)

### 1.2 Graders and gates — `src/envlab/verify/`
- [ ] Robust grader; leaky grader
- [ ] G1, G2, G3
- [ ] Unit tests per family: correct, wrong amount, wrong record, side effect

### 1.3 Tasks and data — `src/envlab/tasks/`
- [ ] F1–F5 generators, 3 templates each, seeded; harder variants for train templates
- [ ] Distractors (2–4 per prompt)
- [ ] Broken variants (25 × 4) + `manifest.json`
- [ ] Data build: 2,000 train, 250 held-out, 100 broken
- [ ] `prompts.py` in native format
- [ ] Unit tests: seed reproducibility, no T3 in train

### 1.4 Training and eval plumbing
- [ ] TRL reward wrappers (robust, leaky; same interface)
- [ ] `configs/grpo.yaml` + per-size overrides for GPU layout only
- [ ] `train/grpo.py`: config + reward + seed; saves **eval weights every 50 steps** and a **rolling full recovery checkpoint** (model, optimizer, scheduler, RNG, trainer state; keep last 2)
- [ ] Resume tested from a recovery checkpoint **before** any pruning is enabled
- [ ] Rolling recovery pruning **during training** (keep last 2; delete an older one only once a newer one has `COMPLETE`; never touch `eval/`), via `train/checkpoints.py`; enabled only after resume is verified. Remaining recovery checkpoints deleted after the final-artifact integrity check
- [ ] `eval/heldout.py`: vLLM path + HF fallback; both graders; per-sample JSONL
- [x] Eval watcher: picks up new eval checkpoints, evaluates, marks done; never deletes final artifacts — 2026-10-12, `eval/watcher.py`, priority final → every 100 → rest; layout in `train/LAYOUT.md`
- [x] **Coordinator** (`train/coordinator.py` or job script): — 2026-10-12, `configs/allocation.yaml`, 38 tests
  - launches runs per GPU allocation
  - per-run timeout (run ends, node stays up)
  - never releases a machine by default: when all training, eval and transfers are done it writes an idle marker ("safe to release"); at the hard deadline it stops runs only
  - actual shutdown only with explicit opt-in flag + configured command (owner's call)
- [ ] 10-step dry run of `train/grpo.py` with each reward on 2B; resume tested
- [ ] CI green

### 1.5 Reporting skeleton (agent, low priority)
- [x] README: setup, data build, one training run, eval
- [x] `reports/results.md` skeleton with H1/H2/H5/H3 sections and empty tables
- [x] `eval/compare.py` stub that runs on fake eval JSONL — 2026-10-12, full tables/verdicts/plots; `make report-fake` passes H1/H2/H5 on fake data

### 🚦 G-Tue
- [ ] Tests pass · data built · dry runs + resume work for both rewards · coordinator tested · if slipping, run §2 overnight

---

## P2 — Wednesday 14 Oct AM: acceptance and baseline

### 2.1 Acceptance — `reports/acceptance.md`
- [ ] G1–G3 on every instance
- [ ] All 100 broken variants rejected; ≥99% of valid instances pass
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

### 🚦 G-Wed (by 13:30)
- [ ] Acceptance passed · main model in band (after any adaptation) · formal base eval done

---

## P3 — Wed 14:00 → Fri 06:00: training

### 3.1 Launch
- [ ] Coordinator launches per GPU allocation; start time `____`
- [ ] Eval watcher running on Node 1 GPU 6–7 (+ Node 2 GPU 6–7 during 0.8B)
- [ ] Spot check: first eval checkpoint of one run evaluated end-to-end

### 3.2 Health check at step 100 (each run)
Healthy = mean reward steps 80–100 > steps 0–20 · dead-group share < 80% · no NaN/crash.

| Run | Step-100 healthy? | Note |
|---|---|---|
| 2b-rob-s1 | [ ] | |
| 2b-rob-s2 | [ ] | |
| 2b-rob-s3 | [ ] | |
| 2b-leak-s1 | [ ] | |
| 2b-leak-s2 | [ ] | |
| 2b-leak-s3 | [ ] | |
| 0.8b-rob-s1 | [ ] | |
| 0.8b-rob-s2 | [ ] | |
| 0.8b-rob-s3 | [ ] | |
| 0.8b-leak-s1 | [ ] | |
| 0.8b-leak-s2 | [ ] | |
| 0.8b-leak-s3 | [ ] | |
| 4b-rob-s1 | [ ] | |
| 4b-leak-s1 | [ ] | |
| 4b-rob-s2 | [ ] | |
| 4b-leak-s2 | [ ] | |
| 4b-rob-s3 | [ ] | |
| 4b-leak-s3 | [ ] | |

Unhealthy → halve LR, restart from step 0, **both members of the pair**; second failure → SFT warm-up (per §0.7 SFT rule). Crash → resume from latest recovery checkpoint.

### 3.3 Completion

| Pair | Node / window | Finished (rob / leak) | All ckpts evaluated (rob / leak) |
|---|---|---|---|
| 2B P1 | N1 · Wed 14–18 | [ ] / [ ] | [ ] / [ ] |
| 2B P2 | N1 · Wed 14–18 | [ ] / [ ] | [ ] / [ ] |
| 2B P3 | N1 · Wed 14–18 | [ ] / [ ] | [ ] / [ ] |
| 0.8B P1 | N2 · Wed 14–18 | [ ] / [ ] | [ ] / [ ] |
| 0.8B P2 | N2 · Wed 14–18 | [ ] / [ ] | [ ] / [ ] |
| 0.8B P3 | N2 · Wed 14–18 | [ ] / [ ] | [ ] / [ ] |
| 4B P1 | N2 · Wed 18 → Thu 02 | [ ] / [ ] | [ ] / [ ] |
| 4B P2 | N1 · rob Wed 18 → Thu 02, leak Thu 02 → 10 | [ ] / [ ] | [ ] / [ ] |
| 4B P3 | N2 · Thu 02 → 10 | [ ] / [ ] | [ ] / [ ] |

- [ ] 🚦 G-core (~Wed 18:00): all six main-model runs finished and evaluated
- [ ] 🚦 G-ladder (~Thu 10:00): all 0.8B and 4B runs finished and evaluated
- [ ] Recovery checkpoints pruned only after the run's final artifacts pass integrity check

### 3.4 Live analysis (Thursday, as results land)
- [ ] `eval/compare.py` fills results tables (size × reward × seed, CIs) from eval JSONL
- [ ] Plots: training reward vs. true success over steps (per size, per reward)
- [ ] Plot: exploit rate vs. size (H5)
- [ ] Per-family gain vs. base pass rate (H3)
- [ ] **Provisional verdicts** for H1, H2 (after G-core) and H5 (after G-ladder) written into `reports/results.md`
- [ ] Exploit gallery: 10–20 real leaky-pass / robust-fail samples, grouped by type
- [ ] Limits section drafted

### 3.5 Spare blocks (Thu 10 → Fri 06, only after G-core)
- [ ] Priority 1: reruns of any unhealthy/failed run
- [ ] Optional: SFT vs. GRPO at matched budget (main model)
- [ ] Optional: LoRA vs. full (main model)
- [ ] Optional: widen the leak if no exploit appeared

---

## P4 — Friday 16 Oct 06:00 → 12:00: final review and publish

- [ ] 06:00: training stopped; coordinator confirms all eval + transfers done (idle marker); **owner releases machines** once certain they're not needed
- [ ] Final verdicts confirmed from complete data (H1, H2, H5; H3 exploratory); sample sizes stated
- [ ] Results, plots, gallery, limits reviewed
- [ ] Field notes: step times, memory, what broke, grader vs. sim effort, early signals of reward hacking
- [ ] Clean-clone check: fresh clone → `uv sync` → tests → data build → 10-step run
- [ ] No secrets, no confidential material, all data synthetic
- [ ] Commits dated before deadline; repo made public (my call)

### 🚦 G-Fri (12:00)
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
| Tuesday slips | Run §2 overnight; each 8 h slip consumes buffer first |

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
| 2026-10-12 | Forked $100-budget plan to orphan branch `docs/10-budget-100-plan` (1× A100, 0.8B, 2–3 seeds, H5 cut) | Owner hand-off to another machine | [ ] |
