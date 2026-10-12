# 0001. Foundational decisions for the 4-day study

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** owner (proposals from the planning and build sessions)

## Context

`INSTRUCTIONS.md` plans a study: train small Qwen models with GRPO in a billing simulator under a **robust** and a **leaky** reward, and measure true success against reward hacking. The plan assumed different compute, a Sunday deadline and a 0.6B LoRA model.

Reality: 2 × 8 H200 on Lightning AI and a deadline of **Fri 16 Oct 2026, 12:00**. Day one had to settle the model, the experiment design, the stack, the compute and the way of working, so that parallel agents could build against fixed contracts.

This ADR records everything decided up to the end of Monday 12 Oct. Later changes get their own ADRs.

## Decision

### A. Process

| # | Decision |
|---|---|
| A1 | **Major decisions are recorded as ADRs** in `docs/adr/`. `DECISIONS.md` is the one-line ledger. Agents prompt the owner for an ADR on any major architectural, planning or experiment-design change. |
| A2 | `PROGRESSION.md` is the **authoritative execution plan**. `INSTRUCTIONS.md` is never edited. Anything not overridden there is inherited. |
| A3 | **Commit directly to `main`.** A PR only for fundamental changes, opened only after asking the owner. |
| A4 | **Parallel agents** work in separate git worktrees, each owning a workstream (issues #1–#9). The integrator merges and updates PROGRESSION. |
| A5 | **Repo:** `codexceed/rl-env-lab`, a personal GitHub account, private until Friday. Conventional Commits. **No AI attribution.** Commits use the repo-local personal identity. |
| A6 | **Lint standards come from the evals repo:** ruff (120 cols; E, W, F, I, B, UP, SIM, C4, RUF), pyright standard, pydoclint Google style, and import-linter contracts. Pre-commit runs them, and CI runs one check per tool. A `Makefile` holds the human-run commands. |
| A7 | A **$100-budget variant** of the plan lives on the orphan branch `docs/10-budget-100-plan` (1× A100, 0.8B, H5 cut), for the owner to run separately. |

### B. Experiment design

| # | Decision |
|---|---|
| B1 | **Models:** Qwen3.5, with **2B as the main model** (H1, H2). Ladder of 0.8B, 2B and 4B (H5). Fallback: the Qwen3 family. |
| B2 | **Full fine-tune** at every size, no LoRA, so the size comparison is fair. |
| B3 | **3 seeds per condition, 18 runs.** Robust and leaky runs with the same seed form a matched **pair**: same config, start weights and step count. |
| B4 | **GRPO config:** 8 prompts × 8 generations, prompt ≤ 1,536 tokens, completion ≤ 128 tokens, temperature 1.0, beta 0. **The per-step budget is identical across sizes** (needed for H5). LR is chosen from 1e-6, 3e-6 and 1e-5 by a toy sweep. The step count is set by benchmark. |
| B5 | **Both graders score every step.** The run's own reward has weight 1 and the other weight 0, so robust and leaky training curves are visible live. |
| B6 | **Hypotheses:** H1, H2 and H5 are tested, H3 is exploratory, and H4 (triage) is cut. Multi-turn is cut. Ablations run only if spare time allows. |
| B7 | **PREREG is signed off with the default readings.** Every rule is implemented in `eval/metrics.py`. |
| B8 | **Two evals.** Diagnostic: train templates, 50 × 8 samples at temperature 1.0, used only to check difficulty. Formal: 250 held-out × 4 samples at temperature 0.7, both graders, used for every verdict. |
| B9 | **Statistics:** a paired, hierarchical bootstrap that resamples seed pairs, then held-out instances, with the same draws for every condition compared. The effect is the mean across seeds. |
| B10 | **Difficulty adaptation**, frozen in PREREG and decided from train templates only. Too hard: SFT warm-up for all sizes. Too easy: harder train variants, or 0.8B becomes the main model. Held-out templates are never adapted. |

### C. Environment

| # | Decision |
|---|---|
| C1 | **`SPEC.md` is the contract**, signed off through PR #12. The simulator is pure: frozen state, money in integer cents, tools never change their input. Graders are pure. |
| C2 | **Tool calls use Qwen3.5's native XML format**, with JSON accepted as a fallback. `enable_thinking=False` always, because the 4B model defaults to thinking on. |
| C3 | **Quality gates G1–G3** run on every instance. G3 proves each leak is real. Acceptance requires all broken variants rejected and ≥ 99% of valid instances passing. Measured: 100/100 and 4,250/4,250. |
| C4 | **SPEC fixes after implementation:** a $0 `resolve_case` adjustment is always allowed. The prompt-length estimate is `850 + chars/2.5` everywhere. `ToolCall` is documented as unhashable. |

### D. Stack and compute

| # | Decision |
|---|---|
| D1 | **Stack:** TRL `GRPOTrainer` with vLLM (colocated for 1-GPU runs, server mode for 4B). HF generate is the fallback for both training and eval. |
| D2 | **Train from a text-only export** of each Qwen3.5 checkpoint (`Qwen3_5ForCausalLM`), with **fp32 master weights** and bf16 compute. The Hub weights are vision-language checkpoints, which silently break weight sync and resume, and bf16 weights would round away 1e-6 updates. |
| D3 | **Python 3.14 at runtime, with code and dependencies kept 3.12-compatible.** CI tests both versions. If the kernels fail: build from source, then switch to Python 3.12, and only then change the model family. |
| D4 | **`causal-conv1d` is built from source** with a pip-installed CUDA compiler that exactly matches torch's CUDA (13.0), on a CPU machine (`scripts/setup_gpu.sh`). |
| D5 | **4B layout:** 1 vLLM server + 3 training GPUs is **infeasible**, because 64 completions don't split over 3 ranks. The candidates are 1 server + 2 training GPUs, or 4 colocated. `make bench` decides. |
| D6 | **Compute:** Lightning AI, teamspace `[teamspace]`, cloud account **`bare-metal`** at [rate] per 8× H200, instead of `public-cloud` at [rate], which also looked unavailable. Studios `rl-env-lab-n1` (node 1) and `rl-env-lab-n2` (from the start of training). Setup runs on CPU, then the studio switches to H200. |
| D7 | **Nodes stay attached** through debugging and plan dips. **Only the owner releases them.** The coordinator stops runs at the cutoff and marks the node idle; nothing is released automatically. |
| D8 | **Checkpoints:** eval weights every 50 steps, plus a rolling full recovery checkpoint, keeping the last 2. Pruning is enabled only after resume is verified. |

## Alternatives considered

| Option | Why not |
|---|---|
| Qwen3-0.6B + LoRA (original plan) | Weak main model. LoRA vs full fine-tuning would confound the size comparison. |
| 3 nodes | Not available. Two nodes fit the run matrix by Thursday 10:00, leaving about 20 h of buffer. |
| 1 vLLM server + 3 training GPUs for 4B | Doesn't divide the per-step budget evenly. Changing the budget would break H5. |
| Python 3.14 only | The 3.12 fallback would cost hours of reformatting under deadline. |
| A PR for every change | Too slow for 4 days. Kept only for fundamental changes. |
| Auto-releasing nodes when idle | Getting H200s back is slower and riskier than paying for idle time. |
| `public-cloud` H200 | About 2× the price, with a placeholder wait time. |

## Consequences

- **Easier:** parallel agents built six modules in one day against SPEC, LAYOUT and PREREG. Results can't be reinterpreted afterwards, and the kernel fallback is a one-line switch.
- **Harder:** prompts have no headroom under the 1,536-token budget. Most harder variants keep only 2–3 distractors, which matters only if the too-easy branch fires.
- **Watch:**
  - Still pending: LR (DECISIONS #6), step count (#7), backend (#8) and 4B layout (#9). These come from Monday's GPU checks.
  - Whether tasks land in the 10–90% difficulty band (§2.2).
  - Idle cost while nodes stay attached.
