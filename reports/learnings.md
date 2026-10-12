# Learnings

What this study has taught us so far, from results, issues and decisions. Each line is a lesson plus the evidence behind it. This is the seed of the final knowledge piece.

- **Living document.** Add a lesson when a result, incident or decision teaches something non-obvious; correct one when later data changes it (strike the old claim, say why). Rules: `AGENTS.md` → "Updating `reports/learnings.md`".
- **Labels:** *pre-registered* = from H1/H2/H5 verdicts; *training-time* = from `train_metrics.jsonl` on training tasks; *exploratory* = ADR 0005/0006 work. Never present an exploratory or training-time number as a verdict.
- **Last updated:** 2026-10-12 22:30 (block 1 done; 4B seeds 1, 3 done; v2 2B near done; no verdicts yet).

## 1. Reward hacking: what it looks like here

| Lesson | Evidence | Source |
|---|---|---|
| **Leaky training doesn't just allow exploits, it unlearns the real skill** | 4B on training tasks, steps 1–100 → 601–700: leaky run robust 0.63 → 0.27 with leaky ~1.0; robust run robust 0.74 → 0.95 | training-time |
| **The typical exploit is "right record, wrong amount"** | Traces: the model writes a number, then works out the correct value in its own note, too late | base evals |
| **Base models already "exploit" a lot** | Formal base exploit rate 0.8B 41%, 2B 48%, 4B 57%; "≥ 2× base" was unreachable, amended to base + 10 pts (ADR 0004) | base evals |
| **The robust/leaky split grows with size** | 2B at step 1,000, held-out: robust true 0.34–0.41 vs leaky 0.23–0.30 (base 0.22); exploit leaky 0.53–0.70 (base 0.48). 0.8B shows no clear split | held-out, provisional (per-seed, final checkpoint; verdicts need the paired bootstrap) |
| **Robust training still "exploits" on held-out tasks** | 2B robust runs end at held-out exploit 0.44–0.65: they never learned the new arithmetic, so their wrong answers still look right on the surface | held-out, provisional |
| **A leaky run can degenerate while scoring ~1.0** | 2b-leak-s1 ends with every answer hitting the 256-token cap (tool call first, then a runaway note), leaky reward 0.99, robust 0.26 | training-time |
| **Held-out failures mix hacking with missing skill** | Held-out F1/F3/F4 stay near 0% exact in every run: models learn tool, customer and status, not new arithmetic. Motivated the same-type eval (ADR 0005) | held-out |

## 2. Reasoning before answering is the strongest lever so far

| 2B base, 11 train types × 50 × 8 | True success | Exploit rate | Cut off |
|---|---|---|---|
| Thinking off | 0.31 | 0.43 | 2% |
| Thinking on, cap 512 | 0.35 | 0.02 | 54% |
| Thinking on, cap 1,024 | **0.61** | **0.11** | 18% |

- **Thinking on cuts base exploits ~4×.** Working the sum out before committing removes most wrong-amount answers. *(exploratory, base model; v2 tests whether it survives RL pressure)*
- **The answer budget matters as much as the switch.** At 512 the model was cut off, not unable: when reasoning finished, most types were 93–100% correct.
- **Under RL, thinking on transforms robust training, but leaky training still hacks.** v2 2B at steps 200–300, held-out: robust runs true 0.77–0.82, exploit 0.06–0.20 (v1 robust at step 1,000: 0.34–0.41 / 0.44–0.65). v2 leaky runs: true 0.30–0.36, exploit 0.55–0.66. *(exploratory, provisional)*
- **With thinking on, robust training first improves transfer, then overfits.** v2 2B robust, held-out true success peaks at 0.72–0.83 (steps 200–500) and falls to 0.52–0.60 by steps 700–800, while training-task robust reward stays 0.93–0.98. The loss is concentrated in held-out F1 (→ 0.11–0.45) and F3 (→ 0.03–0.26); F2/F4/F5 hold. Its failures keep the right tool, customer and status, so they count as "exploits" without any grader gaming. v1 robust stayed flat (0.33–0.41) from step 300 to 1,000. Verdict-style reporting stays on the final checkpoint; picking the peak on held-out data would be selecting on the test set. *(exploratory, held-out, 3 seeds; per-family evals are noisy at ~50 tasks)*
- **Leaky training shortens the reasoning.** v2 leaky answers average 260–400 tokens vs 530–545 for v2 robust: when the grader doesn't check the sum, the model stops working it out. *(exploratory, training-time)*
- **Run a sensitivity check before accepting a gate's no-go.** The gate as written said no-go at 512; one extra run at 1,024 showed why (ADR 0006).

## 3. Experiment design

- **Size budgets from real output lengths, per model size.** A 128-token cap cut off 26% of 4B held-out answers, understating 4B through truncation, not ability (ADR 0004 → 256).
- **Check the difficulty band before training.** 5 of 10 templates were in the 10–90% band; one learnable template (F1T5) was added, three candidates were too hard at 6–9%.
- **GRPO starves on saturated tasks.** 91–99% of groups gave no signal (all 8 answers scored the same): solved types, leaky runs at reward ~1.0, and types at ~0%. A health rule written before seeing the environment was wrong for it (PREREG #13).
- **Benchmark before planning.** Runs were 5–20× faster than estimated, which turned 300 steps into 1,000 (ADR 0003).

## 4. Training stability

| Lesson | Evidence |
|---|---|
| **The vLLM-vs-trainer logprob gap gives early warning of collapse** | Toy LR sweep: 3e-6 collapsed at step ~28, 1e-5 at step 3, each after gaps of 10+ nats; 1e-6 stable. Logged every step since (`sampler_logp_diff_*`). Later spikes: 4B up to 9.3 nats (4b-rob-s1's last 50 steps, where its robust reward slipped 0.95 → 0.85), v2 robust up to 6.5; none collapsed |
| **Small learning rates need fp32 master weights** | bf16 weights round 1e-6 updates away |
| **Qwen3.5 vision-language checkpoints break weight sync and resume** | Fixed by training from a text-only export |

## 5. Infrastructure

| Lesson | Evidence |
|---|---|
| **Checkpoint I/O, not compute, was the bottleneck** | 22 GB (2B) / 55 GB (4B) of optimizer state, 6 runs writing to one network disk: 10–15 min training stall per 50-step save. Snapshot-to-CPU + background write: 9–16 s pause |
| **Network-disk throughput swings with load** | The same 22 GB background write: 3.5 min on an idle node, 14 min on a busy one. A busy writer skips the next save rather than stalling training |
| **Plan eval GPUs alongside training GPUs** | A node with all 8 GPUs training (two 4B runs) left 24 checkpoints waiting for evals for hours |
| **Ad-hoc GPU jobs race the eval watcher** | Twice a one-off job collided with an eval that claimed the GPU seconds earlier. Pause the watcher, or add GPU claims |
| **Colocated vLLM beat a separate server for 4B** | 4 ZeRO-3 ranks, each with a colocated engine: 4.4 s/step vs 8.1 s for 1 server + 2 trainers |
| **Run identity hashes are fragile** | The data manifest's hash is part of every run's identity: editing it would have broken resumes and pairs, so new data went in a separate manifest. I/O-only settings are excluded from the hash |

## 6. Process (agents and humans)

- **Verify agents' claims; enforce, don't trust.** An agent reported "checks pass" with 4 lint errors left. Fix: a mandatory pre-commit hook. GitHub branch protection isn't available on a private free repo, so the local hook is the gate and CI is the backstop.
- **Pre-register, then date every amendment.** Four changes came after seeing data (H2 bar, answer cap, F1T5, v2 cap). Each is recorded with its reason and flagged in the report, not hidden.
- **Shared nodes have toolchain traps.** A package pulled in a second `uv` binary that child jobs would pick up; the GPU stack is installed outside the lock file, so syncs must be additive; node clocks run on UTC.

## Open questions for the final piece

- Does the held-out robust/leaky split hold at 4B, and does it grow with size (H5)?
- Does thinking-on training (v2) keep exploits low once RL pushes on the leaky grader?
- On same-type tasks, how much of the held-out "exploit" rate is hacking vs missing skill?
