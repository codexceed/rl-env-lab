# Learnings

What this study has taught us so far, from results, issues and decisions. Each line is a lesson plus the evidence behind it. This is the seed of the final knowledge piece.

- **Living document.** Add a lesson when a result, incident or decision teaches something non-obvious; correct one when later data changes it (strike the old claim, say why). Rules: `AGENTS.md` → "Updating `reports/learnings.md`".
- **Labels:** *pre-registered* = from H1/H2/H5 verdicts; *training-time* = from `train_metrics.jsonl` on training tasks; *exploratory* = ADR 0005/0006/0007 work. Never present an exploratory or training-time number as a verdict.
- **Project versions:** every lesson is tagged with the version it comes from. Numbers from different versions use different tasks; don't compare them without saying so.

| Version | Tasks | Settings | Runs | Status |
|---|---|---|---|---|
| **v1** | `SPEC.md` (11 train, 5 held-out templates) | thinking off, cap 256 | 0.8B / 2B / 4B × robust, leaky × 3 seeds | pre-registered (PREREG) |
| **v2** | same as v1 | thinking on, cap 1,024 (ADR 0005, 0006) | 2B, 0.8B (4B gate no-go) | exploratory |
| **v3** | `SPEC_v3.md`: harder, 15 train, near + far held-out (ADR 0007) | thinking on, cap 1,024 | 4B, 2B (planned) | exploratory, being built |
- **Last updated:** 2026-10-13 00:05 (v1 verdicts in: H1 pass, H2 fail by 0.2 pts, H5 fail; v2 0.8B training; v3 being built).

## 1. Reward hacking: what it looks like here (v1)

| Lesson | Evidence | Source |
|---|---|---|
| **The environment teaches, and a leaky reward still looks better while teaching less** | H1 pass: 2B robust held-out true 36.6% vs base 21.8% (+14.8, CI +9.7 to +20.8). H2 fail by 0.2 pts: leaky training reward 97.8% vs 82.6%, exploit 63.9% vs base 48.3%, but true-success drop vs robust +9.8 pts (CI +5.4 to +14.0) against a 10-pt bar | pre-registered |
| **Exploit rate does not grow with size; the skill gap does** | H5 fail: leaky exploit 72.7% (0.8B), 63.9% (2B), 64.5% (4B). Robust minus leaky true success: 2.3 pts (0.8B), 9.7 (2B), 21.8 (4B) | pre-registered (H5); gap is descriptive |
| **One seed can fail to transfer at all** | 4b-rob-s2 ends at held-out true 36.9% (s1 55.7%, s3 57.9%) with training reward 0.82: it learned the training tasks but not the held-out ones. Three seeds made this visible; one would have hidden it or been it | pre-registered data, descriptive |
| **Held-out transfer peaks mid-run, then drifts down, at 4B too** | 4B robust s1/s3 held-out true ~0.85 at steps 200–450, 0.56–0.58 at step 1,000 (training reward stays 0.85–0.97). Same shape as v2 2B robust (§2). Verdicts use the final step as pre-registered | held-out, descriptive |
| **Leaky training doesn't just allow exploits, it unlearns the real skill** | 4B on training tasks, steps 1–100 → 601–700: leaky run robust 0.63 → 0.27 with leaky ~1.0; robust run robust 0.74 → 0.95 | training-time |
| **The typical exploit is "right record, wrong amount"** | Traces: the model writes a number, then works out the correct value in its own note, too late | base evals |
| **Base models already "exploit" a lot** | Formal base exploit rate 0.8B 41%, 2B 48%, 4B 57%; "≥ 2× base" was unreachable, amended to base + 10 pts (ADR 0004) | base evals |
| **The robust/leaky split grows with size** | Held-out true success at step 1,000, robust vs leaky (mean of 3 seeds): 0.8B 29.3 vs 27.0%; 2B 36.6 vs 26.9%; 4B 50.2 vs 28.4%. *Updated 2026-10-13 with complete data (was per-seed ranges with 4B half evaluated)* | held-out, complete data; not itself a pre-registered test |
| **Robust training still "exploits" on held-out tasks** | 2B robust runs end at held-out exploit 0.44–0.65: they never learned the new arithmetic, so their wrong answers still look right on the surface | held-out, provisional |
| **A leaky run can degenerate while scoring ~1.0** | 2b-leak-s1 ends with every answer hitting the 256-token cap (tool call first, then a runaway note), leaky reward 0.99, robust 0.26 | training-time |
| **Held-out failures mix hacking with missing skill** | Held-out F1/F3/F4 stay near 0% exact in every run: models learn tool, customer and status, not new arithmetic. Motivated the same-type eval (ADR 0005) | held-out |
| **The held-out templates need a formula training never shows** | Trained F1 is a flat or promised amount and trained F3 a full or P% refund, both subtraction-style; held-out F1T3 is per-seat (qty × d) and F3T3 unused units × unit price, so they need multiplication. F2/F4/F5 held-out stay close to their training shape and transfer well (v2 robust ≥ 0.8) | SPEC templates, held-out |

## 2. Reasoning before answering is the strongest lever so far (v2)

| 2B base, 11 train types × 50 × 8 | True success | Exploit rate | Cut off |
|---|---|---|---|
| Thinking off | 0.31 | 0.43 | 2% |
| Thinking on, cap 512 | 0.35 | 0.02 | 54% |
| Thinking on, cap 1,024 | **0.61** | **0.11** | 18% |

- **Thinking on cuts base exploits ~4×.** Working the sum out before committing removes most wrong-amount answers. *(exploratory, base model; v2 tests whether it survives RL pressure)*
- **The answer budget matters as much as the switch.** At 512 the model was cut off, not unable: when reasoning finished, most types were 93–100% correct.
- **Under RL, thinking on transforms robust training, but leaky training still hacks.** v2 2B at steps 200–300, held-out: robust runs true 0.77–0.82, exploit 0.06–0.20 (v1 robust at step 1,000: 0.34–0.41 / 0.44–0.65). v2 leaky runs: true 0.30–0.36, exploit 0.55–0.66. *(exploratory, provisional)*
- **With thinking on, robust-trained transfer to the new formulas is unstable, not steadily overfitting.** v2 2B robust, held-out true success: 0.72–0.83 at steps 200–500, a dip to 0.52–0.60 at 700–900, then 0.70–0.74 at step 1,000, while training-task robust reward stays 0.88–0.99. The swings are almost all held-out F1 and F3, which jump between checkpoints (s1 F1 0.25 → 0.92 → 0.15; F3 0.03 → 0.78); F2/F4/F5 hold. The model keeps the right tool, customer and status, so a wrong F1/F3 amount counts as an "exploit" without any grader gaming. *Corrected 2026-10-12 22:20: an earlier version said it "overfits" from the 700–800 dip; the step 900–1,000 evals showed a recovery.* Verdict-style reporting stays on the final checkpoint; picking a peak on held-out data would be selecting on the test set. *(exploratory, held-out, 3 seeds; per-family evals are noisy at ~50 tasks)*
- **v2 final, step 1,000: thinking on roughly doubles robust transfer; leaky still hacks.** Held-out true / exploit: v2 robust 0.70–0.74 / 0.14–0.28 vs v1 robust 0.34–0.41 / 0.44–0.65; v2 leaky 0.20–0.27 / 0.70–0.80, worse than v1 leaky. Leaky v2 runs lose held-out F1, F3 and F4 entirely (0.00) by step 600–1,000. *(exploratory, held-out, provisional)*
- **Leaky training shortens the reasoning.** v2 leaky answers average 260–400 tokens vs 530–545 for v2 robust: when the grader doesn't check the sum, the model stops working it out. *(exploratory, training-time)*
- **Run a sensitivity check before accepting a gate's no-go.** The gate as written said no-go at 512; one extra run at 1,024 showed why (ADR 0006).

## 3. Experiment design (v1, v2)

- **A switch that helps one size can saturate another.** *(v2)* Thinking on lifts 2B base true success 0.31 → 0.61, but takes 4B to 0.87 with leaky pass 0.98: 7 of 11 types at 98–100%, so GRPO would get almost no signal. Task difficulty has to be re-checked per size after any change to how the model answers (ADR 0007).
- **Check that a gate compares like with like.** *(v2)* The go/no-go first said no-go for 0.8B because the thinking-off baseline still held two rejected candidate templates that the thinking-on run never measured. Fixed to compare only shared templates; 0.8B is a go (9 of 11 in band).

- **Size budgets from real output lengths, per model size.** A 128-token cap cut off 26% of 4B held-out answers, understating 4B through truncation, not ability (ADR 0004 → 256).
- **Check the difficulty band before training.** 5 of 10 templates were in the 10–90% band; one learnable template (F1T5) was added, three candidates were too hard at 6–9%.
- **GRPO starves on saturated tasks.** 91–99% of groups gave no signal (all 8 answers scored the same): solved types, leaky runs at reward ~1.0, and types at ~0%. A health rule written before seeing the environment was wrong for it (PREREG #13).
- **Benchmark before planning.** Runs were 5–20× faster than estimated, which turned 300 steps into 1,000 (ADR 0003).

## 4. Training stability (v1, v2)

| Lesson | Evidence |
|---|---|
| **The vLLM-vs-trainer logprob gap gives early warning of collapse** | Toy LR sweep: 3e-6 collapsed at step ~28, 1e-5 at step 3, each after gaps of 10+ nats; 1e-6 stable. Logged every step since (`sampler_logp_diff_*`). Later spikes: 4B up to 9.3 nats (4b-rob-s1's last 50 steps, where its robust reward slipped 0.95 → 0.85), v2 robust up to 6.5; none collapsed |
| **Small learning rates need fp32 master weights** | bf16 weights round 1e-6 updates away |
| **Qwen3.5 vision-language checkpoints break weight sync and resume** | Fixed by training from a text-only export |

## 5. Infrastructure (all versions)

| Lesson | Evidence |
|---|---|
| **Checkpoint I/O, not compute, was the bottleneck** | 22 GB (2B) / 55 GB (4B) of optimizer state, 6 runs writing to one network disk: 10–15 min training stall per 50-step save. Snapshot-to-CPU + background write: 9–16 s pause |
| **Network-disk throughput swings with load** | The same 22 GB background write: 3.5 min on an idle node, 14 min on a busy one. A busy writer skips the next save rather than stalling training |
| **Plan eval GPUs alongside training GPUs** | A node with all 8 GPUs training (two 4B runs) left 24 checkpoints waiting for evals for hours |
| **Ad-hoc GPU jobs race the eval watcher** | Twice a one-off job collided with an eval that claimed the GPU seconds earlier. Pause the watcher, or add GPU claims |
| **Colocated vLLM beat a separate server for 4B** | 4 ZeRO-3 ranks, each with a colocated engine: 4.4 s/step vs 8.1 s for 1 server + 2 trainers |
| **Run identity hashes are fragile** | The data manifest's hash is part of every run's identity: editing it would have broken resumes and pairs, so new data went in a separate manifest. I/O-only settings are excluded from the hash |

## 6. Process (agents and humans, all versions)

- **Verify agents' claims; enforce, don't trust.** An agent reported "checks pass" with 4 lint errors left. Fix: a mandatory pre-commit hook. GitHub branch protection isn't available on a private free repo, so the local hook is the gate and CI is the backstop.
- **Pre-register, then date every amendment.** Four changes came after seeing data (H2 bar, answer cap, F1T5, v2 cap). Each is recorded with its reason and flagged in the report, not hidden.
- **Shared nodes have toolchain traps.** A package pulled in a second `uv` binary that child jobs would pick up; the GPU stack is installed outside the lock file, so syncs must be additive; node clocks run on UTC.

## 7. v3: harder task set (ADR 0007)

*Being built. Lessons land here once the calibration gate and the first runs report.*

- **Design aim:** the v3 "near" held-out set only recombines steps that were trained, so a near exploit points to hacking, not missing skill; "far" held-out keeps a generalization test. *(v3, design, not yet a result)*

## Open questions for the final piece

- Does the held-out robust/leaky split hold at 4B once all evals are in, and does it grow with size (H5)? Early 4B numbers say yes.
- ~~Does thinking-on training (v2) keep exploits low once RL pushes on the leaky grader?~~ No at 2B: v2 leaky ends at exploit 0.70–0.80. Open: does it hold at 4B and 0.8B (v2 runs next)?
- On same-type tasks, how much of the held-out "exploit" rate is hacking vs missing skill?
- *(v3)* With real training signal at 4B, does leaky training hack on near held-out tasks, where the skill is known to be there?
