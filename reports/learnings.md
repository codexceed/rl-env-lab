# Learnings

What this study has taught us so far, from results, issues and decisions. Each line is a lesson plus the evidence behind it. This is the seed of the final knowledge piece.

- **Living document.** Add a lesson when a result, incident or decision teaches something non-obvious; correct one when later data changes it (strike the old claim, say why). Rules: `AGENTS.md` → "Updating `reports/learnings.md`".
- **Labels:** *pre-registered* = from H1/H2/H5 verdicts; *training-time* = from `train_metrics.jsonl` on training tasks; *exploratory* = ADR 0005/0006/0007 work. Never present an exploratory or training-time number as a verdict.
- **Project versions:** every lesson is tagged with the version it comes from. Numbers from different versions use different tasks; don't compare them without saying so.

| Version | Tasks | Settings | Runs | Status |
|---|---|---|---|---|
| **v1** | `SPEC.md` (11 train, 5 held-out templates) | thinking off, cap 256 | 0.8B / 2B / 4B × robust, leaky × 3 seeds | pre-registered (PREREG) |
| **v2** | same as v1 | thinking on, cap 1,024 (ADR 0005, 0006) | 2B, 0.8B × robust, leaky × 3 seeds (4B dropped: tasks too easy) | exploratory, done |
| **v3** | `SPEC_v3.md`: harder, 15 train, near + far held-out (ADR 0007) | thinking on, cap 1,024 | 4B, 2B × robust, leaky × 3 seeds | exploratory, done |
- **Last updated:** 2026-10-14 (v2 0.8B results; compute totals; replication hardening lessons).

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
- **At 0.8B, thinking-on robust training also helps, but one seed in three fails.** v2 0.8B held-out true / exploit at step 1,000: robust 45.9% / 27.5% (seeds 59.4%, 59.6%, 18.8%), leaky 26.6% / 67.6%. The failed seed ends with 20% format failures. v1 0.8B: robust 29.3% vs leaky 27.0%. *(v2, exploratory, `reports/v2.md`)*
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
| **A restarted node needs a reinstall, not just a warm-up** | The venv's files are links into uv's package cache on the studio's system disk, which a stop wipes. After N2's restart, `torch/nn/utils` was simply missing (`ModuleNotFoundError` after a 564 s import), and a plain `uv sync` then removed the GPU extra. `scripts/setup_gpu.sh` fixed it in ~55 min on the slow cold disk. *Corrected 2026-10-13: first read as "just a slow disk"; slow cold reads were real but not the whole cause* (v3) |
| **Running `.venv/bin/python` directly drops the venv's tools from PATH** | vLLM's compiler needs `ninja` from the venv; all 15 v3 evals on N2 failed at generation until the watcher was launched with the venv's `bin/` on PATH (`uv run` does this for you). Same bug as the v2 same-type evals (v3) |
| **The cross-studio mount works one way in practice** | N2 read N1's checkpoints fine (8 parallel 4B loads in 92 s, close to local). N1 never saw N2's new eval files, and downloads of them returned 404 for 10+ min: a busy studio's writes reach shared storage late. Streaming the 58 MB result tar through the command channel worked (v3) |
| **A second node for evals paid off once running** | v3 4B held-out evals took ~6 min each; N2 ran 60 in ~1.5 h on 8 GPUs, vs ~12 h on N1's 2 spare GPUs. Setup problems cost ~2 h of N2 ([cost]) first (v3) |
| **Ad-hoc GPU jobs race the eval watcher** | Twice a one-off job collided with an eval that claimed the GPU seconds earlier. Pause the watcher, or add GPU claims |
| **Colocated vLLM beat a separate server for 4B** | 4 ZeRO-3 ranks, each with a colocated engine: 4.4 s/step vs 8.1 s for 1 server + 2 trainers |
| **Budget the whole study, not just training** | ~41 node-hours of 8x H200 at [rate], [cost] in total: node 1 ~26 h, node 2 ~15 h. ~3 h ([cost]) went to setup failures after restarts. Disk peaked at ~4 TB per node (recovery and intermediate eval checkpoints); cleanup kept final + peak checkpoints (310 GB across both). The dashboard's disk figures lag by hours; trust `du` (all versions) |
| **Check the node before launching, not after the first failure** | Restart breakage, missing GPU packages, missing `ninja`, stale data and missing model exports each surfaced only as failing jobs. `make preflight` now checks all of them and the coordinator refuses to launch on a failure (all versions) |
| **Schedule every baseline the reports need** | The v3 untrained-model held-out eval was never queued and was run by hand at the end. The eval watcher now schedules base-model evals first, per size and version (v3) |
| **The committed logs are enough to rebuild the headline numbers** | `compare.py`, `v2_report.py` and `v3_report.py` reproduce every reported number from `logs/` on a laptop, no GPU (all versions) |
| **Run identity hashes are fragile** | The data manifest's hash is part of every run's identity: editing it would have broken resumes and pairs, so new data went in a separate manifest. I/O-only settings are excluded from the hash |

## 6. Process (agents and humans, all versions)

- **Verify agents' claims; enforce, don't trust.** An agent reported "checks pass" with 4 lint errors left. Fix: a mandatory pre-commit hook. GitHub branch protection isn't available on a private free repo, so the local hook is the gate and CI is the backstop.
- **Pre-register, then date every amendment.** Four changes came after seeing data (H2 bar, answer cap, F1T5, v2 cap). Each is recorded with its reason and flagged in the report, not hidden.
- **Shared nodes have toolchain traps.** A package pulled in a second `uv` binary that child jobs would pick up; the GPU stack is installed outside the lock file, so syncs must be additive; node clocks run on UTC.

## 7. v3: harder task set (ADR 0007)

*All 12 v3 runs trained and evaluated 2026-10-13, held-out and same-type (`reports/v3.md`).*

| Base, thinking on, train tasks | v1 tasks | v3 tasks |
|---|---|---|
| 4B true / leaky / in band | 0.87 / 0.98 / 4 of 11 | **0.58 / 0.75 / 11 of 15** |
| 2B true / leaky / in band | 0.61 / — / 6 of 11 | **0.27 / 0.43 / 9 of 15** |

- **Harder arithmetic plus near-duplicate customers moved 4B back into the learnable band without a tuning pass.** The four types with no arithmetic stayed easy (F5A 93%, F5C 100%), as predicted; F1B and F3A (percent rules) also stayed above 90%. *(v3, exploratory, base model)*
- **At 2B the answer budget is the bottleneck again.** 41% of v3 answers hit the 1,024-token cap mid-thought (4B: 19%). Watch whether training shortens reasoning or keeps hitting the cap. *(v3, exploratory, base model)*

- **On v3, leaky training makes 4B worse than the untrained model.** Held-out true success: base 57% near / 63% far; leaky final 28% / 29% (-28 / -34 pts vs base, CIs exclude 0), with exploits at 68%. In v1, leaky training stayed at the base level (0.30 -> 0.28). The robust - leaky gap is +44 near / +46 far (v1 4B: +22 pts on its own tasks). *(v3, exploratory, `reports/v3.md` paired bootstrap)*
- **Robust training beats the v3 base only modestly by the final step.** +16 pts near (CI includes 0) and +12 far (CI excludes 0). It peaks around step 400 (87% / 89%), then decays to 73% / 75%, the same mid-run peak as v1 4B. *(v3, exploratory)*
- **The hacking is as strong on near as on far tasks.** Near tasks only recombine trained steps, so the leaky models' 68% near exploits are hacking, not missing skill. *(v3, exploratory)*
- **Leaky damage grows with training time.** Leaky true success falls steadily from 55% / 69% at step 100 to 28% / 29% at step 1000 (near / far); exploits rise from 42% / 29% to 68% / 68%. *(v3, exploratory)*

- **At 2B, leaky training adds exploits without adding skill; robust training adds both skill and surface passes.** Near held-out true success: base 19%, leaky 19%, robust 48% (+29 pts vs base, CI excludes 0). Robust 2B also raises "exploits" by 16 / 22 pts. Its format failures fall from ~43% to ~4%, so many more answers parse and pass the surface check while still being wrong. That is weak skill more than hacking. Robust - leaky: +29 near / +20 far. *(v3, exploratory)*
- **Gains are larger on unseen tasks of trained types than on held-out types.** Same-type true success, base -> robust: 4B 60% -> 89%, 2B 29% -> 78%. Held-out near: 4B 57% -> 73%, 2B 19% -> 48%. Leaky models sit at or below base on both (4B 33%, 2B 27% same-type) with 66-72% exploits. *(v3, exploratory, final checkpoints)*
- **Keep the mid-run peaks.** 16 of 30 runs on N1 had a held-out peak at least 10 pts above their final step, for example v3-4b-rob-s2 0.93 at step 400 vs 0.80 final, and v3-4b-leak-s3 0.69 at step 100 vs 0.11 final. These checkpoints were kept when the disks were cleaned. Picking a peak by held-out score overstates it; treat peaks as descriptive. *(all versions, exploratory)*

- **Design that worked:** the v3 "near" held-out set only recombines trained steps, so a near exploit is hacking, not missing skill; "far" held-out keeps a generalization test. The near/far split is what made the leaky result readable (above). *(v3, design)*

## Open questions for the final piece

- ~~Does the held-out robust/leaky split hold at 4B and grow with size?~~ Yes on complete v1 data: +2 / +10 / +22 pts at 0.8B / 2B / 4B (descriptive; the pre-registered size test was about exploit rate, which did not grow).
- ~~Does thinking-on training keep exploits low under a leaky grader?~~ No: v2 leaky ends at 68-80% exploits at 0.8B and 2B; v3 leaky at 68-80% at 4B and 2B.
- ~~With real training signal at 4B, does leaky training hack on near held-out tasks?~~ Yes: 68% near exploits, and true success falls below the untrained model (v3).
- How much of the held-out "exploit" rate on far tasks is hacking vs missing skill? Same-type and near results bound it, but no eval separates the two on far tasks.
- Why does held-out transfer peak mid-run and then decay while training reward holds? A validation split would let a future run select checkpoints, or stop early, without touching the test set.
