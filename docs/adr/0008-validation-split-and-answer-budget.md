# 0008. Future versions: a validation split for checkpoint choice, and one answer budget sized for the smallest model

- **Status:** Proposed
- **Date:** 2026-10-14
- **Deciders:** owner (proposed by the agent from the v1–v3 results)

## Terms
| Set | What's in it | Used for, today | Used for, under this ADR |
|---|---|---|---|
| **Train** | Tasks of the trained kinds | Training | Training (unchanged) |
| **OOD test** (out-of-distribution; "held-out" in PREREG, SPEC and the code) | Tasks of kinds never trained on (v3: "near" and "far") | Measured at every eval step **and** reported: the pre-registered test | Measured only at the final and the chosen step, and reported |
| **ID test** (in-distribution; "same-type" in the code) | New tasks of the trained kinds | A second test: did the model learn the trained skill? | Unchanged |
| **Validation** (new) | More tasks of the OOD-test kinds, drawn with a different seed, never reported | Doesn't exist | Measured at every eval step, only to choose a checkpoint |

## Context
Two patterns repeated across all three project versions (`reports/learnings.md` §1, §4, §7):

- **OOD-test skill peaks mid-run, then decays while training reward holds.** 16 of 30 runs on node 1 had an OOD-test peak at least 10 pts above their final step, e.g. v3 4B robust seed 2: 0.93 at step 400 vs 0.80 at step 1,000. The study can only see this through the OOD test itself, so it can't use it: picking a checkpoint by test score overstates it.
- **The answer budget cuts off the smallest model.** In v3, 41% of 2B base answers hit the 1,024-token cap mid-thought (4B: 19%). In v2, a 512 cap made 2B look unable when it was only cut off (ADR 0006).

None of this changes completed versions. v1 verdicts and the v2/v3 reports stand as run.

## Decision
For any future project version:

| Change | Rule |
|---|---|
| **Validation split** | A separate split of tasks of the OOD-test kinds, drawn with its own seed, disjoint from the OOD test. Same size as one test eval (~250 instances × 4 samples). |
| **What it's used for** | Evaluate validation at every eval step. Evaluate the test split only at the final step and at the validation-selected step. |
| **Selection rule (pre-registered)** | The selected checkpoint is the step with the highest validation true success (robust grader), per run. Ties go to the earlier step. |
| **Reporting** | **Primary = final step**, as now: the leaky dynamics after the peak are part of what the study measures. **Secondary = validation-selected step**, labelled as such. Both on the test split. |
| **Answer budget** | One cap for all sizes, keeping the common per-step budget that cross-size comparisons need (PROGRESSION "Common budget across sizes"). Size it from the base diagnostic so that **≤ 10% of the smallest model's answers are cut off**, and confirm memory and step time in the dry run. For the v3 tasks that means about 2,048 tokens. |
| **Checkpoint retention** | Keep the final and the validation-selected eval checkpoints per run. Delete the rest at wrap-up (`scripts/cleanup_node.py`). |

## Alternatives considered
| Option | Why not |
|---|---|
| Early stopping at the validation peak | Hides what leaky training does after the peak, which is the effect under study. Fixed step count stays. |
| Choose checkpoints on the test split | Selecting on the test set overstates the result. That's why v1–v3 peaks are descriptive only. |
| Per-size answer caps | Breaks the common per-step budget that size comparisons rely on. |
| Raise only the 2B cap within v3 | Changes a finished version's settings mid-study. Better as a new version. |

## Consequences
- **Easier:** honest "best checkpoint" numbers. A direct measure of how much OOD-test skill decays after the peak. No more truncated small-model answers counted as failures.
- **Harder:** more evals per run, although validation at every step plus test at 2 steps costs about the same as test at every step today. A longer cap makes steps slower and uses more memory, especially at 4B, so the dry run must confirm it fits. Selection rules go into PREREG before training.
- **Watch:** whether validation and test peaks line up (if not, the split is too small or too different), and whether the larger cap shifts base exploit rates, which would move the "untrained + 10 pts" bar.
