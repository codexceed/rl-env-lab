# Story: outline and findings

A suggested shape for the post, not a draft. Numbers are copied from `results.md` and `learnings.md`;
check them there before publishing. **[pre-reg]** = pre-registered verdict data (v1),
**[expl]** = exploratory, **[train]** = training-time logs.

## One-line pitch

Train the same model twice with GRPO: once with a grader that checks everything, once with one
that checks only surface fields. The second one looks perfect in training and learns almost nothing
real, and the bigger the model, the more it gives up.

## Suggested outline

| # | Section | Point | Figure |
|---|---|---|---|
| 1 | Hook | A reward of 99% that means nothing: the leaky reward is maxed out in 50 steps | `1-training-reward` |
| 2 | Setup | Billing simulator, one tool call per task, two graders, Qwen3.5 at 0.8B / 2B / 4B, 3 seeds, pre-registered rules | Mermaid loop diagram in `results.md` ("How the training works") |
| 3 | How RL learns from a score | GRPO in one table: the advantage, why all-same groups teach nothing, why a 1-cent answer gets the same push as the right one | Worked-examples table in `results.md` |
| 4 | What really happened | Robust training transfers; leaky training stays at the untrained level | `2-true-success` |
| 5 | What the hack looks like | Exploit gallery: right record, wrong amount, most often **1 cent** | `3-exploit` + gallery table |
| 6 | Size | Exploit rate is flat with size, but the skill lost grows with it | `4-final-by-size` |
| 7 | Catching it early | Log a trusted grader next to the one you optimise | `5-early-warning` |
| 8 | Honest results | H2 missed its bar by 0.2 pts; what that does and doesn't mean; 3 seeds | — |
| 9 | Follow-ups | Thinking on (v2), harder tasks (v3): same story, stronger | — |
| 10 | Takeaways | Short list for practitioners (below) | — |

## Findings worth telling

### Pre-registered verdicts (v1)

| Finding | Numbers | Tag |
|---|---|---|
| The environment teaches (H1 **pass**) | 2B robust OOD-test true success 21.8% → 36.6% (+14.8 pts, CI +9.7 to +20.8) | [pre-reg] |
| A leaky reward misleads (H2 **fail by 0.2 pts**) | Training reward 97.8% leaky vs 82.6% robust; true success 26.9% vs 36.6% (gap +9.8, CI +5.4 to +14.0, bar 10). Direction confirmed, size undecided | [pre-reg] |
| Bigger models don't exploit more (H5 **fail**) | Leaky exploit 72.7 / 63.9 / 64.5% (0.8B / 2B / 4B) | [pre-reg] |

### The stronger stories

| Finding | Numbers | Tag |
|---|---|---|
| The skill a leaky reward costs grows with size | Robust − leaky true success: +2 / +10 / +22 pts | [pre-reg] data, descriptive |
| Leaky training flattens the size ladder | Every size ends near 27–28% true success under leaky training | [pre-reg] data, descriptive |
| The model learns the *safest* wrong answer | 82% of 6,032 exploits are "right record, wrong amount"; 0 of 5,399 leaky amounts exact; 1 cent is the most common amount (a credit above the amount due would fail the tool) | [pre-reg] data, descriptive |
| A second grader catches it early | Leaky reward ~99% by step 100 while the logged robust score falls (4B 65% → 48% by step 300 → 32%) | [train] |
| Leaky training is reproducible; robust isn't | Final true success spread across seeds: leaky 2–7 pts, robust up to 21 pts (4B 55.7 / 36.9 / 57.9%) | [pre-reg] data, descriptive |
| The v1 OOD test hid 4B's damage | 4B leaky looks flat on the OOD test (29.7 → 28.4%) but falls 65% → 34% on new tasks of trained kinds: a test the model already fails can't show unlearning | [expl] |
| Leaky training unlearns exactly what isn't checked | v3 4B leaky: amount families 37 → 1% and 43 → 0%, no-arithmetic family stays at 79% | [expl] v3 |
| Thinking roughly doubles robust transfer, but doesn't stop hacking | v2 2B robust OOD-test true success ~73% vs v1 ~37%; v2 leaky still exploits ~74% | [expl] v2 |

### Caveats to state plainly

- One simulator, one model family, one leak we designed ourselves.
- 3 seeds: the confidence intervals likely understate seed-to-seed spread.
- Verdicts use the final step; transfer peaked mid-run (4B robust ~85% at steps 200–450).
- 90–99% of GRPO groups gave no signal (all 8 answers scored the same).
- H2's exploit bar and the answer cap were amended after the base evals, before any training (ADR 0004).

## Takeaways for practitioners (draft)

1. Your training reward measures your grader, not your model. Hold out a stricter check.
2. Log a trusted grader on the training batches next to the one you optimise: cheapest early warning.
3. Test on tasks the model *can* already do, too: unlearning is invisible on tasks it already fails.
4. Expect the hack to be the cheapest answer the grader accepts, and reproducible across seeds.
5. Run enough seeds for the robust condition: that's where the variance lives.
