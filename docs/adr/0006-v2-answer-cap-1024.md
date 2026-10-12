# 0006. v2 round: answer cap 1,024 instead of 512

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** owner (after the thinking-on base check)
- **Amends:** [ADR 0005](0005-exploratory-followup.md) item 3 (answer cap only)

## Context

ADR 0005 gated the v2 round ("thinking on") on a 2B base check: the types at 0–4% must rise, and ≥ 6 of 11 types must land in the 10–90% band. We ran it at the planned 512-token cap, then at 1,024 to see whether the cap was the blocker.

| 2B base, 11 train types × 50 × 8 | True success | Exploit rate | Format failure | Gate |
|---|---|---|---|---|
| Thinking off (v1) | 0.31 | 0.43 | 0.02 | — |
| Thinking on, cap 512 | 0.35 | 0.02 | 0.63 | **no-go**: F4T2 stays 0%, F1T2 4% → 5% |
| Thinking on, cap 1,024 | **0.61** | 0.11 | 0.18 | **go**: low types 4 → 21%, 4 → 72%, 0 → 24%; 7 of 11 in band |

At 512, 54% of answers were cut off mid-thought. When the reasoning finished, most types were 93–100% correct.

## Decision

- **v2 answer cap 1,024** (reasoning included), and a vLLM context of 2,560 (1,536 prompt + 1,024). Both live in `configs/variants/v2-think.yaml`, the only place they change. Thinking-on base evals use the same budget (`eval.think_max_tokens`, `eval.think_max_model_len`).
- **Strict grading kept:** a tool call inside a thought that never closes is a format failure, in training and eval alike.
- **Run:** 2B robust + leaky × 3 seeds on node 1 after its 2B block (`configs/allocation.yaml`, 10 h timeout).
- Everything else in ADR 0005 stands: exploratory only, never in an H1/H2/H5 verdict.

## Alternatives considered

| Option | Why not |
|---|---|
| Go at 512 | Fails the gate as written; most early groups would score 0 from truncation alone |
| No-go | Leaves the clearest signal so far untested: thinking on cuts base exploits from 43% to 11% |
| Cap 2,048 | Not needed: 1,024 already clears the gate; longer steps and more memory |

## Consequences

- **Easier:** v2 tests "work before answering" without truncation being the main effect.
- **Harder:**
  - Answers up to 4× longer than v1, so steps are slower and v2 runs take longer (10 h timeout).
  - The cap was chosen after seeing the base check. The v2 report says so.
- **Watch:** step time and GPU memory on the first v2 steps; the share of answers still cut off at 1,024.
