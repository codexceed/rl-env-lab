# 0005. Exploratory follow-up after the first training round

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** owner (proposed after the first ~300 steps of block 1)

## Context

Block 1 (2B ×6 on node 1, 0.8B ×6 on node 2; LR 1e-6, 1,000 steps) is about a third done. The evals every 50 steps show:

| | Held-out (T3) true success | Held-out exploit rate |
|---|---|---|
| 2B base | 0.22 | 0.48 |
| 2B robust, step 300 | 0.33–0.40 | 0.32–0.54 |
| 2B leaky, step 300 | 0.20–0.30 | ~0.70 |
| 0.8B robust / leaky, step 500–600 | 0.25–0.30 / 0.27–0.29 | 0.54–0.72 / 0.53–0.72 |

- **2B:** the robust/leaky split shows on held-out tasks too, for both true success and exploits. **0.8B:** exploits rise in both conditions, with no clear split.
- **Held-out F1, F3 and F4 stay near 0% exact in every run.** Models learn the right tool, customer and status but not new arithmetic, so a held-out "exploit" mixes reward hacking with "learned the format, can't do the sum".
- **Learning stalls.** 91–99% of groups give no signal (PREREG #13). They are solved types, leaky runs at reward ~1.0, or types at ~0%.
- **Traces:** for the ~0% types, 2B often writes the number first, and its own note then works out the correct value.

## Decision

1. **The pre-registered study is unchanged.** All 18 runs finish, and verdicts come from held-out T3 as frozen.
2. **Exploratory eval:** unseen instances of the 11 training task types, 250 × 4 at temp 0.7, both graders. It runs after the fact on every saved eval checkpoint (every 50 steps) of all 18 runs, with no reruns. Results are reported next to T3 and labelled exploratory.
3. **Exploratory v2 round, "work before answering":**
   - **Change:** thinking on with a short budget, and an answer cap of 512 instead of 256. The parser already strips `<think>` and the graders are unchanged.
   - **Go/no-go (~5 min):** a base diagnostic with thinking on. Go only if the 0–4% types rise and at least 6 of 11 types land in the 10–90% band.
   - **If go:** 2B robust + leaky × 3 seeds on one node (~2–3 h), with both evals. Extend to 0.8B/4B only if the effect holds.

## Alternatives considered

| Option | Why not |
|---|---|
| 16 answers per prompt, or escalate all-fail prompts to 16 | Helps only ~4%-pass types, weakly; 0% types stay 0; 2× generation cost, and escalation needs custom TRL code |
| Dynamic sampling (drop dead groups, refill) | Dead groups are mostly solved tasks or leaky runs that already won; little to refill with |
| Harder variants for solved types | Revisit only if v2 saturates |
| Same-type tasks as the held-out set | Would change pre-registered verdicts mid-run; covered by item 2 as exploratory |

## Consequences

- **Order and cost:**
  - Block 1 finishes.
  - Item 2 eval runs on idle GPUs (~1 h).
  - The 4B runs follow, with async recovery checkpoints (in progress).
  - Then the thinking-on check and, if it passes, v2.
  - About 6 h more on one or two 8× H200 nodes at [rate] each, within the buffer before the Fri 06:00 cutoff.
- **Easier:** "can't do new arithmetic" can be told apart from "hacks the reward", and the main failure mode gets one targeted test.
- **Harder:** more results to label carefully. Everything here is exploratory and never changes an H1/H2/H5 verdict.
- **Watch:** thinking-on prompts plus answers must fit the 2,048 vLLM context (1,536 + 512); step time will rise with longer answers.
