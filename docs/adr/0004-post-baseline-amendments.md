# 0004. Amendments after the base-model evals

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** owner (proposed after the §2.2/§2.3 base evals)

## Context

The base-model evals (§2.2 diagnostic, §2.3 formal) ran before any task training. They showed three problems with the frozen plan:

| Finding | Number |
|---|---|
| 2B train templates in the 10–90% band | 5 of 10 (3 below, 2 above); PREREG §7 wants ≥ 6 and has no branch for this case |
| Formal base exploit rate (leaky pass, robust fail) | 0.8B 41%, 2B 48%, 4B 57%; H2's "≥ 2× base" would need ≥ 97% for 2B |
| 4B held-out answers cut off by the 128-token cap | 26% (4B writes its working into the note) |

Most base "exploits" are wrong-amount arithmetic on the right record: the model writes the number first and then works it out in the note.

## Decision

All of the following were made before any task-training run and are recorded as dated PREREG §10 amendments.

- **#11 Answer budget 128 → 256 tokens** for training and both evals, all sizes. Truncation of 4B answers falls to 5%. The base evals were re-run at 256.
- **#10 H2 part 3:** leaky exploit rate ≥ base + 10 points (was ≥ 2 × base). The results report flags it as changed after seeing base rates.
- **#12 Train template F1T5 added** (overcharge with both prices in the request, whole-dollar difference): 24% for 2B, so 6 of 11 templates are in band. Three other candidates, F1T4, F4T4 and F4T5, were measured at 6–9% for 2B and dropped. Held-out data is byte-identical (sha256 `097ea4e4…`). Training data stays at 2,000 split evenly, so each template's share falls from 200 to about 182.

## Alternatives considered

| Option | Why not |
|---|---|
| Proceed with 5 of 10 and amend the rule instead | Owner preferred adding learnable tasks |
| SFT warm-up (too-hard branch) | Not a majority-below case; it would change every start checkpoint |
| Keep all 4 candidates | 2B would have 6 of 14 in band, below half; 3 of them give 2B no signal |
| Keep "≥ 2× base" | Unreachable for 2B (≥ 97%), so H2 would fail by construction |
| Keep 128 tokens | Would understate the 4B baseline through truncation, not ability |

## Consequences

- **Easier:** H2 is testable, 4B is measured fairly, and 2B has one more template to learn from.
- **Harder:**
  - The step time rises somewhat with longer answers.
  - H2's bar was set after seeing base rates. It is reported as an amendment, not as pre-registered.
- **Watch:** whether leaky training raises exploits above an already-high base. The robust-vs-leaky gap (H2 part 2) is the cleaner signal.
