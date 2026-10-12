# 0007. Project version v3: a harder, larger task set

- **Status:** Proposed
- **Date:** 2026-10-12
- **Deciders:** owner (proposed by the integrating agent at the owner's request)

## Context

- The thinking-on base check said **no-go at 4B**: base 4B already solves 0.87 of training tasks, only 4 of 11 types sit in the 10–90% band, and the leaky grader passes 98%. A v2 4B run on these tasks would get almost no training signal, least of all the leaky runs (ADR 0005 gate; Change log 2026-10-12).
- The v1 held-out set has two confounds:
  - each held-out template needs a formula never trained (per-seat, unused units), so a held-out "exploit" mixes reward hacking with missing skill (`reports/learnings.md` §1);
  - 50 tasks per template give noisy per-family numbers.
- 2,000 training tasks over 1,000 steps × 8 prompts means each task is seen about 4 times.
- The owner asked for harder tasks in larger numbers for both training and held-out, with freedom over template types, and asked that this be a **new project version** with its own reporting and learnings.

## Decision

**1. v3 is a new project version, kept apart from v1/v2.**

| | v1 (pre-registered) | v2 (ADR 0005/0006) | **v3 (this ADR)** |
|---|---|---|---|
| Tasks | SPEC.md §4 (11 train, 5 held-out) | same as v1 | **SPEC_v3.md** (15 train, 5 near + 5 far held-out) |
| Thinking, answer cap | off, 256 | on, 1,024 | on, 1,024 (v2 settings) |
| Data | `data/` + `data/manifest.json` (frozen) | same | **`data/v3/`** + its own manifest |
| Sizes × rewards × seeds | 0.8B/2B/4B × 2 × 3 | 2B, 0.8B | **4B and 2B × 2 × 3 = 12 runs** |
| Report | `reports/results.md` (verdicts) | `reports/v2.md` | **`reports/v3.md`** |
| Status of numbers | verdicts | exploratory | **exploratory** (labelled v3 everywhere) |

- v1 files are untouched: SPEC.md, PREREG.md, `data/manifest.json` (its hash is part of every run's identity). v1 verdicts stay valid.
- `reports/learnings.md` gets a version tag on every lesson (v1, v2, v3) and its own v3 section.

**2. Same simulator, tools, graders and prompt format.**
- No change to `envlab.sim`, the six tools, robust/leaky grading, G1–G3, or the SPEC §10 prompt. Only the task generators, solvers and wrong-value calls are new.
- Each task is still one tool call. Difficulty comes from what the call must contain (the amount, record or value) and from finding the right customer.

**3. Templates: 3 trained per family, plus a "near" and a "far" held-out template per family.**

| Family → tool | Trained (A, B, C) | Near held-out N (trained steps, recombined) | Far held-out X (a rule never trained) |
|---|---|---|---|
| F1 overcharge → `apply_credit` | A per-seat: qty × (unit − rate) · B promised p% off list, stated rounding · C mid-month downgrade, prorated by days | per-seat **and** p% off | tiered seats: first n at one price, the rest at another |
| F2 duplicate → `refund_payment` | A 3–5 payments incl. failed and already-refunded ones · B installment charged twice · C customer found by email or phone only | installment duplicate, customer found by email only | overpayment: refund the payment that pushed the total above the bill (no equal-amount twin) |
| F3 dispute → `resolve_case` | A p% of disputed amount with a cap · B unused units × unit price · C outage credit prorated by hours | unused units, **then** p% policy | usage with free allowance: credit only billed overage units |
| F4 billing error → `adjust_invoice` | A recompute with discount lines · B add tax at a stated rate, rounded down · C 4–6 lines, large quantities | discount **and** tax | minimum commitment: total = max(sum of lines, committed minimum) |
| F5 account change → `update_customer` | A find customer by phone among same-name customers · B message corrects itself ("actually, use …") · C customers share name and email local part | correction **and** phone lookup | phone given in another format, stored in the account's canonical format |

- v1's held-out formulas (per-seat, unused units, discount lines) become **trained** in v3. "Near" tests transfer: every step it needs was trained, so a near exploit is hacking or carelessness, not missing skill. "Far" keeps a generalization test. They are reported separately.
- Leaky has to find the right customer, so customer-finding gets harder too (indirect lookups, near-duplicate customers). This keeps leaky below saturation.

**4. Larger sets.**

| Split | v1 | v3 |
|---|---|---|
| train | 2,000 (11 templates) | **6,000** (15 × 400), about 1.3 passes in 1,000 steps |
| held-out | 250 (5 × 50) | **1,000**: near 500 (5 × 100) + far 500 (5 × 100) |
| same-type | 250 | 500 (unseen train-template tasks) |
| broken (gate check) | 100 | 100 |

**5. Calibration gate before any v3 training (same rule shape as ADR 0005).**
- Thinking-on base diagnostic on the **train split only**, at 4B and 2B. Go if:
  - at 4B, at least 9 of 15 templates are in the 10–90% band and leaky pass ≤ 90%;
  - at 2B, at least 6 of 15 templates are in band.
- One tuning pass is allowed: number ranges, line counts and distractors only. Held-out data is never scored before the gate, and no held-out numbers are used for tuning.
- Acceptance as in v1: ≥ 99% of valid instances pass G1–G3, and every broken instance fails a gate.

**6. Training and eval.**
- `--variant v3`: v2-think settings plus `data/v3`. Prompt budget rises to 2,048 tokens and context to 3,072, since more records make prompts longer.
- Run ids `v3-<size>-<rob|leak>-s<seed>`, which keeps them out of the v1 verdict code. Held-out eval every 100 steps (v1: every 50) to keep eval GPUs in step with training.
- Report: true success and exploit rate, near vs far, robust vs leaky, with the same paired bootstrap as PREREG. Labelled **v3, exploratory**.

## Alternatives considered

| Option | Why not |
|---|---|
| Run v2 4B on the v1 tasks anyway | Gate no-go; leaky runs would barely train (98% leaky pass) |
| Use v1's `train_hard` variant (more distractors) | Harder record-finding only; the arithmetic stays saturated at 4B with thinking on |
| Multi-call tasks (several tools per task) | Changes the grader and rollout contract; too much risk before the Friday deadline |
| Replace v1 tasks in place | Breaks run identity and the pre-registered verdicts |
| Far-only held-out, like v1 | Keeps the hacking vs missing-skill confound |

## Consequences

- Easier: a 4B experiment with real training signal; held-out results that separate hacking (near) from generalization (far); per-family numbers with 100 tasks each.
- Harder: about a day of building and checking new generators before training. 12 more runs at roughly 5.5 h per 4B run (dry run: ~20 s/step); 4B pairs need both nodes overnight.
- Watch:
  - the calibration gate, since a too-hard set starves 2B the way a too-easy one starves 4B;
  - the prompt budget;
  - the training cutoff of Fri 16 Oct 06:00. v3 training must start by Wed 14 Oct evening to finish with evals.
