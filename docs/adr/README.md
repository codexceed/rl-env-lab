# Architecture Decision Records

Each ADR records **why** a major decision was made, so nobody has to dig through chat history to find out. `DECISIONS.md` is the one-line ledger; ADRs hold the reasoning.

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-foundational-decisions.md) | Foundational decisions for the 4-day study | Accepted | 2026-10-12 |
| [0002](0002-run-logs-in-repo-and-lightning-experiments.md) | Run logs in the repo, mirrored to Lightning Experiments | Accepted | 2026-10-12 |
| [0003](0003-step-count-backend-and-4b-layout.md) | Step count, generation backend and 4B layout from the benchmark | Accepted | 2026-10-12 |
| [0004](0004-post-baseline-amendments.md) | Amendments after the base-model evals | Accepted | 2026-10-12 |
| [0005](0005-exploratory-followup.md) | Exploratory follow-up after the first training round | Accepted | 2026-10-12 |
| [0006](0006-v2-answer-cap-1024.md) | v2 round: answer cap 1,024 instead of 512 | Accepted | 2026-10-12 |
| [0007](0007-v3-task-set.md) | Project version v3: harder, larger task set (near + far held-out) | Proposed | 2026-10-12 |

## When to write one

Write an ADR for a **major architectural, planning or experiment-design change**, for example:
- a new model or model family
- a change to hypotheses, thresholds or statistics
- a change to the run matrix, compute or deadline
- a change to the environment, grader or artifact contract
- a change to how the work is run

Small fixes and clarifications don't need one. Agents propose an ADR and the owner approves it (see `AGENTS.md`).

## Rules

- **File:** `NNNN-short-title.md`. Numbers are never reused.
- **Status:** `Proposed` → `Accepted`, or later `Superseded by NNNN`. Never delete or rewrite an accepted ADR. Supersede it with a new one.
- **Keep it short:** about one page. Link to `SPEC.md`, `PREREG.md` and `PROGRESSION.md` rather than repeating them.
- **Cross-reference:** every ADR also gets a row in `DECISIONS.md` and a Change log row in `PROGRESSION.md`.

## Template

```markdown
# NNNN. Title

- **Status:** Proposed | Accepted | Superseded by NNNN
- **Date:** YYYY-MM-DD
- **Deciders:** owner (+ who proposed)

## Context
What forces the decision. 2–5 lines.

## Decision
What we do. Bullets or a table.

## Alternatives considered
| Option | Why not |
|---|---|

## Consequences
What gets easier, what gets harder, and what we must now watch.
```
