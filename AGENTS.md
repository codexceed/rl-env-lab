# AGENTS.md

Guide for any agent (or human) working in this repo. Read this first, then `PROGRESSION.md`.

## What this is

A billing simulator used as an RL environment. Qwen3.5 models are trained with GRPO under two rewards:

- **robust:** checks the full simulator state against the oracle
- **leaky:** checks only surface fields

We measure true success against exploits (samples that pass leaky but fail robust). Deadline: **Fri 16 Oct 2026, 12:00**.

## Sources of truth

| File | Role | Who edits |
|---|---|---|
| `PROGRESSION.md` | **Authoritative execution plan and progress log** | Everyone, per its update rules |
| `DECISIONS.md` | Locked decisions and why | Owner signs off on changes |
| `INSTRUCTIONS.md` | Original plan. **Never edit.** Inherited unless overridden in PROGRESSION | Nobody |
| `SPEC.md` | Environment contract: entities, tools, templates, graders, gates | Agent drafts, owner signs off |
| `PREREG.md` | Hypotheses, thresholds, eval protocol. Frozen before task training | Owner |

If code and SPEC disagree, SPEC wins. Fix the code, or propose a SPEC change in your PR.

## Layout

```
src/envlab/
  sim/       entities, tools, ledger, state hash
  tasks/     f1..f5 templates, generate, broken variants
  verify/    robust, leaky, gates
  prompts.py parse.py
train/       smoke_toy.py  grpo.py  sft.py  coordinator.py
eval/        heldout.py  metrics.py  compare.py
configs/     grpo.yaml (+ per-size overrides: GPU layout only)
reports/     acceptance.md  baseline.md  results.md
tests/
data/ runs/  gitignored (except data/manifest.json, data/splits/)
```

## Commands

```bash
uv sync --group dev              # local dev (macOS/Linux), Python 3.14
uv sync --group dev --extra gpu  # GPU nodes only (torch, transformers, trl, vllm, deepspeed)
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

CI runs ruff and pytest on every PR. **The CI environment has no GPU and no gpu extra installed.** Code imported by tests must not import torch, transformers or vllm at module top level. Import them lazily inside functions, or put the import in `train/` / `eval/` entry points that tests do not import.

## Working rules

**Parallel work**
- Use one git worktree per workstream. Stay inside the files your task owns. If you must touch a shared file (`pyproject.toml`, `SPEC.md`, `PROGRESSION.md`), keep the edit minimal and say so in the PR.
- Don't edit another workstream's module. Code against its interface (from SPEC) and leave a note.

**Git**
- Branch: `<type>/<issue-id>-<description>`, e.g. `feat/3-simulator-core`.
- Commits: Conventional Commits, `<type>(<scope>): <subject>`. Scopes: `sim`, `tasks`, `verify`, `train`, `eval`, `spec`, `prereg`, `plan`, `ci`, `repo`.
- **No AI or assistant attribution anywhere**: no `Co-Authored-By` trailers for agents, no "generated with" footers, in commits, PRs or comments.
- Commit as the repo-local git user. Never override author or committer.

**PRs**
- Sections, in order: **Description**, **Motivation**, **Changes**, **Testing**.
- Use plain language and reference the PROGRESSION section (e.g. "§1.1").
- Merge only when CI is green.

**Updating PROGRESSION.md**
- Tick a box when its *done-when* is met: `[x] … — YYYY-MM-DD HH:MM, note`.
- Never delete a task. Mark it `[~]` cut or `[>]` moved, with a reason.
- Any scope, model, config, schedule or statistics change gets a Change log row. If it touches a locked decision, it also goes in `DECISIONS.md`.

## Code conventions

- **Determinism:** every generator and sampler takes an explicit `seed`. Never use global RNG state. The same seed must give byte-identical output.
- **The simulator is pure:** no I/O and no wall-clock. State is plain dataclasses with a stable hash.
- **Graders:** grader functions take simulator state and return a result. They never mutate state. Both rewards share one interface.
- **Money:** integer cents. No floats.
- **Tests:** every module gets unit tests with known answers. Grader tests cover four cases per family: correct, wrong amount, wrong record, and side effect.
- **Style:** small modules and type hints. Ruff config is in `pyproject.toml`.

## Guardrails

- Secrets live in `.env` and are never committed. Never print tokens.
- Synthetic data only. No real customer data.
- Nothing confidential from employers or vendors goes in this repo. It goes public Friday.
- Every unattended GPU job needs a timeout. **Only the coordinator shuts nodes down.** An individual run never does.
- Never delete final eval weights or eval outputs. Prune a recovery checkpoint only after the run's final artifacts pass their integrity check.
