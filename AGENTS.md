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
| `ARCHITECTURE.md` | How the stack and its flows fit together (diagrams) | Update when a flow or contract changes |
| `PROGRESSION.md` | **Authoritative execution plan and progress log** | Everyone, per its update rules |
| `docs/adr/` | **Architecture Decision Records**: the reasoning behind major decisions | Agents draft, owner accepts |
| `DECISIONS.md` | One-line ledger of decisions, linking to ADRs | Owner signs off on changes |
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
configs/     grpo.yaml (+ per-size overrides: GPU layout, recovery interval)
reports/     acceptance.md  baseline.md  results.md
tests/
data/ runs/  gitignored (except data/manifest.json, data/splits/)
```

## Commands

`make` lists every target. The common ones:

```bash
make setup        # local dev env (Python 3.14) + pre-commit hooks
make setup-gpu    # GPU nodes only: + torch, transformers, trl, vllm, deepspeed, causal-conv1d
make check        # ruff check, ruff format, pyright, pydoclint, import-linter (exactly what CI runs)
make fmt          # auto-fix lint and format
make test         # unit tests
make train RUN=2b-leak-s2 · make eval RUN=2b-leak-s2 STEP=final · make coordinator NODE=node1
make studio-status · make studio-run CMD="make test"   # drive the Lightning studio from a laptop
```

CI runs one required check per lint tool plus the tests. **The CI environment has no GPU and no gpu extra installed.** Code imported by tests must not import torch, transformers or vllm at module top level. Import them lazily inside functions, or put the import in `train/` / `eval/` entry points that tests do not import. Import-linter enforces this for `envlab`.

## Lint standards (shared with the evals repo)

- **Ruff:** line length 120. Rules E, W, F, I, B, UP, SIM, C4, RUF. Use ASCII `x` and `-` in code strings and docstrings, not `×` or `−`; markdown docs may use either.
- **Pyright:** standard mode. Narrow `Optional`s before use.
- **Pydoclint:** Google style. Types live in signatures, not in docstrings. A one-line docstring is fine for small helpers. Once you write `Args:`, `Returns:` or `Raises:`, they must match the code.
- **Import-linter contracts** in `pyproject.toml`:
  - `envlab.sim` never imports tasks, verify or prompts.
  - `envlab` never imports the GPU stack.
- **Versions:** hook versions are pinned in `.pre-commit-config.yaml`. The dev `ruff` pin matches the hook.

## Working rules

**Decisions → ADRs**
- When a request involves a **major architectural, planning or experiment-design change**, **ask the owner whether to record an ADR** before making the change. Examples: model or model family, hypotheses, thresholds or statistics, run matrix, compute or deadline, environment, grader or artifact contract, working process.
- If yes, draft `docs/adr/NNNN-short-title.md` from the template in `docs/adr/README.md`, with status `Proposed`. Add a `DECISIONS.md` row and a PROGRESSION Change log row. Mark it `Accepted` once the owner agrees.
- Never rewrite an accepted ADR. Supersede it with a new one.

**Parallel work**
- Use one git worktree per workstream. Stay inside the files your task owns. If you must touch a shared file (`pyproject.toml`, `SPEC.md`, `PROGRESSION.md`), keep the edit minimal and say so in the PR.
- Don't edit another workstream's module. Code against its interface (from SPEC) and leave a note.

**Git**
- **Commit directly to `main`** by default. Parallel workers use their own worktree branch locally; the integrator merges it into `main` and pushes. No PR.
- **PR only for fundamental changes:** serious architecture or planning changes (e.g. model family, hypotheses or stats rules, run matrix, simulator/grader contract). **Ask the owner before opening one.**
- Branch names, when used: `<type>/<issue-id>-<description>`, e.g. `feat/5-simulator-core`. Reference the issue in commits (`(#5)`) or close it with `Closes #5` in the merge commit.
- Commits: Conventional Commits, `<type>(<scope>): <subject>`. Scopes: `sim`, `tasks`, `verify`, `train`, `eval`, `spec`, `prereg`, `plan`, `ci`, `repo`.
- **No AI or assistant attribution anywhere**: no `Co-Authored-By` trailers for agents, no "generated with" footers, in commits, PRs or comments.
- Commit as the repo-local git user. Never override author or committer.
- Keep `main` green: run `make check && make test` before every push.

**PRs (when the owner approves one)**
- Sections, in order: **Description**, **Motivation**, **Changes**, **Testing**.
- Use plain language and reference the PROGRESSION section (e.g. "§1.1").
- Merge only when CI is green.

**Updating PROGRESSION.md**
- In a parallel workstream, **don't edit `PROGRESSION.md` yourself**. Report which boxes are met in your final summary or PR, and the integrator ticks them on merge. This avoids merge conflicts.
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
