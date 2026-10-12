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
| `reports/learnings.md` | **Living record of lessons** from results, issues and decisions; seed of the final knowledge piece | Everyone, per "Updating `reports/learnings.md`" |

If code and SPEC disagree, SPEC wins. Fix the code, or propose a SPEC change in your PR.

## Layout

```
src/envlab/
  sim/       entities, tools, ledger, state hash
  tasks/     f1..f5 templates, generate, build, broken variants
  verify/    robust, leaky, gates, rollout, acceptance
  calls.py prompts.py parse.py
train/       grpo.py  rewards.py  coordinator.py  async_recovery.py  checkpoints.py  run_manifest.py
             logsync.py  compat_check.py  smoke_toy.py  bench.py  LAYOUT.md (artifact contract)
eval/        heldout.py  watcher.py  metrics.py  compare.py  gallery.py
             sametype_batch.py  sametype_report.py  v2_report.py   (exploratory, ADR 0005)
configs/     grpo.yaml (+ sizes/: GPU layout, recovery interval; variants/: v2-think)  allocation.yaml
scripts/     setup_gpu.sh  dry_runs.sh  gpu_checks.sh  base_evals.sh  studio.py
reports/     acceptance.md  results.md  learnings.md  (+ sametype.md, v2.md when generated)
logs/        run logs synced from the GPU nodes (train/logsync.py)
tests/
data/ runs/  gitignored (except data/manifest*.json, data/splits/)
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
make plan NODE=node1 · make dry-run · make logsync      # print a node's plan · GPU dry runs · log mirroring
make studio-status STUDIO=rl-env-lab-n2 · make studio-run CMD="make test"   # drive a studio from a laptop (default n1)
```

CI runs one required check per lint tool plus the tests. **The CI environment has no GPU and no gpu extra installed.** Code imported by tests must not import torch, transformers or vllm at module top level. Import them lazily inside functions, or put the import in `train/` / `eval/` entry points that tests do not import. Import-linter enforces this for `envlab`.

## Lint standards (shared with the evals repo)

- **Ruff:** line length 120. Rules E, W, F, I, B, UP, SIM, C4, RUF. Use ASCII `x` and `-` in code strings and docstrings, not `×` or `−`; markdown docs may use either.
- **Pyright:** standard mode. Narrow `Optional`s before use.
- **Pydoclint:** Google style. Types live in signatures, not in docstrings. A one-line docstring is fine for small helpers. Once you write `Args:`, `Returns:` or `Raises:`, they must match the code.
- **Import-linter contracts** in `pyproject.toml`:
  - `envlab.sim` never imports tasks, verify or prompts.
  - `envlab` never imports the GPU stack.
- **Versions:** hook versions are pinned in `.pre-commit-config.yaml`; pre-commit itself and `ruff` are pinned in the dev group and match it.
- **Hooks are mandatory:** `make setup` / `make setup-gpu` install the git hook (worktrees share it). Never commit with `--no-verify`; fix the lint instead.

## Working rules

**Decisions → ADRs**
- When a request involves a **major architectural, planning or experiment-design change**, **ask the owner whether to record an ADR** before making the change. Examples: model or model family, hypotheses, thresholds or statistics, run matrix, compute or deadline, environment, grader or artifact contract, working process.
- If yes, draft `docs/adr/NNNN-short-title.md` from the template in `docs/adr/README.md`, with status `Proposed`. Add a `DECISIONS.md` row that links the ADR file in its Status column (`[ADR NNNN](docs/adr/NNNN-short-title.md)`) and a PROGRESSION Change log row. Mark it `Accepted` once the owner agrees.
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

**Updating `reports/learnings.md`**
- Add a lesson when a result, incident or decision teaches something non-obvious: one line, the lesson in bold, then the evidence (numbers, run ids, ADR). Put it in the matching section.
- Label its source: *pre-registered*, *training-time* or *exploratory*. Never present an exploratory or training-time number as a verdict.
- When later data changes a lesson, correct it in place and say why. Bump **Last updated**.
- The integrator updates it at least at each gate (block done, verdicts in) and when closing an incident. Parallel workstreams report candidate lessons in their summary instead of editing it.

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
