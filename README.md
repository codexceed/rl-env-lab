# rl-env-lab

A billing simulator used as an RL environment. Qwen3.5 models (0.8B, 2B, 4B) are trained
with GRPO under two rewards: a **robust** one that checks the full simulator state, and a
**leaky** one that checks only surface fields. We measure how often the leaky runs learn to
pass the leaky check without doing the task (exploits).

**Start here:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (how it fits together) → [`SPEC.md`](SPEC.md) (environment contract) · [`PREREG.md`](PREREG.md) (verdict rules) · [`docs/adr/`](docs/adr/) (why) · [`PROGRESSION.md`](PROGRESSION.md) (live plan) · `reports/results.md` (results).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14 (code stays 3.12-compatible). `make` alone lists every target.

```bash
make setup                               # dev env + pre-commit hooks (macOS or Linux)
make setup-gpu                           # Linux nodes (CPU is fine): GPU stack + causal-conv1d build
uv sync --group dev --extra report       # optional: matplotlib, for the report figures
make check && make test                  # lint, types, docstrings, imports; unit tests
```

## Run it

```bash
make data                                # build train / held-out / broken splits
make train RUN=2b-rob-s1                 # one GRPO run -> runs/2b-rob-s1/
make eval RUN=2b-rob-s1 STEP=final       # held-out eval -> evals/2b-rob-s1/
make report                              # tables, verdicts, figures -> reports/results.md
```

The full 18-run matrix is launched by `make coordinator NODE=node1` (see `train/LAYOUT.md`).

## Try the report on fake data

```bash
make report-fake
```

This writes a fake `evals/` and `runs/` tree with all 18 runs and known effects
(`eval/fake_data.py`) to `$TMPDIR/envlab-fake`, then runs `eval/compare.py` on it. The report
lands in `$TMPDIR/envlab-fake/results.md`, with figures next to it. With fake data, H1, H2
and H5 all pass. `reports/results.md` is not touched.

`eval/compare.py` works on partial data: missing or unfinished runs are listed, and every
verdict that depends on them is marked provisional.

## Layout

```
src/envlab/   sim/ (simulator, tools, ledger)  tasks/ (templates, data build)  verify/ (graders, gates)
train/        coordinator.py  checkpoints.py  LAYOUT.md (artifact layout)  grpo.py (coming)
eval/         metrics.py (stats, verdicts)  compare.py (report)  fake_data.py  watcher.py  heldout.py (coming)
configs/      allocation.yaml (GPU allocation)
reports/      results.md  figures/
scripts/      studio.py (Lightning studio helpers)
tests/        unit tests, no GPU needed
data/ runs/ evals/   artifacts (data/ and runs/ are gitignored)
```
