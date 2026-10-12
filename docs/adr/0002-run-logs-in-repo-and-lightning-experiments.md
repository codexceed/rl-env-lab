# 0002. Run logs in the repo, mirrored to Lightning Experiments

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** owner (proposed in the logging review)

## Context

Runs write their logs and eval results to each studio's own disk (`runs/`, `evals/`, gitignored). This has three problems:

- There is no live view across the 18 runs.
- Nothing moves n2's results to the node that builds the report.
- The public repo would hold verdicts but not the evidence behind them.

A review also asked for per-sample traces and protocol checks, so the evidence has to outlive the studios.

## Decision

- **Source of truth stays the local JSONL files** (`train_metrics.jsonl`, `evals/<run>/step-N.jsonl`, traces). Verdicts are computed only from them.
- **A separate sync process** (`train/logsync.py`, `make logsync NODE=…`) runs on each node beside the coordinator. Training and eval code never call it, so a sync failure can't affect a run.
- **Sink A: repo.** The sync copies the small files (metrics, status, run manifests, eval records and traces, check summaries) into a tracked `logs/` folder. It commits only `logs/` and pushes to `main` with a rebase-and-retry, because both nodes push. It uses an allowlist plus a size cap and never copies weights or checkpoints.
- **Sink B: Lightning Experiments (LitLogger).** One experiment per run, with per-step training metrics, per-checkpoint eval summaries (true success, leaky pass, exploit rate) and eval files as artifacts. It keeps a cursor so restarts never double-log. Failures are retried on the next cycle.
- This also covers moving n2's results: `report` reads n2's runs from `logs/` after a pull.

## Alternatives considered

| Option | Why not |
|---|---|
| Weights & Biases | Another account and key. Lightning Experiments sits next to the studios already. |
| Logging from inside the trainer | A tracker outage or slow upload could stall or crash a run. |
| Commit `runs/` and `evals/` directly | They hold weights (8–47 GB per checkpoint). |
| Files only, no repo copy | Evidence would vanish with the studios, and the public repo couldn't be re-checked. |

## Consequences

- **Easier:**
  - Live curves for every run.
  - Anyone can re-run `compare.py` from the public repo.
  - The exploit gallery is built from committed traces.
- **Harder:**
  - The nodes push to `main` alongside human commits, but the sync touches `logs/` only.
  - The repo grows by tens of MB.
- **Watch:**
  - Repo size if traces grow (gzip them if needed).
  - Push races between nodes, handled by the rebase-and-retry.
  - The two sinks drifting apart. On any disagreement the JSONL files win.
