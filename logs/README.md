# logs/

Small run logs from the GPU nodes, committed so the results can be audited from the repo:

- `runs/<run_id>/`: `status.json`, `train_metrics.jsonl`, `run_manifest.json`, and `eval_steps.json` (eval checkpoints and their eval state)
- `runs/_coordinator/<node>.jsonl`: coordinator event logs
- `evals/<run_id>/`: per-sample eval records `step-NNNNN.jsonl`, plus traces `step-NNNNN.traces.jsonl` when present
- `checks/<node>/`: compat, toy and bench results

These are copies. The source of truth is `runs/` and `evals/` on each node, and verdicts are computed only from those local JSONL files (`eval/compare.py`). The copies are produced by `make logsync NODE=n1` (`train/logsync.py`). It copies an allowlist of `.json`/`.jsonl` files with a size cap (no weights or checkpoints), commits and pushes them, and mirrors the same data to Lightning experiments.
