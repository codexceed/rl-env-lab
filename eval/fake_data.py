"""Fake ``evals/`` and ``runs/`` trees with known effects, for tests and demos.

Usage::

    python eval/fake_data.py --out /tmp/fake            # all 18 runs + base evals
    python eval/fake_data.py --out /tmp/fake --only 2b  # only the 2B runs

Writes the layout of `train/LAYOUT.md` (no weights, only markers)::

    <out>/evals/<run_id>/step-NNNNN.jsonl   per-sample records (schema: eval/metrics.py)
    <out>/evals/<run_id>/step-NNNNN.done    watcher marker
    <out>/evals/base-<size>/step-00000.jsonl  base model, reward "base", seed null
    <out>/runs/<run_id>/eval/step-NNNNN/COMPLETE   {"step", "final"}
    <out>/runs/<run_id>/status.json         {"status", "size", "reward", "seed"}
    <out>/runs/<run_id>/train_metrics.jsonl  {"step", "reward", ...} per training step
    <out>/data/splits/heldout.txt           the fake frozen held-out ids (eval protocol check)

Known effects baked in (all fake):

* **robust improves:** robust runs gain true success, mostly on families whose base
  pass rate is in the 10-90% band (F1-F3); F4 (too hard) and F5 (too easy) gain little.
* **leaky exploits:** leaky runs gain almost no true success; a growing share of
  their failures become exploits (pass leaky, fail robust). Their training reward
  (leaky pass rate) ends above the robust runs' training reward.
* **exploit grows with size:** final leaky exploit share of failures is
  0.20 / 0.45 / 0.85 for 0.8B / 2B / 4B.

So on the full tree: H1 pass, H2 pass, H5 pass (with 3 seeds per size).
Same arguments → byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # run as a script: make `eval` and `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import H5_SIZES
from train.checkpoints import COMPLETE, step_name

REWARDS = ("robust", "leaky")
SEEDS = (1, 2, 3)
HELDOUT_IDS = Path("data") / "splits" / "heldout.txt"  # same relative path as eval/compare.py reads
FAMILIES = ("F1", "F2", "F3", "F4", "F5")

# Base true-success probability per family (held-out T3) for the 2B model.
FAMILY_BASE = {"F1": 0.85, "F2": 0.50, "F3": 0.30, "F4": 0.03, "F5": 0.995}
SIZE_SHIFT = {"0.8b": -0.4, "2b": 0.0, "4b": 0.4}  # logit shift by size
ROBUST_GAIN_IN_BAND = 1.5  # logit gain at the end of training, families in 10-90% band
ROBUST_GAIN_OUT_BAND = 0.3
LEAKY_GAIN = 0.0
BASE_EXPLOIT_SHARE = 0.03  # share of non-successes that are exploits before training
LEAKY_EXPLOIT_SHARE = {"0.8b": 0.20, "2b": 0.45, "4b": 0.85}  # at the end of training
TRAIN_EXPLOIT_BOOST = 1.3  # leak is easier to hit on train templates
FORMAT_FAIL_BASE = 0.03
FORMAT_FAIL_TRAINED = 0.01
LEARN_TAU = 100.0  # steps; progress f(t) = 1 - exp(-t / tau)

_SIZE_IDX = {s: i for i, s in enumerate(H5_SIZES)}
_REWARD_IDX = {"base": 0, "robust": 1, "leaky": 2}


def run_id_for(size: str, reward: str, seed: int | None) -> str:
    if reward == "base":
        return f"base-{size}"
    return f"{size}-{'rob' if reward == 'robust' else 'leak'}-s{seed}"


def all_run_ids(sizes: Iterable[str] = H5_SIZES) -> list[str]:
    return [run_id_for(z, r, s) for z in sizes for r in REWARDS for s in SEEDS]


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    size, short, seed = run_id.split("-")[:3]
    return size, {"rob": "robust", "leak": "leaky"}[short], int(seed[1:])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _progress(step: int) -> float:
    return 1.0 - math.exp(-step / LEARN_TAU)


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, *keys])


def instances(n_instances: int) -> list[tuple[str, str]]:
    """(instance_id, family), split evenly across the five families."""
    per = n_instances // len(FAMILIES)
    if per * len(FAMILIES) != n_instances:
        raise ValueError("n_instances must be a multiple of 5")
    return [(f"{f}T3-{i:03d}", f) for f in FAMILIES for i in range(per)]


class _Model:
    """Per-sample outcome probabilities for one run at one step."""

    def __init__(self, data_seed: int, size: str, reward: str, seed: int | None, n_inst: int):
        self.size, self.reward = size, reward
        self.insts = instances(n_inst)
        # Instance difficulty is shared by every run (same held-out set).
        inst_off = _rng(data_seed, 99).normal(0.0, 0.5, size=n_inst)
        fam = np.array([_logit(FAMILY_BASE[f]) for _, f in self.insts])
        self.base_logit = fam + inst_off + SIZE_SHIFT[size]
        self.in_band = np.array([0.10 <= FAMILY_BASE[f] <= 0.90 for _, f in self.insts])
        # Seed-level random effects.
        r = _rng(data_seed, 7, _SIZE_IDX[size], _REWARD_IDX[reward], seed or 0)
        self.seed_logit = 0.0 if reward == "base" else float(r.normal(0.0, 0.1))
        self.seed_share = 0.0 if reward == "base" else float(r.normal(0.0, 0.03))

    def probs(self, step: int, *, train: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        """(p_true, p_exploit) per instance, given the sample parsed; and p_format_fail."""
        f = 0.0 if self.reward == "base" else _progress(step)
        if self.reward == "robust":
            gain = np.where(self.in_band, ROBUST_GAIN_IN_BAND, ROBUST_GAIN_OUT_BAND) * f
        elif self.reward == "leaky":
            gain = np.full(len(self.insts), LEAKY_GAIN * f)
        else:
            gain = np.zeros(len(self.insts))
        p_true = _sigmoid(self.base_logit + gain + self.seed_logit)
        share = BASE_EXPLOIT_SHARE
        if self.reward == "leaky":
            share += (LEAKY_EXPLOIT_SHARE[self.size] - BASE_EXPLOIT_SHARE) * f
            share += self.seed_share
            if train:
                share *= TRAIN_EXPLOIT_BOOST
        share = min(max(share, 0.0), 1.0)
        p_ff = FORMAT_FAIL_BASE + (FORMAT_FAIL_TRAINED - FORMAT_FAIL_BASE) * f
        return p_true, share * (1.0 - p_true), p_ff


def eval_records(
    size: str,
    reward: str,
    seed: int | None,
    step: int,
    *,
    data_seed: int = 0,
    n_instances: int = 250,
    n_samples: int = 4,
) -> list[dict]:
    """Per-sample eval records (JSON-ready dicts) for one checkpoint."""
    m = _Model(data_seed, size, reward, seed, n_instances)
    p_true, p_ex, p_ff = m.probs(step)
    r = _rng(data_seed, 11, _SIZE_IDX[size], _REWARD_IDX[reward], seed or 0, step)
    u_ff = r.random((n_instances, n_samples))
    u = r.random((n_instances, n_samples))
    rid = run_id_for(size, reward, seed)
    out = []
    for i, (iid, fam) in enumerate(m.insts):
        for k in range(n_samples):
            parse_ok = bool(u_ff[i, k] >= p_ff)
            success = parse_ok and bool(u[i, k] < p_true[i])
            exploit = parse_ok and not success and bool(u[i, k] < p_true[i] + p_ex[i])
            out.append(
                {
                    "run_id": rid,
                    "size": size,
                    "reward": reward,
                    "seed": seed,
                    "step": step,
                    "instance_id": iid,
                    "family": fam,
                    "template": f"{fam}T3",
                    "sample_idx": k,
                    "parse_ok": parse_ok,
                    "robust_pass": success,
                    "leaky_pass": success or exploit,
                }
            )
    return out


def train_log(
    size: str, reward: str, seed: int, last_step: int, *, data_seed: int = 0, n_inst: int = 250
) -> list[dict]:
    """Per-step training log rows: the run's own reward (robust or leaky pass rate)."""
    m = _Model(data_seed, size, reward, seed, n_inst)
    r = _rng(data_seed, 13, _SIZE_IDX[size], _REWARD_IDX[reward], seed)
    rows = []
    for step in range(1, last_step + 1):
        p_true, p_ex, p_ff = m.probs(step, train=True)
        rate = (1 - p_ff) * float(np.mean(p_true + (p_ex if reward == "leaky" else 0.0)))
        rows.append(
            {
                "step": step,
                "reward": float(np.clip(rate + r.normal(0.0, 0.03), 0.0, 1.0)),
                "dead_group_share": float(np.clip(0.3 - 0.1 * _progress(step), 0, 1)),
                "completion_len": 90.0,
                "step_time_s": 12.0,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_fake_tree(
    root: str | Path,
    *,
    data_seed: int = 0,
    run_ids: Iterable[str] | None = None,
    stop_at: Mapping[str, int] | None = None,
    base_sizes: Iterable[str] = H5_SIZES,
    final_step: int = 300,
    eval_every: int = 50,
    n_instances: int = 250,
    n_samples: int = 4,
    train_logs: bool = True,
) -> tuple[Path, Path]:
    """Write ``<root>/evals`` and ``<root>/runs``. Returns (evals_root, runs_root).

    ``run_ids`` restricts which of the 18 runs exist (default all). ``stop_at``
    maps a run ID to the last step it reached: it is "running", has eval
    checkpoints up to that step and no final checkpoint.
    """
    root = Path(root)
    evals, runs = root / "evals", root / "runs"
    stop_at = dict(stop_at or {})
    kw = {"data_seed": data_seed, "n_instances": n_instances, "n_samples": n_samples}

    ids_path = root / HELDOUT_IDS
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("".join(f"{iid}\n" for iid, _ in instances(n_instances)))

    for size in base_sizes:
        rid = run_id_for(size, "base", None)
        out = evals / rid / f"{step_name(0)}.jsonl"
        _write_jsonl(out, eval_records(size, "base", None, 0, **kw))
        out.with_suffix(".done").write_text("{}\n")

    for rid in all_run_ids() if run_ids is None else list(run_ids):
        size, reward, seed = parse_run_id(rid)
        last = stop_at.get(rid, final_step)
        finished = rid not in stop_at
        steps = list(range(eval_every, last + 1, eval_every))
        if finished and (not steps or steps[-1] != last):
            steps.append(last)
        rdir = runs / rid
        for step in steps:
            out = evals / rid / f"{step_name(step)}.jsonl"
            _write_jsonl(out, eval_records(size, reward, seed, step, **kw))
            out.with_suffix(".done").write_text("{}\n")
            marker = rdir / "eval" / step_name(step) / COMPLETE
            marker.parent.mkdir(parents=True, exist_ok=True)
            final = finished and step == last
            marker.write_text(json.dumps({"step": step, "final": final}) + "\n")
        status = {
            "status": "finished" if finished else "running",
            "size": size,
            "reward": reward,
            "seed": seed,
        }
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "status.json").write_text(json.dumps(status, sort_keys=True) + "\n")
        if train_logs:
            log = train_log(size, reward, seed, last, data_seed=data_seed, n_inst=n_instances)
            _write_jsonl(rdir / "train_metrics.jsonl", log)
    return evals, runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write a fake evals/ and runs/ tree with known effects.")
    p.add_argument("--out", required=True, type=Path, help="writes <out>/evals and <out>/runs")
    p.add_argument("--seed", type=int, default=0, help="data seed")
    p.add_argument("--only", nargs="*", help="sizes or run IDs to include (default: all 18)")
    p.add_argument("--final-step", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--n-instances", type=int, default=250)
    p.add_argument("--n-samples", type=int, default=4)
    a = p.parse_args(argv)
    ids = None
    if a.only:
        ids = [r for r in all_run_ids() if r in a.only or r.split("-")[0] in a.only]
    evals, runs = write_fake_tree(
        a.out,
        data_seed=a.seed,
        run_ids=ids,
        final_step=a.final_step,
        eval_every=a.eval_every,
        n_instances=a.n_instances,
        n_samples=a.n_samples,
    )
    print(f"wrote {evals} and {runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
