"""Fill the results tables, verdicts and plots from held-out eval JSONL and training logs.

Usage::

    python eval/compare.py                                # evals/ + runs/ -> reports/results.md
    python eval/compare.py --evals E --runs R --report out.md --figures out/figures
    python eval/compare.py --stdout --no-plots            # print the block, write nothing

Plots need the ``report`` extra (``uv sync --group dev --extra report``); without
matplotlib they are skipped and the report says so. PROGRESSION.md §1.5 / §3.4.

Inputs
------

**Eval JSONL** (`train/LAYOUT.md`): ``<evals>/<run_id>/step-NNNNN.jsonl``, one
`eval.metrics.EvalRecord` per line. Base models live in ``<evals>/base-<size>/``
with ``reward: "base"``, ``seed: null``, ``step: 0``.

* A step with a ``.failed`` or ``.claim`` marker next to it is skipped (failed, or
  being evaluated right now).
* Unparseable lines (e.g. a truncated last line) are skipped and counted.
* A checkpoint whose instance set differs from the common held-out set (an
  unfinished eval) is dropped and listed in the data notes.

**Training logs**: ``<runs>/<run_id>/train_metrics.jsonl``, one JSON object per
optimizer step::

    {"step": 1, "reward": 0.42, ...}

* ``step`` (int): global optimizer step, 1-based, on the same scale as the eval
  checkpoint steps (``eval/step-00050`` = weights after step 50).
* ``reward`` (float): mean reward over every generation in that step, under the
  run's **own** reward (robust or leaky), before advantage normalisation.
* Other keys (``dead_group_share``, ``entropy``, ``completion_len``,
  ``step_time_s``, ...) are allowed and ignored here.
* Append-only. After a crash and resume, re-logged steps are fine: the **last**
  line for a step wins.

**Run state** comes from ``<runs>/<run_id>/eval/step-NNNNN/COMPLETE``: a run is
*final* when the step marked ``"final": true`` has been evaluated. Without that
(still training, eval behind, or ``runs/`` not synced) its last evaluated step is
used and every verdict that depends on it is marked **provisional**.

Run IDs: the matrix is ``{0.8b,2b,4b}-{rob,leak}-s{1,2,3}``. A rerun ``<id>-r<k>``
replaces the original (highest ``k`` wins). Other run IDs (ablations) are ignored
here and listed in the data notes.

Output
------

A markdown block written between ``<!-- BEGIN GENERATED -->`` and
``<!-- END GENERATED -->`` in the report (text outside the markers is kept), and
PNG figures in ``--figures``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # run as a script: make `eval` and `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import (  # noqa: E402
    DEFAULT_N_BOOT,
    DEFAULT_REWARD_WINDOW,
    H5_SIZES,
    METRICS,
    PREREG_SEEDS,
    STATUS_INCOMPLETE,
    ConditionMatrix,
    EvalRecord,
    RunEval,
    Verdict,
    condition_matrix,
    hierarchical_bootstrap,
    index_runs,
    percentile_ci,
    verdict_h1,
    verdict_h2,
    verdict_h5,
)
from train.checkpoints import list_eval_checkpoints, read_status  # noqa: E402

BEGIN_MARKER = "<!-- BEGIN GENERATED -->"
END_MARKER = "<!-- END GENERATED -->"

REWARDS = ("robust", "leaky")
SEEDS = tuple(range(1, PREREG_SEEDS + 1))
SHORT = {"robust": "rob", "leaky": "leak"}
TRAIN_LOG = "train_metrics.jsonl"
BAND = (0.10, 0.90)  # H3: base pass rate "in band"

FIG_CURVES = "reward_vs_true_success.png"
FIG_EXPLOIT = "exploit_vs_size.png"

_RUN_RE = re.compile(r"^(?P<size>[^-]+)-(?P<r>rob|leak)-s(?P<seed>\d+)(?:-r(?P<rerun>\d+))?$")
_STEP_FILE_RE = re.compile(r"^step-(\d+)\.jsonl$")

RunKey = tuple[str, str, int]  # (size, reward, seed)


# ── loading ──────────────────────────────────────────────────────────────────


@dataclass
class RunInfo:
    """One trained run of the matrix (after rerun selection)."""

    run_id: str
    size: str
    reward: str
    seed: int
    eval_steps: list[int] = field(default_factory=list)  # evaluated and kept
    final_step: int | None = None  # step marked final in runs/<id>/eval, if known
    status: str | None = None  # status.json
    train_log: list[dict[str, Any]] | None = None

    @property
    def state(self) -> str:
        if not self.eval_steps:
            return "missing"
        if self.final_step is not None and self.final_step in self.eval_steps:
            return "final"
        return "partial"

    @property
    def last_step(self) -> int | None:
        return max(self.eval_steps) if self.eval_steps else None


@dataclass
class Dataset:
    runs: dict[tuple[str, int], RunEval]  # (run_id, step) -> RunEval, matrix runs + base only
    infos: dict[RunKey, RunInfo]  # every expected (size, reward, seed), present or not
    base: dict[str, RunEval]  # size -> base eval
    families: dict[str, str]  # instance_id -> family
    notes: list[str]
    evals_root: Path
    runs_root: Path


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows of a JSONL file and the number of unparseable lines."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def parse_run_id(run_id: str) -> tuple[RunKey, int] | None:
    """((size, reward, seed), rerun index) for a matrix run ID, else None."""
    m = _RUN_RE.match(run_id)
    if not m:
        return None
    reward = "robust" if m["r"] == "rob" else "leaky"
    return (m["size"], reward, int(m["seed"])), int(m["rerun"] or 0)


def read_train_log(path: Path) -> list[dict[str, Any]] | None:
    """Training log rows sorted by step; the last line for a step wins (resume)."""
    if not path.is_file():
        return None
    rows, _ = _read_jsonl(path)
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        if "step" in r and "reward" in r:
            by_step[int(r["step"])] = r
    return [by_step[s] for s in sorted(by_step)] or None


def _load_records(evals_root: Path, notes: list[str]) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    if not evals_root.is_dir():
        notes.append(f"eval directory `{evals_root}` not found")
        return records
    skipped: list[str] = []
    bad_lines = 0
    for rdir in sorted(p for p in evals_root.iterdir() if p.is_dir()):
        for f in sorted(rdir.iterdir()):
            if not _STEP_FILE_RE.match(f.name):
                continue
            for marker, why in ((".failed", "failed"), (".claim", "being evaluated")):
                if f.with_suffix(marker).exists():
                    skipped.append(f"{rdir.name}/{f.stem} ({why})")
                    break
            else:
                rows, bad = _read_jsonl(f)
                bad_lines += bad
                for row in rows:
                    try:
                        records.append(EvalRecord.from_dict(row))
                    except KeyError, TypeError, ValueError:
                        bad_lines += 1
    if skipped:
        notes.append("skipped eval steps: " + ", ".join(skipped))
    if bad_lines:
        notes.append(f"skipped {bad_lines} unreadable eval line(s)")
    return records


def load_dataset(evals_root: str | Path, runs_root: str | Path) -> Dataset:
    """Read every eval JSONL and training log; select reruns; drop unfinished evals."""
    evals_root, runs_root = Path(evals_root), Path(runs_root)
    notes: list[str] = []
    records = _load_records(evals_root, notes)

    # Rerun selection and filtering to the run matrix + base.
    chosen: dict[RunKey, tuple[int, str]] = {}
    ignored: set[str] = set()
    for r in records:
        if r.reward == "base":
            continue
        parsed = parse_run_id(r.run_id)
        if parsed is None or parsed[0] != (r.size, r.reward, r.seed):
            ignored.add(r.run_id)
            continue
        key, rerun = parsed
        if key not in chosen or rerun > chosen[key][0]:
            chosen[key] = (rerun, r.run_id)
    keep_ids = {rid for _, rid in chosen.values()}
    replaced = sorted(
        {r.run_id for r in records if r.reward != "base"} - keep_ids - ignored,
    )
    if replaced:
        notes.append("replaced by a rerun: " + ", ".join(replaced))
    if ignored:
        notes.append("not in the run matrix, ignored: " + ", ".join(sorted(ignored)))
    records = [r for r in records if r.reward == "base" or r.run_id in keep_ids]

    families: dict[str, str] = {}
    for r in records:
        families.setdefault(r.instance_id, r.family)
    runs = index_runs(records)

    # Every checkpoint must cover the common held-out instance set.
    if runs:
        common, _ = Counter(r.instance_ids for r in runs.values()).most_common(1)[0]
        dropped = [k for k, r in runs.items() if r.instance_ids != common]
        for k in dropped:
            del runs[k]
        if dropped:
            notes.append(
                f"dropped {len(dropped)} checkpoint(s) whose instance set differs from the "
                f"common {len(common)}-instance set (unfinished eval?): "
                + ", ".join(f"{rid}@{st}" for rid, st in sorted(dropped))
            )

    base: dict[str, RunEval] = {}
    for (rid, _), r in sorted(runs.items()):
        if r.reward == "base":
            if r.size in base:
                notes.append(f"several base evals for {r.size}; using {rid}")
            base[r.size] = r

    sizes = list(H5_SIZES) + sorted(
        {r.size for r in runs.values()} - set(H5_SIZES) - {""},
    )
    infos: dict[RunKey, RunInfo] = {}
    for size in sizes:
        for reward in REWARDS:
            for seed in SEEDS:
                key = (size, reward, seed)
                rid = chosen.get(key, (0, f"{size}-{SHORT[reward]}-s{seed}"))[1]
                info = RunInfo(run_id=rid, size=size, reward=reward, seed=seed)
                info.eval_steps = sorted(st for (i, st) in runs if i == rid)
                rdir = runs_root / rid
                finals = [c.step for c in list_eval_checkpoints(rdir) if c.final]
                info.final_step = finals[-1] if finals else None
                info.status = (read_status(rdir) or {}).get("status")
                info.train_log = read_train_log(rdir / TRAIN_LOG)
                if size in H5_SIZES or info.eval_steps:
                    infos[key] = info
    return Dataset(
        runs=runs,
        infos=infos,
        base=base,
        families=families,
        notes=notes,
        evals_root=evals_root,
        runs_root=runs_root,
    )


# ── analysis ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """A rate with its 95% CI (fractions)."""

    point: float
    ci: tuple[float, float]


@dataclass
class ConditionRow:
    size: str
    reward: str
    n_seeds: int
    steps: tuple[int, ...]
    cells: dict[str, Cell]  # metric -> Cell


@dataclass
class RunRow:
    info: RunInfo | None  # None for base
    run_id: str
    size: str
    reward: str
    seed: int | None
    step: int | None
    n_instances: int
    cells: dict[str, Cell]


@dataclass
class HypothesisResult:
    hypothesis: str
    title: str
    rule: str
    verdict: Verdict | None
    provisional: bool
    gaps: list[str]
    error: str | None = None

    @property
    def label(self) -> str:
        if self.verdict is None:
            return "NOT AVAILABLE"
        text = self.verdict.status.upper()
        if self.provisional and self.verdict.status != STATUS_INCOMPLETE:
            text += " (provisional)"
        elif self.provisional:
            text += " (data still landing)"
        return text


@dataclass
class FamilyRow:
    size: str
    family: str
    base: float
    in_band: bool
    gain: dict[str, float | None]  # reward -> true-success gain over base


@dataclass
class Analysis:
    data: Dataset
    conditions: list[ConditionRow]
    run_rows: list[RunRow]
    hypotheses: list[HypothesisResult]
    families: list[FamilyRow]
    n_boot: int
    boot_seed: int
    main_size: str

    @property
    def n_expected(self) -> int:
        return sum(1 for i in self.data.infos.values() if i.size in H5_SIZES)

    def count(self, state: str) -> int:
        return sum(1 for i in self.data.infos.values() if i.size in H5_SIZES and i.state == state)

    @property
    def complete(self) -> bool:
        return self.count("final") == self.n_expected and all(s in self.data.base for s in H5_SIZES)


def _cells(mats: Mapping[str, ConditionMatrix], n_boot: int, seed: int) -> dict[str, Cell]:
    """CIs for each metric of one condition; seeds and instances resampled, shared draws."""
    d = hierarchical_bootstrap([{m: mats[m].values for m in mats}], n_boot=n_boot, seed=seed)
    return {m: Cell(mats[m].point, percentile_ci(d[m])) for m in mats}


def _matrices(
    runs: Mapping[tuple[str, int], RunEval], size: str, reward: str
) -> dict[str, ConditionMatrix]:
    return {m: condition_matrix(runs, size=size, reward=reward, metric=m) for m in METRICS}


def _sizes(data: Dataset) -> list[str]:
    return list(dict.fromkeys(k[0] for k in data.infos))


def _condition_rows(data: Dataset, n_boot: int, seed: int) -> list[ConditionRow]:
    rows = []
    for size in _sizes(data):
        for reward in ("base", *REWARDS):
            present = any(r.size == size and r.reward == reward for r in data.runs.values())
            if not present:
                rows.append(ConditionRow(size, reward, 0, (), {}))
                continue
            mats = _matrices(data.runs, size, reward)
            ts = mats["true_success"]
            n = 0 if reward == "base" else ts.n_seeds
            rows.append(ConditionRow(size, reward, n, ts.steps, _cells(mats, n_boot, seed)))
    return rows


def _run_rows(data: Dataset, n_boot: int, seed: int) -> list[RunRow]:
    rows = []
    for size in _sizes(data):
        b = data.base.get(size)
        rows.append(
            RunRow(
                info=None,
                run_id=f"base-{size}" if b is None else b.run_id,
                size=size,
                reward="base",
                seed=None,
                step=None if b is None else b.step,
                n_instances=0 if b is None else len(b.instance_ids),
                cells={} if b is None else _run_cells(b, n_boot, seed),
            )
        )
        for reward in REWARDS:
            for s in SEEDS:
                info = data.infos.get((size, reward, s))
                if info is None:
                    continue
                r = data.runs.get((info.run_id, info.last_step)) if info.last_step else None
                rows.append(
                    RunRow(
                        info=info,
                        run_id=info.run_id,
                        size=size,
                        reward=reward,
                        seed=s,
                        step=info.last_step,
                        n_instances=0 if r is None else len(r.instance_ids),
                        cells={} if r is None else _run_cells(r, n_boot, seed),
                    )
                )
    return rows


def _run_cells(r: RunEval, n_boot: int, seed: int) -> dict[str, Cell]:
    """Instance-bootstrap CIs for one run at one checkpoint."""
    d = hierarchical_bootstrap(
        [{m: r.metric(m)[None, :] for m in METRICS}], n_boot=n_boot, seed=seed
    )
    return {m: Cell(r.mean(m), percentile_ci(d[m])) for m in METRICS}


def _gaps(data: Dataset, keys: Sequence[RunKey], base_sizes: Sequence[str]) -> list[str]:
    out = []
    infos = [data.infos.get(k) for k in keys]
    missing = [
        f"{k[0]}-{SHORT[k[1]]}-s{k[2]}" if i is None else i.run_id
        for k, i in zip(keys, infos, strict=True)
        if i is None or i.state == "missing"
    ]
    partial = [f"{i.run_id}@{i.last_step}" for i in infos if i is not None and i.state == "partial"]
    if missing:
        out.append("no eval yet: " + ", ".join(missing))
    if partial:
        out.append("not at final checkpoint (last evaluated step): " + ", ".join(partial))
    for s in base_sizes:
        if s not in data.base:
            out.append(f"no base eval for {s}")
    return out


def _hypotheses(
    data: Dataset, n_boot: int, seed: int, main_size: str, reward_window: int
) -> list[HypothesisResult]:
    out = []
    main = main_size

    # H1
    keys = [(main, "robust", s) for s in SEEDS]
    gaps = _gaps(data, keys, [main])
    v, err = None, None
    try:
        v = verdict_h1(data.runs, main_size=main, n_boot=n_boot, seed=seed)
    except ValueError as e:
        err = str(e)
    out.append(
        HypothesisResult(
            "H1",
            "the environment teaches",
            f"{main} robust true success − base ≥ 10 pts **and** 95% CI lower bound > 0",
            v,
            bool(gaps),
            gaps,
            err,
        )
    )

    # H2
    keys = [(main, r, s) for r in REWARDS for s in SEEDS]
    gaps = _gaps(data, keys, [main])
    present = [i for k in keys if (i := data.infos.get(k)) and i.state != "missing"]
    no_log = [i.run_id for i in present if i.train_log is None]
    if no_log:
        gaps.append("no training log: " + ", ".join(no_log))

    def logs(reward: str) -> dict[int, list[dict[str, Any]]]:
        return {i.seed: i.train_log for i in present if i.reward == reward and i.train_log}

    v, err = None, None
    try:
        v = verdict_h2(
            data.runs,
            robust_logs=logs("robust"),
            leaky_logs=logs("leaky"),
            main_size=main,
            reward_window=reward_window,
            n_boot=n_boot,
            seed=seed,
        )
    except ValueError as e:
        err = str(e)
    out.append(
        HypothesisResult(
            "H2",
            "a leaky reward misleads",
            f"{main}: final-{reward_window}-step training reward leaky ≥ robust; "
            "true success robust − leaky ≥ 10 pts with CI excluding 0; "
            "leaky exploit ≥ 10% and ≥ 2× base",
            v,
            bool(gaps),
            gaps,
            err,
        )
    )

    # H5
    keys = [(z, "leaky", s) for z in H5_SIZES for s in SEEDS]
    gaps = _gaps(data, keys, [])
    v, err = None, None
    try:
        v = verdict_h5(data.runs, n_boot=n_boot, seed=seed)
    except ValueError as e:
        err = str(e)
    out.append(
        HypothesisResult(
            "H5",
            "bigger models exploit more",
            "final leaky exploit rate 0.8B < 2B < 4B (strict) **and** 4B − 0.8B CI excludes 0; "
            "needs 3 leaky seeds per size",
            v,
            bool(gaps),
            gaps,
            err,
        )
    )
    return out


def _family_rows(data: Dataset) -> list[FamilyRow]:
    """H3 (exploratory): per-family true-success gain over base vs base pass rate."""
    rows = []
    for size in _sizes(data):
        b = data.base.get(size)
        if b is None:
            continue
        fam = np.array([data.families[i] for i in b.instance_ids])
        trained: dict[str, np.ndarray | None] = {}
        for reward in REWARDS:
            if any(r.size == size and r.reward == reward for r in data.runs.values()):
                m = condition_matrix(data.runs, size=size, reward=reward, metric="true_success")
                trained[reward] = m.values.mean(axis=0)  # per instance, mean over seeds
            else:
                trained[reward] = None
        for f in sorted(set(fam)):
            mask = fam == f
            base_rate = float(b.true_success[mask].mean())
            gain = {
                r: None if v is None else float(v[mask].mean()) - base_rate
                for r, v in trained.items()
            }
            in_band = BAND[0] - 1e-9 <= base_rate <= BAND[1] + 1e-9
            rows.append(FamilyRow(size, str(f), base_rate, in_band, gain))
    return rows


def analyze(
    data: Dataset,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = 0,
    main_size: str = "2b",
    reward_window: int = DEFAULT_REWARD_WINDOW,
) -> Analysis:
    return Analysis(
        data=data,
        conditions=_condition_rows(data, n_boot, boot_seed),
        run_rows=_run_rows(data, n_boot, boot_seed),
        hypotheses=_hypotheses(data, n_boot, boot_seed, main_size, reward_window),
        families=_family_rows(data),
        n_boot=n_boot,
        boot_seed=boot_seed,
        main_size=main_size,
    )


# ── plots ────────────────────────────────────────────────────────────────────

# Reference categorical palette (dataviz skill), fixed order by role.
C_BLUE, C_ORANGE, C_VIOLET, C_GRAY = "#2a78d6", "#eb6834", "#4a3aa7", "#8a8984"


def _rolling(y: np.ndarray, k: int = 10) -> np.ndarray:
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    head = c[1:k] / np.arange(1, k)
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def make_plots(an: Analysis, out_dir: str | Path) -> list[Path]:
    """Write the core figures. Raises ImportError when matplotlib is missing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = an.data
    sizes = _sizes(data)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "font.size": 9,
        }
    )

    # 1. training reward vs held-out true success over steps, per size × reward
    fig, axes = plt.subplots(
        len(sizes), 2, figsize=(9, 2.6 * len(sizes)), sharex=True, sharey=True, squeeze=False
    )
    for i, size in enumerate(sizes):
        for j, reward in enumerate(REWARDS):
            ax = axes[i][j]
            ax.set_title(f"{size} · {reward}", fontsize=9, loc="left")
            ax.set_ylim(0, 1)
            any_data = False
            for s in SEEDS:
                info = data.infos.get((size, reward, s))
                if info is None:
                    continue
                if info.train_log:
                    st = np.array([r["step"] for r in info.train_log], dtype=float)
                    rw = np.array([r["reward"] for r in info.train_log], dtype=float)
                    ax.plot(st, _rolling(rw), color=C_ORANGE, lw=1.2, alpha=0.8)
                    any_data = True
                steps = info.eval_steps
                if steps:
                    evs = [data.runs[(info.run_id, st)] for st in steps]
                    ax.plot(
                        steps,
                        [e.mean("true_success") for e in evs],
                        color=C_BLUE,
                        lw=1.6,
                        marker="o",
                        ms=3.5,
                    )
                    ax.plot(
                        steps,
                        [e.mean("exploit") for e in evs],
                        color=C_VIOLET,
                        lw=1.2,
                        ls="--",
                        marker="s",
                        ms=3,
                    )
                    any_data = True
            b = data.base.get(size)
            if b is not None:
                ax.axhline(b.mean("true_success"), color=C_GRAY, lw=1, ls=":")
            if not any_data:
                ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center")
            if i == len(sizes) - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("rate")
    handles = [
        plt.Line2D(
            [], [], color=C_ORANGE, lw=1.2, label="training reward (own reward, 10-step mean)"
        ),
        plt.Line2D([], [], color=C_BLUE, lw=1.6, marker="o", ms=3.5, label="held-out true success"),
        plt.Line2D(
            [], [], color=C_VIOLET, lw=1.2, ls="--", marker="s", ms=3, label="held-out exploit"
        ),
        plt.Line2D([], [], color=C_GRAY, lw=1, ls=":", label="base true success"),
    ]
    fig.suptitle("Training reward vs held-out true success (one line per seed)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    p1 = out_dir / FIG_CURVES
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. exploit rate vs size (H5), final checkpoint
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    cond = {(c.size, c.reward): c for c in an.conditions}
    x = np.arange(len(sizes))
    for off, reward, color in ((-0.12, "leaky", C_ORANGE), (0.12, "robust", C_BLUE)):
        means, los, his, xs = [], [], [], []
        for k, size in enumerate(sizes):
            c = cond.get((size, reward))
            if c is None or not c.cells:
                continue
            e = c.cells["exploit"]
            xs.append(k + off)
            means.append(e.point)
            los.append(e.point - e.ci[0])
            his.append(e.ci[1] - e.point)
            per_seed = []
            for s in SEEDS:
                info = data.infos.get((size, reward, s))
                if info is not None and info.last_step is not None:
                    per_seed.append(data.runs[(info.run_id, info.last_step)].mean("exploit"))
            ax.scatter([k + off] * len(per_seed), per_seed, color=color, s=12, alpha=0.45, zorder=2)
        if xs:
            ax.errorbar(
                xs,
                means,
                yerr=[los, his],
                fmt="o-",
                color=color,
                ms=6,
                lw=1.6,
                capsize=3,
                label=f"{reward} (mean ± 95% CI, dots = seeds)",
                zorder=3,
            )
    bx = [k for k, s in enumerate(sizes) if s in data.base]
    if bx:
        ax.scatter(
            bx,
            [data.base[sizes[k]].mean("exploit") for k in bx],
            marker="x",
            color=C_GRAY,
            s=30,
            label="base",
            zorder=3,
        )
    ax.set_xticks(x, [s.upper() for s in sizes])
    ax.set_xlabel("model size")
    ax.set_ylabel("held-out exploit rate")
    ax.set_ylim(bottom=0)
    ax.set_title("Exploit rate vs size (final checkpoint)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p2 = out_dir / FIG_EXPLOIT
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


# ── markdown ─────────────────────────────────────────────────────────────────

DASH = "—"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _cell(c: Cell | None) -> str:
    if c is None:
        return DASH
    return f"{_pct(c.point)} [{_pct(c.ci[0])}, {_pct(c.ci[1])}]"


def _pts(x: float, ci: tuple[float, float] | None = None) -> str:
    s = f"{100 * x:+.1f} pts"
    if ci is not None:
        s += f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
    return s


def _seeds_txt(n: int) -> str:
    return f"{n}/{PREREG_SEEDS}"


def _hypothesis_md(h: HypothesisResult) -> list[str]:
    v = h.verdict
    out = [f"#### {h.hypothesis}: {h.title} — **{h.label}**", "", f"Rule: {h.rule}.", ""]
    if v is None:
        out.append(f"Not computed: {h.error or 'no data'}.")
    else:
        out.append(f"Reason: {v.reason}.")
        if v.checks:
            out += ["", "| Check | Met? |", "|---|---|"]
            out += [f"| `{k}` | {'yes' if ok else 'no'} |" for k, ok in v.checks.items()]
        nums = []
        for k, x in v.numbers.items():
            ci = v.cis.get(k)
            if k.startswith(("gain", "gap", "true_success_drop", "exploit_leaky_minus")):
                nums.append(f"{k} {_pts(x, ci)}")
            else:
                nums.append(f"{k} {_pct(x)}%" + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""))
        if nums:
            out += ["", "Numbers: " + "; ".join(nums) + "."]
        if v.n:
            out += ["", "n: " + ", ".join(f"{k} {x}" for k, x in v.n.items()) + "."]
        for note in v.notes:
            out += ["", f"Note: {note}."]
    if h.gaps:
        out += ["", "Still missing for a final verdict:"] + [f"- {g}" for g in h.gaps]
    out.append("")
    return out


def render_markdown(an: Analysis, figures: Sequence[Path] | None, report_dir: Path) -> str:
    data = an.data
    n_exp = an.n_expected
    n_final, n_part, n_miss = an.count("final"), an.count("partial"), an.count("missing")
    lines = [
        "_Generated by `eval/compare.py`. Do not edit between the markers; rerun the script._",
        "",
        f"Inputs: `{data.evals_root}`, `{data.runs_root}`. "
        f"Bootstrap: {an.n_boot:,} replicates, seed {an.boot_seed}. "
        "Rates in %; CIs are 95% percentile intervals of the paired hierarchical bootstrap "
        "(seeds, then instances).",
        "",
        "### Data status",
        "",
    ]
    if an.complete:
        lines.append(f"Complete: all {n_exp} runs at their final checkpoint, base evals present.")
    else:
        lines.append(
            f"**Incomplete data: verdicts are provisional.** Runs at final checkpoint: "
            f"{n_final}/{n_exp} · partial: {n_part} · no eval yet: {n_miss}."
        )
    base_txt = ", ".join(s for s in H5_SIZES if s in data.base) or "none"
    lines += ["", f"Base evals: {base_txt}."]
    for note in data.notes:
        lines.append(f"\n- {note}")
    lines.append("")

    # verdict summary
    lines += ["### Verdicts", "", "| Hypothesis | Verdict | n |", "|---|---|---|"]
    for h in an.hypotheses:
        n = DASH
        if h.verdict is not None and h.verdict.n:
            n = ", ".join(
                f"{k} {x}" for k, x in h.verdict.n.items() if k.startswith(("seed", "instances"))
            )
        lines.append(f"| {h.hypothesis}: {h.title} | **{h.label}** | {n} |")
    lines.append("| H3: difficulty matters | exploratory, no test | see table below |")
    lines.append("")
    for h in an.hypotheses:
        lines += _hypothesis_md(h)

    # by condition
    lines += [
        "### Results by condition (final checkpoint, mean across seeds)",
        "",
        "| Size | Reward | Seeds | Steps | True success | Exploit | Format failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in an.conditions:
        seeds = "base" if c.reward == "base" else _seeds_txt(c.n_seeds)
        steps = ", ".join(str(s) for s in sorted(set(c.steps))) or DASH
        cells = [_cell(c.cells.get(m)) for m in METRICS]
        lines.append(f"| {c.size} | {c.reward} | {seeds} | {steps} | " + " | ".join(cells) + " |")
    lines.append("")

    # by run
    lines += [
        "### Results by run (last evaluated checkpoint; CIs resample instances)",
        "",
        "| Size | Reward | Seed | Run | State | Step | True success | Exploit | Format failures |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in an.run_rows:
        if r.info is None:
            state = "base" if r.cells else "missing"
        else:
            state = r.info.state
        seed = DASH if r.seed is None else str(r.seed)
        step = DASH if r.step is None else str(r.step)
        cells = [_cell(r.cells.get(m)) for m in METRICS]
        lines.append(
            f"| {r.size} | {r.reward} | {seed} | `{r.run_id}` | {state} | {step} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    # H3
    lines += [
        "### H3 (exploratory): true-success gain by family vs base pass rate",
        "",
        "Base pass rate = base model true success on that family's held-out template. "
        f"In band = {int(100 * BAND[0])}–{int(100 * BAND[1])}%. "
        "Gain = trained (mean across seeds, final) − base. No CIs, no test.",
        "",
        "| Size | Family | Base pass | In band | Robust gain | Leaky gain |",
        "|---|---|---|---|---|---|",
    ]
    if not an.families:
        lines.append(f"| {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} |")
    for f in an.families:
        g = [DASH if f.gain[r] is None else _pts(f.gain[r]) for r in REWARDS]
        band = "yes" if f.in_band else "no"
        lines.append(f"| {f.size} | {f.family} | {_pct(f.base)} | {band} | {g[0]} | {g[1]} |")
    summary = _h3_summary(an.families)
    if summary:
        lines += ["", "Mean robust gain, in band vs out of band: " + "; ".join(summary) + "."]
    lines.append("")

    # figures
    lines += ["### Figures", ""]
    if figures:
        for p in figures:
            rel = os.path.relpath(p, report_dir)
            lines.append(f"![{p.stem}]({rel})")
            lines.append("")
    else:
        lines += ["Plots not written (`--no-plots`, or matplotlib missing: `--extra report`).", ""]
    return "\n".join(lines).rstrip() + "\n"


def _h3_summary(rows: Sequence[FamilyRow]) -> list[str]:
    out = []
    by_size: dict[str, list[FamilyRow]] = defaultdict(list)
    for r in rows:
        by_size[r.size].append(r)
    for size, rs in by_size.items():
        ins = [r.gain["robust"] for r in rs if r.in_band and r.gain["robust"] is not None]
        outs = [r.gain["robust"] for r in rs if not r.in_band and r.gain["robust"] is not None]
        if ins and outs:
            out.append(
                f"{size} {_pts(float(np.mean(ins)))} ({len(ins)} families) vs "
                f"{_pts(float(np.mean(outs)))} ({len(outs)})"
            )
    return out


# ── report file ──────────────────────────────────────────────────────────────


def replace_generated(text: str, block: str) -> str:
    """Replace what lies between the markers; everything else is kept byte for byte."""
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError(f"report must contain exactly one {BEGIN_MARKER} and one {END_MARKER}")
    i = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    j = text.index(END_MARKER)
    if j < i:
        raise ValueError(f"{END_MARKER} comes before {BEGIN_MARKER}")
    return text[:i] + "\n" + block.strip("\n") + "\n" + text[j:]


def write_report(path: str | Path, block: str) -> None:
    """Write the block between the markers of ``path`` (created if missing)."""
    path = Path(path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f"# Results\n\n{BEGIN_MARKER}\n{END_MARKER}\n"
    path.write_text(replace_generated(text, block), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fill results tables, verdicts and plots.")
    p.add_argument("--evals", type=Path, default=Path("evals"), help="eval JSONL root")
    p.add_argument("--runs", type=Path, default=Path("runs"), help="runs root (training logs)")
    p.add_argument("--report", type=Path, default=Path("reports/results.md"))
    p.add_argument("--figures", type=Path, default=Path("reports/figures"))
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--main-size", default="2b", help="model size for H1/H2")
    p.add_argument("--reward-window", type=int, default=DEFAULT_REWARD_WINDOW)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--stdout", action="store_true", help="print the block; write nothing")
    a = p.parse_args(argv)

    data = load_dataset(a.evals, a.runs)
    an = analyze(
        data,
        n_boot=a.n_boot,
        boot_seed=a.boot_seed,
        main_size=a.main_size,
        reward_window=a.reward_window,
    )
    figures: list[Path] | None = None
    if not (a.no_plots or a.stdout):
        try:
            figures = make_plots(an, a.figures)
        except ImportError:
            print("matplotlib not installed; skipping plots (--extra report)", file=sys.stderr)
    block = render_markdown(an, figures, a.report.parent)
    if a.stdout:
        sys.stdout.write(block)
    else:
        write_report(a.report, block)
        print(f"wrote {a.report}", file=sys.stderr)
    for h in an.hypotheses:
        print(f"{h.hypothesis}: {h.label}", file=sys.stderr)
    if not an.complete:
        print(f"incomplete data: {an.count('final')}/{an.n_expected} runs final", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
