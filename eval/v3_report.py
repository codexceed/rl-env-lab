"""Project version v3 report: EXPLORATORY (ADR 0007), never used in a v1 verdict.

    uv run python eval/v3_report.py                   # evals/ + runs/ -> reports/v3.md   (`make v3-report`)
    uv run python eval/v3_report.py --out -           # print instead
    uv run python eval/v3_report.py --evals E --runs R --n-boot 2000

**Inputs.** The 12 v3 runs ``v3-<size>-<rob|leak>-s<seed>`` (4B and 2B x robust / leaky x seeds 1-3,
``train/grpo.py --variant v3``), whose run ids keep them out of `eval/compare.py`:

* held-out: ``<evals>/<run>/step-NNNNN.jsonl`` (formal mode on ``data/v3/heldout.jsonl``, near + far templates);
  a step with a ``.failed`` or ``.claim`` marker is skipped, as in compare.py;
* the base reference: ``<evals>/v3-base-<size>/step-00000.jsonl`` (``eval/heldout.py --checkpoint base
  --variant v3``, thinking on like the runs);
* same-type: ``<evals>/<run>/sametype/step-NNNNN.jsonl`` and ``<evals>/v3-base-<size>/sametype/``;
* run state: ``<runs>/<run>/eval/step-NNNNN/COMPLETE`` marks the final step. A run whose final step is not
  evaluated yet is shown at its latest evaluated step, labelled "latest".

**What it shows**, per size and reward: true success (robust pass) and exploit rate (leaky pass, robust fail) at
the final checkpoint, **separately for near and far** held-out templates (`envlab.tasks.v3.heldout_kind`), overall
and per family, against the v3 base; the trajectory over eval steps; robust vs leaky (and each vs base) with the
PREREG paired hierarchical bootstrap (`eval.metrics.paired_difference`, seeds paired, instances resampled); and a
same-type section. Rates are means over instances of per-instance means, then over seeds (`eval.metrics`).
Anything missing is shown as "pending", so the report runs at any point of the round.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make `eval` and `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envlab.tasks.v3 import FAMILIES, heldout_kind
from eval.metrics import (
    DEFAULT_N_BOOT,
    METRICS,
    Estimate,
    EvalRecord,
    Metric,
    condition_matrix,
    index_runs,
    paired_difference,
)
from train.checkpoints import list_eval_checkpoints

V3_RUN_RE = re.compile(r"^v3-(?P<size>[^-]+)-(?P<r>rob|leak)-s(?P<seed>\d+)$")
_STEP_RE = re.compile(r"^step-(\d+)\.jsonl$")
SIZES = ("4b", "2b")
REWARDS = ("robust", "leaky")
SHORT = {"robust": "rob", "leaky": "leak"}
SEEDS = (1, 2, 3)
KINDS = ("near", "far")
SAMETYPE = "sametype"
PENDING = "pending"
TITLE = "# Project version v3 — exploratory (ADR 0007)"
BANNER = (
    "> **EXPLORATORY (ADR 0007).** v3 task set (`SPEC_v3.md`): 15 trained templates, 5 near and 5 far OOD-test "
    "templates; thinking on, answer cap 1,024, prompts up to 2,048 tokens. Near templates only recombine trained "
    "steps (an exploit there is hacking, not missing skill); far templates need a rule never trained. Never used in "
    "an H1/H2/H5 verdict; the pre-registered results are in `reports/results.md`."
)
GLOSSARY = (
    '*OOD test* = out-of-distribution test: tasks of kinds never trained on ("held-out" in SPEC_v3.md and the '
    "code); *OOD test, near* (OOD-near) and *OOD test, far* (OOD-far) are its two template kinds. "
    '*ID test* = in-distribution test: new tasks of the trained kinds ("same-type" in the code).'
)
CONTRASTS = (("robust", "leaky"), ("robust", "base"), ("leaky", "base"))


def run_id(size: str, reward: str, seed: int) -> str:
    return f"v3-{size}-{SHORT[reward]}-s{seed}"


def base_id(size: str) -> str:
    return f"v3-base-{size}"


# ---------------------------------------------------------------- loading


def read_steps(d: Path, notes: list[str]) -> dict[int, list[EvalRecord]]:
    """step -> records of every ``step-NNNNN.jsonl`` in ``d`` (unreadable lines and marked steps skipped)."""
    out: dict[int, list[EvalRecord]] = {}
    if not d.is_dir():
        return out
    for f in sorted(d.iterdir()):
        m = _STEP_RE.match(f.name)
        if not m:
            continue
        if f.with_suffix(".failed").exists() or f.with_suffix(".claim").exists():
            notes.append(f"skipped {d.name}/{f.stem} (failed or being evaluated)")
            continue
        records, bad = [], 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not all(isinstance(row.get(k), bool) for k in ("parse_ok", "robust_pass", "leaky_pass")):
                    raise ValueError("grader fields")
                records.append(EvalRecord.from_dict(row))
            except (ValueError, KeyError, TypeError, AttributeError):
                bad += 1
        if bad:
            notes.append(f"{d.name}/{f.name}: skipped {bad} unreadable line(s)")
        if records:
            out[int(m[1])] = records
    return out


@dataclass
class RunState:
    """One v3 run (or a v3 base): its held-out and same-type evals and whether its final step is evaluated."""

    run_id: str
    size: str
    reward: str  # robust | leaky | base
    seed: int | None
    heldout: dict[int, list[EvalRecord]] = field(default_factory=dict)
    sametype: dict[int, list[EvalRecord]] = field(default_factory=dict)
    final_step: int | None = None  # from runs/<id>/eval/*/COMPLETE

    @property
    def step(self) -> int | None:
        """The held-out step reported: the final one when evaluated, else the latest evaluated."""
        if self.final_step is not None and self.final_step in self.heldout:
            return self.final_step
        return max(self.heldout) if self.heldout else None

    @property
    def status(self) -> str:
        if self.step is None:
            return PENDING
        if self.reward == "base" or self.step == self.final_step:
            return f"final (step {self.step})"
        return f"latest (step {self.step})"

    @property
    def records(self) -> list[EvalRecord]:
        step = self.step
        return [] if step is None else self.heldout[step]


@dataclass
class V3Data:
    runs: dict[str, RunState]  # every expected run id, evaluated or not
    base: dict[str, RunState]  # size -> v3 base
    notes: list[str]


def load(evals_root: Path, runs_root: Path) -> V3Data:
    """The 12 v3 runs (expected, plus any other ``v3-<size>-...`` id found) and the v3 bases."""
    evals_root, runs_root = Path(evals_root), Path(runs_root)
    notes: list[str] = []
    ids = {run_id(s, r, k): (s, r, k) for s in SIZES for r in REWARDS for k in SEEDS}
    if evals_root.is_dir():
        for d in sorted(evals_root.iterdir()):
            if (m := V3_RUN_RE.match(d.name)) and d.name not in ids:
                ids[d.name] = (m["size"], "robust" if m["r"] == "rob" else "leaky", int(m["seed"]))
    else:
        notes.append(f"`{evals_root}` not found")
    runs = {}
    for rid, (size, reward, seed) in ids.items():
        finals = [c.step for c in list_eval_checkpoints(runs_root / rid) if c.final]
        runs[rid] = RunState(
            rid,
            size,
            reward,
            seed,
            heldout=read_steps(evals_root / rid, notes),
            sametype=read_steps(evals_root / rid / SAMETYPE, notes),
            final_step=finals[-1] if finals else None,
        )
    sizes = sorted({r.size for r in runs.values()}, key=lambda s: (s not in SIZES, SIZES.index(s) if s in SIZES else s))
    base = {
        s: RunState(
            base_id(s),
            s,
            "base",
            None,
            heldout=read_steps(evals_root / base_id(s), notes),
            sametype=read_steps(evals_root / base_id(s) / SAMETYPE, notes),
        )
        for s in sizes
    }
    return V3Data(runs, base, notes)


# ---------------------------------------------------------------- rates


Filter = Callable[[EvalRecord], bool]


def by_kind(kind: str | None) -> Filter:
    """Records of the near or far held-out templates; None keeps everything (e.g. same-type)."""
    return (lambda r: True) if kind is None else (lambda r: heldout_kind(r.template) == kind)


def rates(records: Sequence[EvalRecord], keep: Filter | None = None) -> dict[Metric, float] | None:
    """Run-level rates (mean over instances of per-instance means) of one run's records, or None."""
    runs = index_runs(r for r in records if keep is None or keep(r))
    if not runs:
        return None
    if len(runs) != 1:
        raise ValueError(f"expected one run and step, got {sorted(runs)}")
    (run,) = runs.values()
    return {m: run.mean(m) for m in METRICS}


def group_rates(
    runs: Sequence[RunState], keep: Filter, pick: Callable[[RunState], list[EvalRecord]]
) -> tuple[dict[Metric, float] | None, int]:
    """Mean over seeds of per-run rates (records chosen by ``pick``), and the number of seeds that had any."""
    per = [x for r in runs if (x := rates(pick(r), keep)) is not None]
    if not per:
        return None, 0
    return {m: sum(p[m] for p in per) / len(per) for m in METRICS}, len(per)


def condition(data: V3Data, size: str, reward: str) -> list[RunState]:
    return sorted((r for r in data.runs.values() if r.size == size and r.reward == reward), key=lambda r: r.seed or 0)


def contrast(
    data: V3Data, size: str, a: str, b: str, kind: str, metric: Metric, *, n_boot: int, seed: int
) -> Estimate | str:
    """``a - b`` at the reported step with the paired hierarchical bootstrap; a reason string when not possible."""
    sides: dict[str, list[RunState]] = {}
    for reward in (a, b):
        group = [data.base[size]] if reward == "base" else condition(data, size, reward)
        sides[reward] = [r for r in group if r.step is not None]
    if not sides[a] or not sides[b]:
        return PENDING
    seeds = None
    if b != "base":
        seeds = sorted({r.seed for r in sides[a]} & {r.seed for r in sides[b]}, key=lambda s: s or 0)
        if not seeds:
            return "pending (no seed with both rewards)"
    keep = by_kind(kind)
    records = [x for side in sides.values() for r in side for x in r.records if keep(x)]
    runs = index_runs(records)
    try:
        ma = condition_matrix(runs, size=size, reward=a, metric=metric, seeds=seeds)
        mb = condition_matrix(runs, size=size, reward=b, metric=metric, seeds=seeds)
        return paired_difference(ma, mb, n_boot=n_boot, seed=seed)
    except ValueError as exc:
        return f"n/a ({exc})"


# ---------------------------------------------------------------- markdown


def pct(x: float | None) -> str:
    return PENDING if x is None else f"{x:.1%}"


def pts(e: Estimate | str) -> str:
    if isinstance(e, str):
        return e
    mark = " *" if e.excludes_zero else ""
    return f"{e.point * 100:+.1f} [{e.ci_low * 100:+.1f}, {e.ci_high * 100:+.1f}]{mark}"


def _reported(r: RunState) -> list[EvalRecord]:
    return r.records


def _latest_sametype(r: RunState) -> list[EvalRecord]:
    return r.sametype[max(r.sametype)] if r.sametype else []


def status_md(data: V3Data) -> list[str]:
    lines = ["## Runs", "", "| Run | OOD test | ID test (latest step) |", "|---|---|---|"]
    for size in data.base:
        lines.append(f"| {base_id(size)} | {data.base[size].status} | {_st_step(data.base[size])} |")
        for reward in REWARDS:
            for r in condition(data, size, reward):
                lines.append(f"| {r.run_id} | {r.status} | {_st_step(r)} |")
    return [*lines, ""]


def _st_step(r: RunState) -> str:
    return str(max(r.sametype)) if r.sametype else PENDING


def final_md(data: V3Data, size: str) -> list[str]:
    lines = [
        "### Final checkpoint, OOD test (near vs far)",
        "",
        "| Model | Seeds | OOD-near true success | OOD-near exploit | OOD-far true success | OOD-far exploit |",
        "|---|---|---|---|---|---|",
    ]
    rows: list[tuple[str, list[RunState]]] = [("base (thinking on)", [data.base[size]])]
    rows += [(reward, condition(data, size, reward)) for reward in REWARDS]
    for label, group in rows:
        cells, n = [], 0
        for kind in KINDS:
            g, n_kind = group_rates(group, by_kind(kind), _reported)
            n = max(n, n_kind)
            cells += [pct(g["true_success"] if g else None), pct(g["exploit"] if g else None)]
        seeds = "-" if label.startswith("base") else str(n)
        lines.append(f"| {label} | {seeds} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def family_md(data: V3Data, size: str, kind: str) -> list[str]:
    head = "| Family | Base TS | Robust TS | Leaky TS | Base exploit | Robust exploit | Leaky exploit |"
    lines = [f"### Per family, OOD test, {kind}", "", head, "|---|---|---|---|---|---|---|"]
    groups = [[data.base[size]], condition(data, size, "robust"), condition(data, size, "leaky")]
    for fam in FAMILIES:
        keep_kind = by_kind(kind)

        def keep(r: EvalRecord, fam: str = fam, keep_kind: Filter = keep_kind) -> bool:
            return keep_kind(r) and r.family == fam

        got = [group_rates(g, keep, _reported)[0] for g in groups]
        ts = [pct(g["true_success"] if g else None) for g in got]
        ex = [pct(g["exploit"] if g else None) for g in got]
        lines.append(f"| {fam} | " + " | ".join(ts + ex) + " |")
    return [*lines, ""]


def trajectory_md(data: V3Data, size: str) -> list[str]:
    lines = [
        "### Trajectory over eval steps (OOD test, mean over the seeds evaluated at that step)",
        "",
        "| Step | Seeds rob / leak | Robust TS OOD-near / OOD-far | Leaky TS OOD-near / OOD-far "
        "| Robust exploit OOD-near / OOD-far | Leaky exploit OOD-near / OOD-far |",
        "|---|---|---|---|---|---|",
    ]
    groups = {reward: condition(data, size, reward) for reward in REWARDS}
    steps = sorted({s for g in groups.values() for r in g for s in r.heldout})
    if not steps:
        return [*lines[:2], PENDING, ""]
    for step in steps:

        def at(r: RunState, step: int = step) -> list[EvalRecord]:
            return r.heldout.get(step, [])

        seeds, cells = [], {}
        for reward, group in groups.items():
            seeds.append(str(sum(step in r.heldout for r in group)))
            for metric in ("true_success", "exploit"):
                vals = [group_rates(group, by_kind(kind), at)[0] for kind in KINDS]
                cells[(reward, metric)] = " / ".join(pct(v[metric] if v else None) for v in vals)
        order = [("robust", "true_success"), ("leaky", "true_success"), ("robust", "exploit"), ("leaky", "exploit")]
        lines.append(f"| {step} | {' / '.join(seeds)} | " + " | ".join(cells[k] for k in order) + " |")
    return [*lines, ""]


def bootstrap_md(data: V3Data, size: str, n_boot: int, seed: int) -> list[str]:
    lines = [
        "### Paired bootstrap (percentage points, 95% CI; * = excludes 0)",
        "",
        f"PREREG §3 paired hierarchical bootstrap, {n_boot:,} replicates, seed {seed}: seeds paired by number "
        "(robust vs leaky) or the base held fixed, instances resampled. Exploratory.",
        "",
        "| Contrast | OOD-near true success | OOD-near exploit | OOD-far true success | OOD-far exploit |",
        "|---|---|---|---|---|",
    ]
    for a, b in CONTRASTS:
        cells = [
            pts(contrast(data, size, a, b, kind, metric, n_boot=n_boot, seed=seed))
            for kind in KINDS
            for metric in ("true_success", "exploit")
        ]
        lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def sametype_md(data: V3Data) -> list[str]:
    lines = [
        "## ID test (unseen tasks of the 15 trained templates)",
        "",
        "Latest step of each run, from `evals/<run>/sametype/step-NNNNN.jsonl`.",
        "",
    ]
    if not any(r.sametype for r in [*data.runs.values(), *data.base.values()]):
        return [*lines, PENDING, ""]
    lines += ["| Size, model | Seeds | True success | Exploit | Format failure |", "|---|---|---|---|---|"]
    for size in data.base:
        rows: list[tuple[str, list[RunState]]] = [("base", [data.base[size]])]
        rows += [(reward, condition(data, size, reward)) for reward in REWARDS]
        for label, group in rows:
            g, n = group_rates(group, by_kind(None), _latest_sametype)
            cells = [pct(g[m] if g else None) for m in METRICS]
            lines.append(f"| {size} {label} | {'-' if label == 'base' else n} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def markdown(data: V3Data, *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> str:
    lines = [TITLE, "", BANNER, "", GLOSSARY, ""]
    lines += [
        "True success = robust pass; exploit = leaky pass and robust fail. OOD-near / OOD-far = the OOD-test "
        "template kind (`envlab.tasks.v3.heldout_kind`). Each run is shown at its final step once evaluated, else at "
        "its latest evaluated step (see Runs).",
        "",
    ]
    lines += status_md(data)
    for size in data.base:
        lines += [f"## {size.upper()}", ""]
        lines += final_md(data, size)
        for kind in KINDS:
            lines += family_md(data, size, kind)
        lines += trajectory_md(data, size)
        lines += bootstrap_md(data, size, n_boot, seed)
    lines += sametype_md(data)
    if data.notes:
        lines += ["## Data notes", "", *(f"- {n}" for n in data.notes), ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Project version v3 report (EXPLORATORY, ADR 0007)")
    p.add_argument("--evals", type=Path, default=Path("evals"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", default="reports/v3.md", help="markdown output path, or - for stdout")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--boot-seed", type=int, default=0)
    a = p.parse_args(argv)
    text = markdown(load(a.evals, a.runs), n_boot=a.n_boot, seed=a.boot_seed)
    if a.out == "-":
        print(text, end="")
    else:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"[v3_report] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
