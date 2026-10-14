"""End-of-study disk cleanup of a GPU node's run dirs (dry run by default).

    uv run python scripts/cleanup_node.py                 # `make cleanup-plan`: dry run, prints and writes the plan
    uv run python scripts/cleanup_node.py --apply         # deletes the plan; THE OWNER RUNS THIS, never an agent

**Project rule: an agent never runs ``--apply`` on real data. The owner reviews the dry-run plan and runs
``--apply`` themselves.**

**Policy** (what the hand cleanup at the end of the study applied; ~4 TB down to 236 GB on one node). Only
run dirs ``<runs>/<run_id>/`` whose id matches ``--run-pattern`` (default: the v1/v2/v3 run matrix,
``[v2-|v3-]<0.8b|2b|4b>-<rob|leak>-s<seed>``) are looked at. For each one:

* **Integrity first.** ``check_final_integrity`` (`train/checkpoints.py`) must pass. A run that fails it is
  reported and left untouched: its recovery state and every eval checkpoint stay.
* **Keep the final eval checkpoint** ``eval/step-NNNNN/`` (``COMPLETE`` with ``final: true``).
* **Keep the held-out peak** too when its true success (mean ``robust_pass`` in
  ``<evals>/<run_id>/step-NNNNN.jsonl``) beats the final step's by at least ``--peak-margin`` (default 0.10).
  Peaks are descriptive only (picking a step by its held-out score overstates it); they are kept so they can be
  re-studied. If the final step has no held-out score, the peak is kept and the run is flagged.
* **Delete every other** ``eval/step-NNNNN/`` dir (complete or partial) and the whole ``recovery/`` dir.

Nothing else is ever touched: not ``<evals>/``, not logs, not ``status.json``, not other run ids. Two opt-in
extras: ``--include-models`` (``<runs>/models``, the text-only base-model exports, re-made on demand by
``train/export_text.py``) and ``--include-hf-cache`` (the Hugging Face hub cache, ``--hf-cache``).

**Dry run** (default): a table per run (final step, peak with both scores and whether it is kept, bytes to
delete and to keep), totals, integrity refusals, and the plan written to ``--plan-out`` (one path per line).

**Apply** (``--apply``, owner only): refuses to start while training or eval processes run (``pgrep -f`` on
``train/grpo.py``, ``eval/heldout.py``, ``eval/sametype_batch.py``, ``eval/watcher.py``) unless
``--force-busy``. Rebuilds the plan, writes it to ``--plan-out``, then re-validates every path before
``shutil.rmtree`` (no shell): only ``<runs>/<run_id>/recovery`` and ``<runs>/<run_id>/eval/step-NNNNN`` of a
matching run id (or an enabled extra) are accepted, never through a symlink, never a final eval step, and a
recovery dir only when the run still passes ``check_final_integrity``. Prints what was deleted and what remains.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.checkpoints import (
    COMPLETE,
    check_final_integrity,
    list_eval_checkpoints,
    read_marker,
    recovery_root,
    step_name,
)

DEFAULT_RUN_PATTERN = r"^(v[23]-)?(0\.8b|2b|4b)-(rob|leak)-s\d$"
DEFAULT_PEAK_MARGIN = 0.10
BUSY_PATTERNS = ("train/grpo.py", "eval/heldout.py", "eval/sametype_batch.py", "eval/watcher.py")
PLAN_FILENAME = "cleanup_plan.txt"
MODELS_DIRNAME = "models"
APPLY_COMMAND = "uv run python scripts/cleanup_node.py --apply"
_STEP_DIR_RE = re.compile(r"^step-\d{5}$")
_SCORE_FILE_RE = re.compile(r"^step-(\d+)\.jsonl$")
_TOL = 1e-9  # so 0.40 - 0.30 counts as a 10-point gap despite float rounding

BusyCheck = Callable[[], list[str]]


# ---------------------------------------------------------------- measuring


def dir_size(path: Path) -> int:
    """Apparent size in bytes of everything under ``path`` (like ``du -sb``); symlinks are not followed."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
    return total


def true_success(records_path: Path) -> float | None:
    """Mean ``robust_pass`` over the samples of one held-out eval file; None if unreadable or empty."""
    passes = []
    try:
        with open(records_path) as fh:
            for line in fh:
                if line.strip():
                    passes.append(bool(json.loads(line).get("robust_pass", False)))
    except (OSError, ValueError, AttributeError):
        return None
    return sum(passes) / len(passes) if passes else None


def heldout_scores(evals_root: Path, run_id: str) -> dict[int, float]:
    """True success per step from ``<evals>/<run_id>/step-NNNNN.jsonl`` (traces and subdirs ignored)."""
    root = Path(evals_root) / run_id
    if not root.is_dir():
        return {}
    scores = {}
    for child in root.iterdir():
        m = _SCORE_FILE_RE.match(child.name)
        if m and child.is_file():
            score = true_success(child)
            if score is not None:
                scores[int(m.group(1))] = score
    return scores


def fmt_bytes(n: int) -> str:
    """Decimal units, as the node's disk usage was quoted (GB = 1e9 bytes)."""
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


# ---------------------------------------------------------------- planning


@dataclass
class RunPlan:
    run_id: str
    rdir: Path
    integrity_ok: bool
    integrity_reasons: list[str] = field(default_factory=list)
    final_step: int | None = None
    final_score: float | None = None
    peak_step: int | None = None
    peak_score: float | None = None
    peak_kept: bool = False
    keep: list[Path] = field(default_factory=list)
    delete: list[Path] = field(default_factory=list)
    delete_bytes: int = 0
    keep_bytes: int = 0  # everything left in the run dir after the deletions
    notes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    runs: list[RunPlan] = field(default_factory=list)
    extras: list[tuple[Path, int]] = field(default_factory=list)  # (path, bytes)

    @property
    def paths(self) -> list[Path]:
        """Every path to delete, in order."""
        return [p for r in self.runs for p in r.delete] + [p for p, _ in self.extras]

    @property
    def delete_bytes(self) -> int:
        return sum(r.delete_bytes for r in self.runs) + sum(n for _, n in self.extras)

    @property
    def keep_bytes(self) -> int:
        return sum(r.keep_bytes for r in self.runs)

    @property
    def refusals(self) -> list[RunPlan]:
        return [r for r in self.runs if not r.integrity_ok]


def plan_run(rdir: Path, evals_root: Path, peak_margin: float = DEFAULT_PEAK_MARGIN) -> RunPlan:
    """What to keep and delete in one run dir under the cleanup policy (module docstring).

    Args:
        rdir: The run dir ``<runs>/<run_id>``.
        evals_root: Held-out eval outputs root (``<evals>``); read only.
        peak_margin: Keep the held-out peak when it beats the final step's true success by at least this much.

    Returns:
        The run's plan; nothing is deleted.
    """
    rdir = Path(rdir)
    integrity = check_final_integrity(rdir)
    plan = RunPlan(run_id=rdir.name, rdir=rdir, integrity_ok=integrity.ok, integrity_reasons=list(integrity.reasons))
    total = dir_size(rdir)
    if not integrity.ok or integrity.checkpoint is None:
        plan.keep_bytes = total
        return plan

    final = integrity.checkpoint
    scores = heldout_scores(evals_root, rdir.name)
    plan.final_step, plan.final_score = final.step, scores.get(final.step)
    keep = {final.path}

    scored = [c for c in list_eval_checkpoints(rdir) if c.step in scores]
    if scored:
        best = max(scored, key=lambda c: (scores[c.step], -c.step))  # earliest step on ties
        plan.peak_step, plan.peak_score = best.step, scores[best.step]
        if best.step != final.step:
            if plan.final_score is None:
                plan.peak_kept = True
                plan.notes.append(f"final {step_name(final.step)} has no held-out score: peak kept")
            elif plan.peak_score - plan.final_score >= peak_margin - _TOL:
                plan.peak_kept = True
        if plan.peak_kept:
            keep.add(best.path)
    elif plan.final_score is None:
        plan.notes.append("no held-out scores found: only the final step kept")

    eval_root = rdir / "eval"
    step_dirs = sorted(p for p in eval_root.iterdir() if _STEP_DIR_RE.match(p.name)) if eval_root.is_dir() else []
    plan.keep = sorted(keep)
    plan.delete = [p for p in step_dirs if p not in keep and p.is_dir() and not p.is_symlink()]
    rec = recovery_root(rdir)
    if rec.is_dir() and not rec.is_symlink():
        plan.delete.append(rec)
    plan.delete_bytes = sum(dir_size(p) for p in plan.delete)
    plan.keep_bytes = total - plan.delete_bytes
    return plan


def select_runs(runs_root: Path, run_re: re.Pattern[str]) -> list[Path]:
    """Run dirs directly under ``runs_root`` whose id matches ``run_re`` (symlinked dirs are skipped)."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if run_re.match(p.name) and p.is_dir() and not p.is_symlink())


def build_plan(
    runs_root: Path,
    evals_root: Path,
    *,
    run_re: re.Pattern[str],
    peak_margin: float = DEFAULT_PEAK_MARGIN,
    extras: Sequence[Path] = (),
) -> Plan:
    """The cleanup plan for every matching run dir, plus the opt-in extra dirs that exist."""
    plan = Plan(runs=[plan_run(r, evals_root, peak_margin) for r in select_runs(runs_root, run_re)])
    plan.extras = [(Path(p), dir_size(Path(p))) for p in extras if Path(p).is_dir() and not Path(p).is_symlink()]
    return plan


def default_hf_cache() -> Path:
    """The Hugging Face hub cache dir, as `huggingface_hub` resolves it from the environment."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache") / "huggingface")
    return Path(home).expanduser() / "hub"


# ---------------------------------------------------------------- validating and applying


def validate_path(
    path: Path,
    runs_root: Path,
    run_re: re.Pattern[str],
    extras: Iterable[Path] = (),
) -> str | None:
    """Why ``path`` must not be deleted, or None when it is an allowed target.

    Allowed: ``<runs>/<run_id>/recovery`` or ``<runs>/<run_id>/eval/step-NNNNN`` of a run id matching
    ``run_re`` (an eval step must not be marked final), or exactly one of ``extras``. Symlinks are refused.

    Args:
        path: The candidate path.
        runs_root: The runs root the plan was built from.
        run_re: Run ids that may be touched.
        extras: Opt-in extra dirs enabled for this invocation.

    Returns:
        A refusal reason, or None.
    """
    path = Path(path)
    if path.is_symlink():
        return "is a symlink"
    if not path.exists():
        return "does not exist"
    resolved = path.resolve()
    root = Path(runs_root).resolve()
    allowed_extras = {Path(e).resolve() for e in extras}
    if resolved in allowed_extras:
        if len(resolved.parts) < 3 or resolved == Path.home().resolve() or resolved in (root, *root.parents):
            return "extra path is too broad"
        return None
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return f"outside the runs root {root}"
    if not parts or not run_re.match(parts[0]):
        return "not under a matching run id"
    if parts[1:] == ("recovery",):
        return None
    if len(parts) == 3 and parts[1] == "eval" and _STEP_DIR_RE.match(parts[2]):
        if read_marker(resolved / COMPLETE).get("final"):
            return "is the run's final eval checkpoint"
        return None
    return "matches neither <run_id>/recovery nor <run_id>/eval/step-NNNNN"


def busy_processes(patterns: Sequence[str] = BUSY_PATTERNS) -> list[str]:
    """Training or eval processes still running, as ``"<pattern>: <pids>"`` (via ``pgrep -f``)."""
    busy = []
    for pattern in patterns:
        try:
            out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            busy.append(f"{pattern}: cannot check ({exc})")
            continue
        pids = out.stdout.split()
        if pids:
            busy.append(f"{pattern}: pids {' '.join(pids)}")
    return busy


@dataclass
class ApplyResult:
    deleted: list[tuple[Path, int]] = field(default_factory=list)
    refused: list[tuple[Path, str]] = field(default_factory=list)
    busy: list[str] = field(default_factory=list)


def apply_plan(
    paths: Sequence[Path],
    runs_root: Path,
    run_re: re.Pattern[str],
    *,
    extras: Iterable[Path] = (),
    busy_check: BusyCheck = busy_processes,
    force_busy: bool = False,
) -> ApplyResult:
    """Delete the planned paths after re-validating each one (owner only; see the module docstring).

    Args:
        paths: The plan's paths.
        runs_root: The runs root the plan was built from.
        run_re: Run ids that may be touched.
        extras: Opt-in extra dirs enabled for this invocation.
        busy_check: Lists running training or eval processes; any hit refuses the whole apply.
        force_busy: Apply even when ``busy_check`` finds processes.

    Returns:
        What was deleted and refused; ``busy`` is set (and nothing deleted) when the apply was refused.
    """
    result = ApplyResult()
    busy = busy_check()
    if busy and not force_busy:
        result.busy = busy
        return result
    extras = list(extras)
    for path in paths:
        path = Path(path)
        reason = validate_path(path, runs_root, run_re, extras)
        if reason is None and path.name == "recovery":
            integrity = check_final_integrity(path.parent)
            if not integrity.ok:
                reason = "run fails check_final_integrity: " + "; ".join(integrity.reasons)
        if reason is not None:
            result.refused.append((path, reason))
            continue
        size = dir_size(path)
        shutil.rmtree(path)
        result.deleted.append((path, size))
    return result


# ---------------------------------------------------------------- reporting


def _score(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def format_plan(plan: Plan, peak_margin: float = DEFAULT_PEAK_MARGIN) -> str:
    """The dry-run table, totals and refusals."""
    head = (
        f"{'run':<18} {'final':>6} {'final TS':>8} {'peak':>6} {'peak TS':>8} {'peak kept':>9} "
        f"{'delete':>10} {'keep':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in plan.runs:
        if not r.integrity_ok:
            lines.append(
                f"{r.run_id:<18} {'REFUSED: fails integrity, nothing deleted':<52} {fmt_bytes(r.keep_bytes):>10}"
            )
            continue
        final = "-" if r.final_step is None else str(r.final_step)
        peak = "-" if r.peak_step is None else str(r.peak_step)
        kept = "yes" if r.peak_kept else ("-" if r.peak_step in (None, r.final_step) else "no")
        lines.append(
            f"{r.run_id:<18} {final:>6} {_score(r.final_score):>8} {peak:>6} {_score(r.peak_score):>8} {kept:>9} "
            f"{fmt_bytes(r.delete_bytes):>10} {fmt_bytes(r.keep_bytes):>10}"
        )
    for path, size in plan.extras:
        lines.append(f"extra (opt-in): {path}  delete {fmt_bytes(size)}")
    lines += [
        "",
        f"runs: {len(plan.runs)}   paths to delete: {len(plan.paths)}   "
        f"delete: {fmt_bytes(plan.delete_bytes)}   keep (selected run dirs): {fmt_bytes(plan.keep_bytes)}",
        f"peak kept when its true success beats the final step's by >= {peak_margin:.2f} (descriptive only)",
    ]
    notes = [(r.run_id, n) for r in plan.runs for n in r.notes]
    if notes:
        lines += ["", "notes:"] + [f"  {run}: {n}" for run, n in notes]
    if plan.refusals:
        lines += ["", "integrity refusals (recovery and eval checkpoints kept):"]
        lines += [f"  {r.run_id}: {'; '.join(r.integrity_reasons)}" for r in plan.refusals]
    return "\n".join(lines)


def format_remaining(plan: Plan) -> str:
    """What is left in each selected run dir after an apply."""
    lines = ["remaining:"]
    for r in plan.runs:
        eval_root = r.rdir / "eval"
        steps = sorted(p.name for p in eval_root.iterdir() if _STEP_DIR_RE.match(p.name)) if eval_root.is_dir() else []
        rec = "recovery kept" if recovery_root(r.rdir).exists() else "no recovery"
        lines.append(f"  {r.run_id:<18} {fmt_bytes(dir_size(r.rdir)):>10}  {rec}; eval: {', '.join(steps) or '-'}")
    return "\n".join(lines)


def write_plan(paths: Sequence[Path], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{p}\n" for p in paths))


# ---------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Dry run by default. To delete, the OWNER (never an agent) runs: {APPLY_COMMAND}",
    )
    ap.add_argument("--runs-root", type=Path, default=Path("runs"), help="run dirs root (default: runs)")
    ap.add_argument("--evals-root", type=Path, default=Path("evals"), help="held-out eval outputs root (read only)")
    ap.add_argument(
        "--plan-out", type=Path, help=f"plan file, one path per line (default: <runs-root>/{PLAN_FILENAME})"
    )
    ap.add_argument("--run-pattern", default=DEFAULT_RUN_PATTERN, help="regex of run ids to touch")
    ap.add_argument("--peak-margin", type=float, default=DEFAULT_PEAK_MARGIN, help="keep the peak above this gain")
    ap.add_argument("--include-models", action="store_true", help="also delete <runs-root>/models (base exports)")
    ap.add_argument("--include-hf-cache", action="store_true", help="also delete the Hugging Face hub cache")
    ap.add_argument("--hf-cache", type=Path, default=None, help="hub cache dir (default: from HF_HUB_CACHE/HF_HOME)")
    ap.add_argument("--apply", action="store_true", help=f"DELETE the plan. Owner only: {APPLY_COMMAND}")
    ap.add_argument("--force-busy", action="store_true", help="apply even if training/eval processes are running")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, busy_check: BusyCheck = busy_processes) -> int:
    """Print (and with ``--apply``, carry out) the cleanup plan.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.
        busy_check: Lists running training or eval processes (injected by tests).

    Returns:
        0 on success; 1 when the apply was refused (busy processes) or any path was refused.
    """
    args = parse_args(argv)
    run_re = re.compile(args.run_pattern)
    extras = []
    if args.include_models:
        extras.append(args.runs_root / MODELS_DIRNAME)
    if args.include_hf_cache:
        extras.append(args.hf_cache or default_hf_cache())
    plan_out = args.plan_out or args.runs_root / PLAN_FILENAME

    plan = build_plan(args.runs_root, args.evals_root, run_re=run_re, peak_margin=args.peak_margin, extras=extras)
    print(format_plan(plan, args.peak_margin))
    write_plan(plan.paths, plan_out)
    print(f"\nplan ({len(plan.paths)} paths) written to {plan_out}")

    if not args.apply:
        busy = busy_check()
        if busy:
            print("\nwarning: training or eval processes are running (--apply would refuse):")
            print("\n".join(f"  {b}" for b in busy))
        print(f"\nDRY RUN: nothing deleted. To delete, the owner (never an agent) runs: {APPLY_COMMAND}")
        return 0

    result = apply_plan(
        plan.paths, args.runs_root, run_re, extras=extras, busy_check=busy_check, force_busy=args.force_busy
    )
    if result.busy:
        print("\nREFUSED: training or eval processes are running (use --force-busy to override):")
        print("\n".join(f"  {b}" for b in result.busy))
        return 1
    print(f"\ndeleted {len(result.deleted)} paths, {fmt_bytes(sum(n for _, n in result.deleted))}:")
    print("\n".join(f"  {p}  ({fmt_bytes(n)})" for p, n in result.deleted) or "  -")
    if result.refused:
        print("\nrefused (not deleted):")
        print("\n".join(f"  {p}: {why}" for p, why in result.refused))
    print("\n" + format_remaining(plan))
    return 1 if result.refused else 0


if __name__ == "__main__":
    sys.exit(main())
