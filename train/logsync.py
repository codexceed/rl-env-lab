"""Mirror small run logs to the repo (``logs/``) and to Lightning experiments (LitLogger).

Usage::

    python train/logsync.py --node n1 [--once] [--litlogger-interval 60] [--git-interval 1800]
        [--no-git] [--no-litlogger] [--teamspace NAME] [--max-file-mb 50] [--scope all|node]
    make logsync NODE=n1

A separate process, decoupled from training and eval. It only reads ``<runs>`` and ``<evals>``
(``paths`` in `configs/allocation.yaml`), so a sync failure can never affect a run. Two sinks, each
on its own cadence:

* **Repo logs** (every ``--git-interval``, default 30 min). Copies an explicit allowlist of small
  JSON/JSONL files into ``logs/``, mirroring the artifact layout (`train/LAYOUT.md`). Files over the
  size cap are skipped with a warning; weights and checkpoints never match the allowlist. Only
  changed files are copied (size + mtime). Then, unless ``--no-git``, it stages ``logs/`` only,
  commits ``chore(logs): sync <node> run logs (<n> runs)`` and pushes with ``git pull --rebase`` and
  retries, because both nodes push to ``main``. It refuses to commit if anything outside ``logs/``
  is staged, and refuses to push local commits that touch paths outside ``logs/``.
* **Lightning experiments** (every ``--litlogger-interval``, default 60 s). One experiment per run
  id: new ``train_metrics.jsonl`` rows go to ``train/<key>``, and each new eval file gives
  ``eval/true_success``, ``eval/leaky_pass_rate``, ``eval/exploit_rate`` and ``eval/parse_rate`` at
  its step, plus the eval files as File artifacts. Metadata (size, reward, seed, node, status) lets
  the dashboard filter and group; ``status`` is updated when it changes. A cursor per experiment
  (``<runs>/<run_id>/.logsync.json``) makes restarts resume without double-logging. Check runs are
  mirrored too: ``toy-<name>-<node>``, ``compat-<model>-<node>``, ``bench-<node>``.

When a run's ``status.json`` turns terminal, and on SIGTERM / SIGINT, both sinks flush at once.
``--once`` runs both sinks once and exits.

``logs/`` layout::

    logs/runs/<run_id>/{status.json, train_metrics.jsonl, run_manifest.json, eval_steps.json}
    logs/runs/_coordinator/<node>.jsonl
    logs/evals/<run_id>/{step-NNNNN.jsonl, step-NNNNN.traces.jsonl, *.summary.json}
    logs/checks/<node>/{compat/*.json, toy/<name>/{summary.json, metrics.jsonl}, bench/summary.json}

The local JSONL files stay the source of truth; verdicts are computed only from them
(`eval/compare.py`). Both mirrors are for browsing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.checkpoints import (
    TERMINAL_STATUSES,
    atomic_write_json,
    atomic_write_text,
    list_eval_checkpoints,
    read_json,
    step_name,
)
from train.coordinator import load_config, resolve_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIRNAME = "logs"
DEFAULT_LITLOGGER_INTERVAL_S = 60.0
DEFAULT_GIT_INTERVAL_S = 1800.0
DEFAULT_MAX_FILE_MB = 50.0
STATUS_POLL_S = 30.0
GIT_TIMEOUT_S = 300
PUSH_TRIES = 3

# Explicit allowlist, as globs relative to the run / eval / runs root. Anything else is never copied.
RUN_FILES = ("status.json", "train_metrics.jsonl", "run_manifest.json")
EVAL_FILES = ("step-*.jsonl", "*.summary.json")  # records, traces (step-N.traces.jsonl), diagnostic
CHECK_FILES = ("compat/*.json", "toy/*/summary.json", "toy/*/metrics.jsonl", "bench/summary.json")
COORDINATOR_FILES = ("_coordinator/*.jsonl",)
ALLOWED_SUFFIXES = (".json", ".jsonl")  # belt and braces: weights and checkpoints never qualify
CHECK_DIRS = frozenset({"compat", "toy", "bench", "gpu_checks"})  # under <runs>, not run ids
NO_EXPERIMENT_DIRS = frozenset({"diagnostic"})  # <evals>/diagnostic is a baseline table, not a run
EVAL_STEPS_FILE = "eval_steps.json"
CURSOR_FILE = ".logsync.json"
STATE_DIRNAME = "_logsync"  # <runs>/_logsync/: cursors for experiments without a run dir, litlogger root
NODE_ALIASES = {"n1": "node1", "n2": "node2"}

# status.json status -> dashboard status
DASHBOARD_STATUS = {"running": "running", "finished": "done", "failed": "failed", "timeout": "timeout"}
STATUS_CODES = {"pending": 0.0, "running": 1.0, "done": 2.0, "failed": 3.0, "timeout": 4.0}

_RECORDS_RE = re.compile(r"^step-(\d+)\.jsonl$")
_TRACES_RE = re.compile(r"^step-(\d+)\.traces\.jsonl$")
_RUN_ID_RE = re.compile(r"^(\d+(?:\.\d+)?b)-(rob|leak)-s(\d+)")
_BASE_ID_RE = re.compile(r"^base-(\d+(?:\.\d+)?b)$")


def info(msg: str) -> None:
    print(f"[logsync] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[logsync] WARNING: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- file selection (sink A)


@dataclass(frozen=True)
class CopyItem:
    src: Path
    dst: Path


def allowed_file(path: Path) -> bool:
    """Only visible ``.json`` / ``.jsonl`` files are ever mirrored."""
    return path.suffix in ALLOWED_SUFFIXES and not path.name.startswith(".")


def match_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    """Regular, allowed files under ``root`` matching any glob in ``patterns``; sorted, no duplicates."""
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and not path.is_symlink() and allowed_file(path):
                found.add(path)
    return sorted(found)


RUN_MARKERS = ("status.json", "train_metrics.jsonl", "run_manifest.json")  # a training run dir has one


def is_run_dir(path: Path) -> bool:
    """A training run dir: not hidden or internal, and holding at least one run marker file.

    Other folders under runs/ (model exports, setup logs, archived checks) are never runs.
    """
    if not path.is_dir() or path.name.startswith(("_", ".")) or path.name in CHECK_DIRS:
        return False
    return any((path / m).is_file() for m in RUN_MARKERS)


def list_run_ids(runs_root: Path, evals_root: Path, exclude: Iterable[str] = ()) -> list[str]:
    """Run ids with a run dir or an eval dir (base and diagnostic evals included), minus ``exclude``."""
    ids: set[str] = set()
    if runs_root.is_dir():
        ids.update(child.name for child in runs_root.iterdir() if is_run_dir(child))
    if evals_root.is_dir():
        ids.update(
            child.name for child in evals_root.iterdir() if child.is_dir() and not child.name.startswith(("_", "."))
        )
    return sorted(ids - set(exclude))


def plan_copies(
    runs_root: Path, evals_root: Path, logs_root: Path, node: str, exclude: Iterable[str] = ()
) -> list[CopyItem]:
    """Every allowlisted source file and its destination under ``logs_root``."""
    items: list[CopyItem] = []
    for rid in list_run_ids(runs_root, evals_root, exclude):
        items += [CopyItem(p, logs_root / "runs" / rid / p.name) for p in match_files(runs_root / rid, RUN_FILES)]
        items += [CopyItem(p, logs_root / "evals" / rid / p.name) for p in match_files(evals_root / rid, EVAL_FILES)]
    for p in match_files(runs_root, CHECK_FILES):
        items.append(CopyItem(p, logs_root / "checks" / node / p.relative_to(runs_root)))
    for p in match_files(runs_root, COORDINATOR_FILES):
        items.append(CopyItem(p, logs_root / "runs" / p.relative_to(runs_root)))
    return items


def apply_size_cap(items: Iterable[CopyItem], max_bytes: int) -> tuple[list[CopyItem], list[tuple[CopyItem, int]]]:
    """Split ``items`` into (within the cap, [(over the cap, size)]). Vanished files are dropped."""
    keep: list[CopyItem] = []
    over: list[tuple[CopyItem, int]] = []
    for item in items:
        try:
            size = item.src.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            over.append((item, size))
        else:
            keep.append(item)
    return keep, over


def needs_copy(src: Path, dst: Path) -> bool:
    """True when ``dst`` is missing or differs from ``src`` in size or mtime (copies keep the mtime)."""
    try:
        s = src.stat()
    except OSError:
        return False
    try:
        d = dst.stat()
    except OSError:
        return True
    return s.st_size != d.st_size or s.st_mtime_ns != d.st_mtime_ns


def copy_file(src: Path, dst: Path) -> None:
    """Copy with metadata through a hidden temp file in the destination dir, then rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def eval_step_state(edir: Path, step: int) -> str:
    """Watcher state of one eval step: done, failed, running, records (no marker) or pending."""
    name = step_name(step)
    for suffix, state in ((".done", "done"), (".failed", "failed"), (".claim", "running"), (".jsonl", "records")):
        if (edir / f"{name}{suffix}").exists():
            return state
    return "pending"


def eval_steps_summary(rdir: Path, edir: Path) -> dict[str, Any]:
    """Complete eval checkpoints of a run (from ``COMPLETE`` markers) and their eval state."""
    steps = [
        {"step": c.step, "final": c.final, "eval": eval_step_state(edir, c.step)} for c in list_eval_checkpoints(rdir)
    ]
    return {"run_id": rdir.name, "steps": steps}


@dataclass
class RepoSyncResult:
    copied: list[Path] = field(default_factory=list)
    unchanged: int = 0
    over_cap: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)


def sync_repo_logs(
    runs_root: Path,
    evals_root: Path,
    logs_root: Path,
    node: str,
    *,
    max_bytes: int,
    exclude: Iterable[str] = (),
) -> RepoSyncResult:
    """Copy changed allowlisted files into ``logs_root`` and refresh each run's ``eval_steps.json``.

    Args:
        runs_root: Source ``<runs>`` directory.
        evals_root: Source ``<evals>`` directory.
        logs_root: Destination, normally ``<repo>/logs``.
        node: Node label, used for ``logs/checks/<node>/``.
        max_bytes: Per-file size cap; larger files are skipped with a warning.
        exclude: Run ids to leave out (another node's runs).

    Returns:
        What was copied, left alone, skipped or failed.
    """
    exclude = frozenset(exclude)
    result = RepoSyncResult()
    keep, over = apply_size_cap(plan_copies(runs_root, evals_root, logs_root, node, exclude), max_bytes)
    for item, size in over:
        warn(f"skipped {item.src}: {size / 1e6:.1f} MB is over the {max_bytes / 1e6:.0f} MB cap")
        result.over_cap.append(item.src)
    for item in keep:
        if not needs_copy(item.src, item.dst):
            result.unchanged += 1
            continue
        try:
            copy_file(item.src, item.dst)
            result.copied.append(item.dst)
        except OSError as exc:
            warn(f"copy {item.src} failed: {exc}")
            result.failed.append(item.src)
    for rid in list_run_ids(runs_root, evals_root, exclude):
        rdir = runs_root / rid
        if not rdir.is_dir():
            continue
        dst = logs_root / "runs" / rid / EVAL_STEPS_FILE
        text = json.dumps(eval_steps_summary(rdir, evals_root / rid), indent=2, sort_keys=True) + "\n"
        try:
            if not dst.is_file() or dst.read_text() != text:
                atomic_write_text(dst, text)
                result.copied.append(dst)
            else:
                result.unchanged += 1
        except OSError as exc:
            warn(f"write {dst} failed: {exc}")
            result.failed.append(dst)
    return result


# ---------------------------------------------------------------- git (sink A)


def _git(repo: Path, *args: str, timeout: float = GIT_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=timeout, check=False)


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _err(proc: subprocess.CompletedProcess[str]) -> str:
    """Last non-empty output line of a failed git command."""
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"exit {proc.returncode}"


def staged_paths(repo: Path) -> list[str]:
    proc = _git(repo, "diff", "--cached", "--name-only", "-z")
    if proc.returncode != 0:
        raise RuntimeError(f"git diff --cached failed: {_err(proc)}")
    return [p for p in proc.stdout.split("\0") if p]


def count_runs(paths: Iterable[str], logs_rel: str = LOGS_DIRNAME) -> int:
    """Distinct run ids among ``logs/runs/<id>/...`` and ``logs/evals/<id>/...`` paths."""
    ids = set()
    for p in paths:
        parts = Path(p).parts
        if len(parts) >= 4 and parts[0] == logs_rel and parts[1] in ("runs", "evals") and parts[2][:1] != "_":
            ids.add(parts[2])
    return len(ids)


def commit_message(node: str, n_runs: int) -> str:
    return f"chore(logs): sync {node} run logs ({n_runs} runs)"


def git_commit_logs(repo: Path, node: str, logs_rel: str = LOGS_DIRNAME) -> str:
    """Stage ``logs_rel`` only and commit it.

    Refuses when anything outside ``logs_rel`` is already staged, so unrelated work is never committed.

    Args:
        repo: Repository root.
        node: Node label for the commit message.
        logs_rel: The logs folder, relative to ``repo``.

    Returns:
        ``refused``, ``nothing`` or ``committed``.

    Raises:
        RuntimeError: A git command failed (the caller warns and retries next cycle).
    """
    others = [p for p in staged_paths(repo) if not _under(p, logs_rel)]
    if others:
        warn(f"git step skipped: paths outside {logs_rel}/ are staged ({', '.join(others[:5])}); commit them first")
        return "refused"
    if not (repo / logs_rel).is_dir():
        return "nothing"
    add = _git(repo, "add", "--all", "--", logs_rel)
    if add.returncode != 0:
        raise RuntimeError(f"git add failed: {_err(add)}")
    staged = staged_paths(repo)
    if not staged:
        return "nothing"
    # --no-verify: data files only, and the lint hooks need a toolchain the node may not have
    commit = _git(
        repo, "commit", "--no-verify", "-m", commit_message(node, count_runs(staged, logs_rel)), "--", logs_rel
    )
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed: {_err(commit)}")
    info(f"committed {len(staged)} log files")
    return "committed"


def git_push_logs(repo: Path, logs_rel: str = LOGS_DIRNAME, *, tries: int = PUSH_TRIES, backoff_s: float = 5.0) -> str:
    """Push local commits with ``git pull --rebase`` + retry, but only if they touch ``logs_rel`` alone.

    Args:
        repo: Repository root.
        logs_rel: The logs folder, relative to ``repo``.
        tries: Pull + push attempts.
        backoff_s: Sleep between attempts, times the attempt number.

    Returns:
        ``no-upstream``, ``up-to-date``, ``refused``, ``pushed`` or ``push-failed``.
    """
    if _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}").returncode != 0:
        warn("no upstream branch: log commits stay local")
        return "no-upstream"
    ahead = _git(repo, "rev-list", "--count", "@{u}..HEAD").stdout.strip()
    if ahead in ("", "0"):
        return "up-to-date"
    touched = [p for p in _git(repo, "log", "--name-only", "--format=", "@{u}..HEAD").stdout.splitlines() if p]
    outside = sorted({p for p in touched if not _under(p, logs_rel)})
    if outside:
        warn(f"not pushing: local commits also change {', '.join(outside[:5])}; push them yourself")
        return "refused"
    for attempt in range(1, tries + 1):
        pull = _git(repo, "pull", "--rebase")
        if pull.returncode != 0:
            _git(repo, "rebase", "--abort")
            warn(f"git pull --rebase failed (try {attempt}/{tries}): {_err(pull)}")
        else:
            push = _git(repo, "push")
            if push.returncode == 0:
                info(f"pushed {ahead} log commit(s)")
                return "pushed"
            warn(f"git push failed (try {attempt}/{tries}): {_err(push)}")
        if attempt < tries:
            time.sleep(backoff_s * attempt)
    warn("push failed; the commits stay local and are retried next cycle")
    return "push-failed"


# ---------------------------------------------------------------- metrics (sink B)


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """JSON objects in a JSONL file. Blank, invalid and partial (still being written) lines are skipped."""
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def new_metric_rows(rows: Iterable[Mapping[str, Any]], after_step: int) -> list[tuple[int, dict[str, float]]]:
    """Numeric fields of rows with ``step > after_step``, by step; the last row for a step wins."""
    by_step: dict[int, dict[str, float]] = {}
    for row in rows:
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= after_step:
            continue
        values = {k: f for k, v in row.items() if k != "step" and (f := _number(v)) is not None}
        by_step[step] = values
    return [(s, by_step[s]) for s in sorted(by_step) if by_step[s]]


def eval_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Sample-level rates of one eval file (SPEC §12): exploit = leaky pass and robust fail."""
    n = len(records)
    if not n:
        return {}
    robust = sum(bool(r.get("robust_pass")) for r in records)
    leaky = sum(bool(r.get("leaky_pass")) for r in records)
    exploit = sum(bool(r.get("leaky_pass")) and not bool(r.get("robust_pass")) for r in records)
    parsed = sum(bool(r.get("parse_ok")) for r in records)
    return {
        "true_success": robust / n,
        "leaky_pass_rate": leaky / n,
        "exploit_rate": exploit / n,
        "parse_rate": parsed / n,
        "n_samples": float(n),
    }


def file_stamp(path: Path) -> str:
    """Change detector for a file: size and mtime."""
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def dashboard_status(status: Mapping[str, Any] | None, *, has_run_dir: bool) -> str:
    """running / done / failed / timeout from ``status.json``; ``pending`` before launch; eval-only = done."""
    if not has_run_dir:
        return "done"
    if not status:
        return "pending"
    return DASHBOARD_STATUS.get(str(status.get("status")), "pending")


def parse_run_id(run_id: str) -> dict[str, str]:
    """``2b-leak-s2`` -> size, reward, seed; ``base-2b`` -> size, reward=base."""
    m = _RUN_ID_RE.match(run_id)
    if m:
        return {"size": m.group(1), "reward": "robust" if m.group(2) == "rob" else "leaky", "seed": m.group(3)}
    m = _BASE_ID_RE.match(run_id)
    if m:
        return {"size": m.group(1), "reward": "base"}
    return {}


def _flatten(obj: Mapping[str, Any], prefix: str = "", depth: int = 2) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and depth > 1:
            out.update(_flatten(value, f"{name}.", depth - 1))
        elif isinstance(value, str | int | float | bool):
            out[name] = str(value)[:200]
    return out


def run_metadata(run_id: str, rdir: Path, node: str) -> dict[str, str]:
    """Dashboard tags: size, reward, seed, node, status, plus scalar fields of ``run_manifest.json``."""
    status = read_json(rdir / "status.json", default=None)
    status = status if isinstance(status, dict) else {}
    manifest = read_json(rdir / "run_manifest.json", default=None)
    meta: dict[str, str] = {"run_id": run_id, "node": node}
    if isinstance(manifest, dict):
        meta.update(dict(list(_flatten(manifest).items())[:50]))
    meta.update(parse_run_id(run_id))
    for key in ("size", "reward", "seed", "node", "group"):
        if status.get(key) is not None:
            meta[key] = str(status[key])
    meta["status"] = dashboard_status(status, has_run_dir=rdir.is_dir())
    return meta


@dataclass
class Cursor:
    """What an experiment already has: last train step, eval steps and files (by stamp), status."""

    train_step: int = -1
    eval_steps: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    status: str | None = None

    @classmethod
    def load(cls, path: Path) -> Cursor:
        data = read_json(path, default=None)
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(
                train_step=int(data.get("train_step", -1)),
                eval_steps={str(k): str(v) for k, v in dict(data.get("eval_steps") or {}).items()},
                files={str(k): str(v) for k, v in dict(data.get("files") or {}).items()},
                status=data.get("status"),
            )
        except (TypeError, ValueError):
            warn(f"unreadable cursor {path}; starting over")
            return cls()

    def save(self, path: Path) -> None:
        atomic_write_json(
            path,
            {"train_step": self.train_step, "eval_steps": self.eval_steps, "files": self.files, "status": self.status},
        )


@dataclass
class MirrorTarget:
    """One LitLogger experiment and the local files it mirrors."""

    name: str
    cursor_path: Path
    metadata: dict[str, str] = field(default_factory=dict)
    metrics_file: Path | None = None
    metrics_prefix: str = "train/"
    eval_dir: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)
    ended: bool = True  # finalize the experiment after each mirror (False keeps it open while live)


@dataclass
class Pending:
    rows: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    evals: list[tuple[int, Path, str]] = field(default_factory=list)
    files: list[tuple[str, Path, str]] = field(default_factory=list)
    status: str | None = None

    @property
    def empty(self) -> bool:
        return not (self.rows or self.evals or self.files or self.status)


def eval_files(edir: Path | None) -> tuple[dict[int, Path], dict[int, Path]]:
    """(records by step, traces by step) in an eval dir."""
    records: dict[int, Path] = {}
    traces: dict[int, Path] = {}
    if edir is None or not edir.is_dir():
        return records, traces
    for path in edir.iterdir():
        if not path.is_file():
            continue
        if m := _RECORDS_RE.match(path.name):
            records[int(m.group(1))] = path
        elif m := _TRACES_RE.match(path.name):
            traces[int(m.group(1))] = path
    return records, traces


def pending_for(target: MirrorTarget, cursor: Cursor, max_bytes: int) -> Pending:
    """Everything in ``target`` the cursor says the experiment does not have yet."""
    pending = Pending()
    if target.metrics_file is not None:
        pending.rows = new_metric_rows(read_jsonl(target.metrics_file), cursor.train_step)
    files = dict(target.files)
    records, traces = eval_files(target.eval_dir)
    for step, path in sorted(records.items()):
        files[f"eval_files/{step_name(step)}"] = path
        if str(step) not in cursor.eval_steps:
            pending.evals.append((step, path, file_stamp(path)))
    for step, path in sorted(traces.items()):
        files[f"eval_files/{step_name(step)}-traces"] = path
    for key, path in sorted(files.items()):
        if not path.is_file():
            continue
        if path.stat().st_size > max_bytes:
            continue  # already warned by the repo sink; never uploaded
        stamp = file_stamp(path)
        if cursor.files.get(key) != stamp:
            pending.files.append((key, path, stamp))
    status = target.metadata.get("status")
    if status and status != cursor.status:
        pending.status = status
    return pending


class LitSink:
    """LitLogger mirror. ``litlogger`` is imported lazily; when it is missing this sink does nothing.

    Experiments of live runs stay open between cycles and are finalized when the run ends or the
    process stops. A restart re-opens each experiment by name (``litlogger.init(name=...)``) and the
    cursor skips what it already has.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        max_bytes: int,
        teamspace: str | None = None,
        module: Any = None,
        after_init: Callable[[], None] | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.max_bytes = max_bytes
        self.teamspace = teamspace
        self._module = module
        # litlogger's Experiment installs its own SIGTERM/SIGINT handlers (finalize + sys.exit);
        # the CLI passes a hook that puts ours back so a signal still flushes both sinks
        self._after_init = after_init
        self._open: dict[str, Any] = {}

    def available(self) -> bool:
        """Import ``litlogger`` if needed; False (with a warning) when it is missing or broken."""
        if self._module is None:
            try:
                self._module = importlib.import_module("litlogger")
            except Exception as exc:
                warn(f"litlogger unavailable ({exc!r}); skipping Lightning experiments this cycle")
                return False
        return True

    def _experiment(self, target: MirrorTarget) -> Any:
        exp = self._open.get(target.name)
        if exp is None:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {"name": target.name, "metadata": target.metadata, "root_dir": str(self.root_dir)}
            if self.teamspace:
                kwargs["teamspace"] = self.teamspace
            try:
                exp = self._module.init(**kwargs)
            finally:
                if self._after_init is not None:
                    self._after_init()
            self._open[target.name] = exp
        return exp

    def close(self, name: str) -> None:
        exp = self._open.pop(name, None)
        if exp is not None:
            try:
                exp.finalize()
            except Exception as exc:
                warn(f"finalize {name} failed: {exc!r}")

    def close_all(self) -> None:
        for name in list(self._open):
            self.close(name)

    def mirror(self, target: MirrorTarget) -> bool:
        """Append whatever ``target`` has that its experiment does not. Returns True if anything was sent."""
        cursor = Cursor.load(target.cursor_path)
        pending = pending_for(target, cursor, self.max_bytes)
        if pending.empty:
            if target.ended:
                self.close(target.name)
            return False
        ok = False
        try:
            exp = self._experiment(target)
            for step, values in pending.rows:
                for key, value in values.items():
                    exp[f"{target.metrics_prefix}{key}"].append(value, step=step)
                cursor.train_step = step
            for step, path, stamp in pending.evals:
                for key, value in eval_summary(read_jsonl(path)).items():
                    exp[f"eval/{key}"].append(value, step=step)
                cursor.eval_steps[str(step)] = stamp
            for key, path, stamp in pending.files:
                exp[key] = self._module.File(str(path))
                cursor.files[key] = stamp
            if pending.status:
                self._set_status(exp, pending.status, max(cursor.train_step, 0))
                cursor.status = pending.status
            ok = True
        finally:
            cursor.save(target.cursor_path)
            if not ok or target.ended:
                self.close(target.name)  # on error: drop the handle, re-open next cycle
        return True

    @staticmethod
    def _set_status(exp: Any, status: str, step: int) -> None:
        """Metadata ``status`` (documented: ``experiment["key"] = "value"``); a metric if that fails."""
        try:
            exp["status"] = status
        except Exception as exc:
            warn(f"metadata update failed ({exc!r}); logging status_code instead")
            exp["status_code"].append(STATUS_CODES.get(status, 0.0), step=step)


def experiment_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-") or "unnamed"


def build_targets(runs_root: Path, evals_root: Path, node: str, exclude: Iterable[str] = ()) -> list[MirrorTarget]:
    """One experiment per run id (and base eval), plus the check runs (toy, compat, bench)."""
    state = runs_root / STATE_DIRNAME
    targets: list[MirrorTarget] = []
    for rid in list_run_ids(runs_root, evals_root, exclude):
        if rid in NO_EXPERIMENT_DIRS:
            continue
        rdir = runs_root / rid
        edir = evals_root / rid
        name = experiment_name(rid)
        meta = run_metadata(rid, rdir, node)
        files = {"files/run_manifest": rdir / "run_manifest.json"} if (rdir / "run_manifest.json").is_file() else {}
        targets.append(
            MirrorTarget(
                name=name,
                cursor_path=rdir / CURSOR_FILE if rdir.is_dir() else state / f"{name}.json",
                metadata=meta,
                metrics_file=rdir / "train_metrics.jsonl",
                metrics_prefix="train/",
                eval_dir=edir if edir.is_dir() else None,
                files=files,
                ended=meta["status"] in ("done", "failed", "timeout"),
            )
        )
    toy_root = runs_root / "toy"
    for d in sorted(p for p in toy_root.iterdir() if p.is_dir()) if toy_root.is_dir() else []:
        name = experiment_name(f"toy-{d.name}-{node}")
        summary = d / "summary.json"
        targets.append(
            MirrorTarget(
                name=name,
                cursor_path=state / f"{name}.json",
                metadata={"check": "toy", "node": node, "source": d.name},
                metrics_file=d / "metrics.jsonl",
                metrics_prefix="toy/",
                files={"files/summary": summary},
                ended=summary.is_file(),
            )
        )
    for p in match_files(runs_root, ("compat/*.json",)):
        name = experiment_name(f"compat-{p.stem}-{node}")
        meta = {"check": "compat", "node": node, "source": p.stem}
        targets.append(
            MirrorTarget(name=name, cursor_path=state / f"{name}.json", metadata=meta, files={"files/result": p})
        )
    bench = runs_root / "bench" / "summary.json"
    if bench.is_file():
        name = experiment_name(f"bench-{node}")
        meta = {"check": "bench", "node": node}
        targets.append(
            MirrorTarget(name=name, cursor_path=state / f"{name}.json", metadata=meta, files={"files/summary": bench})
        )
    return targets


# ---------------------------------------------------------------- cycles and schedule


@dataclass
class Options:
    node: str
    repo: Path
    logs_root: Path
    runs_root: Path
    evals_root: Path
    max_bytes: int
    git: bool = True
    exclude: frozenset[str] = frozenset()


def repo_cycle(opts: Options) -> str | None:
    """Sink A: copy into ``logs/``, then commit and push. Never raises. Returns the git outcome."""
    try:
        res = sync_repo_logs(
            opts.runs_root, opts.evals_root, opts.logs_root, opts.node, max_bytes=opts.max_bytes, exclude=opts.exclude
        )
        info(f"repo logs: {len(res.copied)} updated, {res.unchanged} unchanged, {len(res.over_cap)} over the cap")
    except Exception as exc:
        warn(f"copying repo logs failed: {exc!r}")
    if not opts.git:
        return None
    try:
        logs_rel = opts.logs_root.resolve().relative_to(opts.repo.resolve()).as_posix()
    except ValueError:
        warn(f"{opts.logs_root} is outside {opts.repo}; git step skipped")
        return None
    try:
        outcome = git_commit_logs(opts.repo, opts.node, logs_rel)
        if outcome == "refused":
            return outcome
        return git_push_logs(opts.repo, logs_rel)
    except Exception as exc:
        warn(f"git step failed: {exc!r}; retrying next cycle")
        return "error"


def litlogger_cycle(opts: Options, sink: LitSink) -> int:
    """Sink B: mirror every target. Never raises. Returns how many experiments got new data."""
    if not sink.available():
        return 0
    try:
        targets = build_targets(opts.runs_root, opts.evals_root, opts.node, opts.exclude)
    except Exception as exc:
        warn(f"listing experiments failed: {exc!r}")
        return 0
    sent = 0
    for target in targets:
        try:
            sent += sink.mirror(target)
        except Exception as exc:
            warn(f"litlogger {target.name} failed: {exc!r}; retrying next cycle")
    return sent


def run_statuses(runs_root: Path, exclude: Iterable[str] = ()) -> dict[str, str | None]:
    """``status`` field of every run's ``status.json`` (None when missing)."""
    out: dict[str, str | None] = {}
    if not runs_root.is_dir():
        return out
    skip = set(exclude)
    for d in runs_root.iterdir():
        if is_run_dir(d) and d.name not in skip:
            status = read_json(d / "status.json", default=None)
            out[d.name] = str(status.get("status")) if isinstance(status, dict) else None
    return out


def newly_ended(before: Mapping[str, str | None], now: Mapping[str, str | None]) -> list[str]:
    """Runs that are terminal now but were not in ``before`` (missing counts as not terminal)."""
    return sorted(rid for rid, s in now.items() if s in TERMINAL_STATUSES and before.get(rid) not in TERMINAL_STATUSES)


def serve(
    opts: Options,
    sink: LitSink | None,
    *,
    once: bool = False,
    litlogger_interval: float = DEFAULT_LITLOGGER_INTERVAL_S,
    git_interval: float = DEFAULT_GIT_INTERVAL_S,
    stop: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    poll_s: float = STATUS_POLL_S,
) -> None:
    """Run both sinks on their cadences until ``stop`` is set (or once).

    A run turning terminal, and ``stop`` itself, trigger an immediate flush of both sinks.

    Args:
        opts: Paths and git settings.
        sink: The LitLogger sink, or None when disabled.
        once: Run both sinks once and return.
        litlogger_interval: Seconds between LitLogger cycles.
        git_interval: Seconds between repo-log cycles (copy, commit, push).
        stop: Set it to flush and return (the CLI sets it on SIGTERM / SIGINT).
        clock: Monotonic clock, for tests.
        poll_s: Longest sleep between ``status.json`` checks.
    """
    stop = stop or threading.Event()
    seen = run_statuses(opts.runs_root, opts.exclude)
    next_lit = next_git = clock()
    try:
        while True:
            now = clock()
            current = run_statuses(opts.runs_root, opts.exclude)
            ended = newly_ended(seen, current)
            seen = current
            flush = bool(ended) or stop.is_set()
            if ended:
                info(f"run(s) ended: {', '.join(ended)}; flushing both sinks")
            if sink is not None and (flush or once or now >= next_lit):
                litlogger_cycle(opts, sink)
                next_lit = clock() + litlogger_interval
            if flush or once or now >= next_git:
                repo_cycle(opts)
                next_git = clock() + git_interval
            if once or stop.is_set():
                return
            wake = min(next_git, next_lit if sink is not None else math.inf, clock() + poll_s)
            stop.wait(max(0.0, wake - clock()))
    finally:
        if sink is not None:
            sink.close_all()


# ---------------------------------------------------------------- CLI


def resolve_node(node: str, cfg: Mapping[str, Any]) -> str | None:
    """The allocation.yaml node key for ``node`` (``n1`` -> ``node1``), or None."""
    nodes = cfg.get("nodes") or {}
    for key in (node, NODE_ALIASES.get(node)):
        if key in nodes:
            return key
    return None


def other_nodes_runs(cfg: Mapping[str, Any], node: str) -> frozenset[str]:
    """Run ids that allocation.yaml assigns to other nodes (``--scope node`` leaves them out)."""
    key = resolve_node(node, cfg)
    if key is None:
        warn(f"node {node!r} is not in the allocation config; syncing every run")
        return frozenset()
    own = {rid for g in cfg["nodes"][key].get("groups") or [] for rid in g.get("queue") or []}
    return frozenset(set(cfg.get("runs") or {}) - own)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mirror small run logs to logs/ (git) and Lightning experiments.")
    ap.add_argument("--node", required=True, help="node label, e.g. n1 (commit message, logs/checks/<node>/)")
    ap.add_argument("--config", type=Path, default=Path("configs/allocation.yaml"))
    ap.add_argument("--once", action="store_true", help="run both sinks once and exit")
    ap.add_argument("--litlogger-interval", type=float, default=DEFAULT_LITLOGGER_INTERVAL_S)
    ap.add_argument("--git-interval", "--interval", dest="git_interval", type=float, default=DEFAULT_GIT_INTERVAL_S)
    ap.add_argument("--no-git", action="store_true", help="copy into logs/ but do not commit or push")
    ap.add_argument("--no-litlogger", action="store_true", help="skip Lightning experiments")
    ap.add_argument("--teamspace", default=None, help="LitLogger teamspace (default: the studio's)")
    ap.add_argument("--max-file-mb", type=float, default=DEFAULT_MAX_FILE_MB)
    ap.add_argument("--scope", choices=("all", "node"), default="all", help="node: skip runs of other nodes")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--logs", type=Path, default=None, help="default: <repo>/logs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config.is_file() else {}
    paths = resolve_paths(cfg or {})["paths"]
    runs_root, evals_root = Path(paths["runs"]), Path(paths["evals"])
    max_bytes = int(args.max_file_mb * 1e6)
    opts = Options(
        node=args.node,
        repo=args.repo,
        logs_root=args.logs or args.repo / LOGS_DIRNAME,
        runs_root=runs_root,
        evals_root=evals_root,
        max_bytes=max_bytes,
        git=not args.no_git,
        exclude=other_nodes_runs(cfg, args.node) if args.scope == "node" else frozenset(),
    )
    stop = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        info(f"signal {signum}: final flush")
        stop.set()

    def _install_signals() -> None:
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

    _install_signals()
    sink = None
    if not args.no_litlogger:
        sink = LitSink(
            root_dir=runs_root / STATE_DIRNAME / "litlogger",
            max_bytes=max_bytes,
            teamspace=args.teamspace,
            after_init=_install_signals,
        )
    info(
        f"node {args.node}: runs {runs_root}, evals {evals_root} -> {opts.logs_root}"
        f" (git {'on' if opts.git else 'off'}, litlogger {'on' if sink else 'off'})"
    )
    serve(
        opts,
        sink,
        once=args.once,
        litlogger_interval=args.litlogger_interval,
        git_interval=args.git_interval,
        stop=stop,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
