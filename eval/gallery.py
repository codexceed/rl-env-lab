"""Exploit gallery: pull exploit examples per family from held-out eval traces (results report).

Usage::

    python eval/gallery.py --run-id 2b-leak-s1                 # latest traced step, 3 per family
    python eval/gallery.py --run-id 2b-leak-s1 --step 300 --n 5 --evals evals

Reads ``<evals>/<run_id>/step-NNNNN.traces.jsonl`` (`eval/heldout.py::trace_row`, `train/LAYOUT.md`)
and prints a markdown block: for each family, up to ``n`` samples that passed the leaky grader and
failed the robust one, with the parsed call and the robust grader's reason. Selection is
deterministic: distinct instances first (sorted by instance id), then further samples of the same
instances, so reruns of the report pick the same examples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `eval` and `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.checkpoints import TRACES_SUFFIX, eval_output_path, eval_run_dir, eval_traces_path

DEFAULT_N = 3
_TRACE_RE = re.compile(r"^step-(\d+)" + re.escape(TRACES_SUFFIX) + "$")


def read_traces(path: str | Path) -> list[dict[str, Any]]:
    """Trace rows of one file; unparseable lines (e.g. a truncated last line) are skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def traced_steps(evals_root: str | Path, run_id: str) -> list[int]:
    """Steps of ``run_id`` that have a trace file, ascending."""
    d = eval_run_dir(Path(evals_root), run_id)
    if not d.is_dir():
        return []
    return sorted(int(m[1]) for p in d.iterdir() if (m := _TRACE_RE.match(p.name)))


def traces_file(evals_root: str | Path, run_id: str, step: int | None = None) -> Path:
    """The trace file of ``run_id`` at ``step`` (latest traced step when None). Raises if absent."""
    if step is None:
        steps = traced_steps(evals_root, run_id)
        if not steps:
            raise FileNotFoundError(f"no trace files for {run_id} under {evals_root}")
        step = steps[-1]
    path = eval_traces_path(eval_output_path(Path(evals_root), run_id, step))
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return path


def is_exploit(row: Mapping[str, Any]) -> bool:
    """Leaky pass and robust fail (recomputed from the passes, so an old row without the flag works)."""
    return bool(row.get("leaky_pass")) and not bool(row.get("robust_pass"))


def exploit_examples(rows: Iterable[Mapping[str, Any]], n_per_family: int = DEFAULT_N) -> dict[str, list[dict]]:
    """Up to ``n_per_family`` exploit rows per family, distinct instances first; families sorted."""
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if is_exploit(r):
            by_family[str(r.get("family"))].append(r)
    out: dict[str, list[dict]] = {}
    for fam in sorted(by_family):
        ordered = sorted(by_family[fam], key=lambda r: (str(r.get("instance_id")), int(r.get("sample_idx", 0))))
        first, rest, seen = [], [], set()
        for r in ordered:
            (rest if r.get("instance_id") in seen else first).append(dict(r))
            seen.add(r.get("instance_id"))
        out[fam] = (first + rest)[:n_per_family]
    return out


def exploit_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    """Per family: (exploit samples, all samples)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        c = counts[str(r.get("family"))]
        c[0] += is_exploit(r)
        c[1] += 1
    return {f: (c[0], c[1]) for f, c in sorted(counts.items())}


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _call_txt(row: Mapping[str, Any]) -> str:
    call = row.get("call")
    if not call:
        return "(no call)"
    args = ", ".join(f"{k}={v!r}" for k, v in sorted((call.get("arguments") or {}).items()))
    return f"{call.get('name')}({args})"


def gallery_markdown(
    examples: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    run_id: str,
    step: int | None,
    counts: Mapping[str, tuple[int, int]] | None = None,
    max_chars: int = 400,
) -> str:
    """A markdown block: per family, each example's call, robust reason and (clipped) completion."""
    lines = [f"### Exploit gallery: `{run_id}` step {step}", ""]
    families = sorted(set(examples) | set(counts or {}))
    if not families:
        return "\n".join([*lines, "No traces.", ""])
    for fam in families:
        ex = examples.get(fam, [])
        head = f"#### {fam}"
        if counts and fam in counts:
            k, n = counts[fam]
            head += f" ({k}/{n} samples exploit)"
        lines += [head, ""]
        if not ex:
            lines += ["No exploits.", ""]
            continue
        for r in ex:
            robust = r.get("robust") or {}
            reason = robust.get("reason", "?")
            detail = robust.get("detail") or ""
            lines.append(
                f"- `{r.get('instance_id')}` sample {r.get('sample_idx')}: `{_call_txt(r)}` "
                f"(robust: {reason}{f', {detail}' if detail else ''})"
            )
            lines += ["", "  ```", *("  " + ln for ln in _clip(str(r.get("completion", "")), max_chars).splitlines())]
            lines += ["  ```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exploit examples per family from held-out eval traces.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, help="eval step (default: the latest traced step)")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="examples per family")
    p.add_argument("--evals", type=Path, default=Path("evals"))
    p.add_argument("--max-chars", type=int, default=400, help="clip each completion to this many characters")
    a = p.parse_args(argv)
    try:
        path = traces_file(a.evals, a.run_id, a.step)
    except FileNotFoundError as exc:
        print(f"[gallery] {exc}", file=sys.stderr)
        return 2
    rows = read_traces(path)
    m = _TRACE_RE.match(path.name)
    step = a.step if a.step is not None or m is None else int(m[1])
    block = gallery_markdown(
        exploit_examples(rows, a.n), run_id=a.run_id, step=step, counts=exploit_counts(rows), max_chars=a.max_chars
    )
    sys.stdout.write(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
