"""v2 round table: EXPLORATORY (ADR 0005 item 3), never used in verdicts.

    uv run python eval/v2_report.py                    # evals/v2-*/ -> reports/v2.md   (`make v2-report`)
    uv run python eval/v2_report.py --out -            # print instead

The v2 runs (``v2-<size>-<rob|leak>-s<seed>``, ``train/grpo.py --variant v2-think``) are kept out
of `eval/compare.py` by their run ids. This table shows them on their own: for each eval (held-out T3
``evals/<run>/step-NNNNN.jsonl``, same-type ``evals/<run>/sametype/step-NNNNN.jsonl``) and each run,
the latest evaluated step, then robust vs leaky averaged over seeds. Sample-level true success and
exploit rate (leaky pass, robust fail); point estimates only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make `eval` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.sametype_report import rates

V2_RUN_RE = re.compile(r"^v2-(?P<size>[^-]+)-(?P<r>rob|leak)-s(?P<seed>\d+)$")
_STEP_RE = re.compile(r"^step-(\d+)\.jsonl$")
EVALS = (("heldout", "Held-out (T3)", ""), ("sametype", "Same-type", "sametype"))
BANNER = (
    "> **EXPLORATORY (ADR 0005).** v2 round: thinking on, answer cap 1,024 (ADR 0006). Never used in an H1/H2/H5 "
    "verdict; the pre-registered results are in `reports/results.md`."
)


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and all(
            isinstance(row.get(k), bool) for k in ("parse_ok", "robust_pass", "leaky_pass")
        ):
            rows.append(row)
    return rows


def latest(evals_root: Path, sub: str) -> dict[str, tuple[int, list[dict[str, Any]]]]:
    """run id -> (latest step, its records) for every v2 run with an eval of this kind."""
    out = {}
    for rdir in sorted(Path(evals_root).glob("v2-*")):
        if not V2_RUN_RE.match(rdir.name) or not (rdir / sub).is_dir():
            continue
        steps = {int(m[1]): f for f in (rdir / sub).iterdir() if (m := _STEP_RE.match(f.name))}
        if steps:
            step = max(steps)
            out[rdir.name] = (step, _read(steps[step]))
    return out


def summarize(evals_root: Path) -> dict[str, dict[str, Any]]:
    """Per eval kind: per-run rates at the latest step, and robust / leaky means over seeds."""
    result = {}
    for key, _, sub in EVALS:
        runs = {rid: {"step": step, **rates(rows)} for rid, (step, rows) in latest(evals_root, sub).items()}
        by_reward: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rid, r in runs.items():
            m = V2_RUN_RE.match(rid)
            assert m is not None
            by_reward[f"{m['size']} {'robust' if m['r'] == 'rob' else 'leaky'}"].append(r)
        groups = {
            g: {
                "seeds": len(rs),
                "true_success": sum(r["true_success"] for r in rs) / len(rs),
                "exploit": sum(r["exploit"] for r in rs) / len(rs),
            }
            for g, rs in sorted(by_reward.items())
        }
        result[key] = {"runs": runs, "groups": groups}
    return result


def markdown(summary: dict[str, dict[str, Any]], source: str = "evals") -> str:
    lines = ["# v2 round (EXPLORATORY, ADR 0005)", "", BANNER, ""]
    for key, title, sub in EVALS:
        s = summary[key]
        where = f"{source}/v2-*/{sub + '/' if sub else ''}step-NNNNN.jsonl"
        lines += [f"## {title}", "", f"Latest step of each run, from `{where}`.", ""]
        if not s["runs"]:
            lines += ["No v2 evals yet.", ""]
            continue
        lines += ["| Size, reward | Seeds | True success | Exploit |", "|---|---|---|---|"]
        for g, r in s["groups"].items():
            lines.append(f"| {g} | {r['seeds']} | {r['true_success']:.1%} | {r['exploit']:.1%} |")
        lines += ["", "| Run | Step | n | True success | Exploit | Format failure |", "|---|---|---|---|---|---|"]
        for rid, r in sorted(s["runs"].items()):
            cells = (r["true_success"], r["exploit"], r["format_failure"])
            lines.append(f"| {rid} | {r['step']} | {r['n']} | " + " | ".join(f"{x:.1%}" for x in cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="v2 round table (EXPLORATORY, ADR 0005)")
    p.add_argument("--evals", type=Path, default=Path("evals"))
    p.add_argument("--out", default="reports/v2.md", help="markdown output path, or - for stdout")
    a = p.parse_args(argv)
    text = markdown(summarize(a.evals), source=str(a.evals))
    if a.out == "-":
        print(text, end="")
    else:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"[v2_report] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
