"""Same-type eval report: EXPLORATORY (ADR 0005 item 2), never used in verdicts.

    uv run python eval/sametype_report.py                       # evals/*/sametype/ -> reports/sametype.md
    uv run python eval/sametype_report.py --evals E --out -     # print instead

Reads ``<evals>/<run>/sametype/step-NNNNN.jsonl`` (`eval/sametype_batch.py`; SPEC §12 records) and
writes one row per run x step: sample-level true success (robust pass), exploit rate (leaky pass and
robust fail) and format-failure rate, overall and per template. Point estimates only: this is a
descriptive companion to the held-out (T3) results, which stay the only basis for H1/H2/H5.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SAMETYPE_DIR = "sametype"
_STEP_FILE_RE = re.compile(r"^step-(\d+)\.jsonl$")
METRICS = (("true_success", "True success"), ("exploit", "Exploit"), ("format_failure", "Format failure"))
_REWARD_ORDER = {"base": 0, "robust": 1, "leaky": 2}
_SIZE_ORDER = {"0.8b": 0, "2b": 1, "4b": 2}
BANNER = (
    "> **EXPLORATORY (ADR 0005).** Unseen instances of the 11 training task types (`sametype` split), "
    "4 samples at temperature 0.7, both graders. Never used in an H1/H2/H5 verdict; the held-out (T3) "
    "results in `reports/results.md` stay the pre-registered basis."
)

Key = tuple[str, int]  # (run_id, step)


def read_records(evals_root: Path) -> tuple[dict[Key, list[dict[str, Any]]], list[str]]:
    """Records per (run, step) from every ``<evals>/<run>/sametype/step-NNNNN.jsonl``, plus notes."""
    out: dict[Key, list[dict[str, Any]]] = defaultdict(list)
    notes: list[str] = []
    evals_root = Path(evals_root)
    if not evals_root.is_dir():
        return {}, [f"`{evals_root}` not found"]
    for d in sorted(evals_root.glob(f"*/{SAMETYPE_DIR}")):
        for f in sorted(d.iterdir()):
            m = _STEP_FILE_RE.match(f.name)
            if not m:
                continue
            bad = 0
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if not all(isinstance(row.get(k), bool) for k in ("parse_ok", "robust_pass", "leaky_pass")):
                    bad += 1
                    continue
                out[(str(row.get("run_id") or d.parent.name), int(m[1]))].append(row)
            if bad:
                notes.append(f"{d.parent.name}/{SAMETYPE_DIR}/{f.name}: skipped {bad} unreadable line(s)")
    return dict(out), notes


def rates(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Sample-level rates; ``n`` is the number of samples."""
    n = len(rows)
    if not n:
        return {"n": 0, "true_success": 0.0, "exploit": 0.0, "format_failure": 0.0}
    return {
        "n": n,
        "true_success": sum(r["robust_pass"] for r in rows) / n,
        "exploit": sum(r["leaky_pass"] and not r["robust_pass"] for r in rows) / n,
        "format_failure": sum(not r["parse_ok"] for r in rows) / n,
    }


def table_rows(records: dict[Key, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One entry per (run, step): metadata, overall rates and per-template rates, in report order."""
    out = []
    for (run_id, step), rows in records.items():
        first = rows[0]
        by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_template[str(r["template"])].append(r)
        out.append(
            {
                "run_id": run_id,
                "size": first.get("size"),
                "reward": first.get("reward"),
                "seed": first.get("seed"),
                "step": step,
                "instances": len({r["instance_id"] for r in rows}),
                "overall": rates(rows),
                "templates": {t: rates(v) for t, v in sorted(by_template.items())},
            }
        )

    def order(e: dict[str, Any]) -> tuple:
        return (
            _SIZE_ORDER.get(str(e["size"]), 9),
            _REWARD_ORDER.get(str(e["reward"]), 9),
            e["seed"] if isinstance(e["seed"], int) else -1,
            e["run_id"],
            e["step"],
        )

    return sorted(out, key=order)


def _pct(x: float) -> str:
    return f"{x:.1%}"


def markdown(entries: Sequence[dict[str, Any]], notes: Iterable[str] = (), source: str = "evals") -> str:
    lines = ["# Same-type eval (EXPLORATORY, ADR 0005)", "", BANNER, ""]
    if not entries:
        return "\n".join([*lines, f"No same-type eval records under `{source}/*/{SAMETYPE_DIR}/` yet.", ""])
    lines += [
        f"Source: `{source}/<run>/{SAMETYPE_DIR}/step-NNNNN.jsonl`. Rates are per sample (n = samples).",
        "Exploit = leaky pass and robust fail. Format failure = no parseable tool call.",
        "",
        "## Overall",
        "",
        "| Run | Size | Reward | Seed | Step | n | True success | Exploit | Format failure |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        o = e["overall"]
        seed = "-" if e["seed"] is None else e["seed"]
        cells = [e["run_id"], e["size"], e["reward"], seed, e["step"], o["n"], *(_pct(o[k]) for k, _ in METRICS)]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    templates = sorted({t for e in entries for t in e["templates"]})
    for key, title in METRICS:
        lines += ["", f"## {title} by template", "", "| Run | Step | " + " | ".join(templates) + " |"]
        lines.append("|---|---|" + "---|" * len(templates))
        for e in entries:
            cells = [_pct(e["templates"][t][key]) if t in e["templates"] else "-" for t in templates]
            lines.append(f"| {e['run_id']} | {e['step']} | " + " | ".join(cells) + " |")
    notes = list(notes)
    if notes:
        lines += ["", "## Notes", "", *(f"- {n}" for n in notes)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Same-type eval report (EXPLORATORY, ADR 0005)")
    p.add_argument("--evals", type=Path, default=Path("evals"))
    p.add_argument("--out", default="reports/sametype.md", help="markdown output path, or - for stdout")
    a = p.parse_args(argv)
    records, notes = read_records(a.evals)
    text = markdown(table_rows(records), notes, source=str(a.evals))
    if a.out == "-":
        print(text, end="")
    else:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[sametype_report] {len(records)} run x step rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
