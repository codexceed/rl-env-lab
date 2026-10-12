"""Gate acceptance over a built dataset (SPEC §7, §9).

    python -m envlab.verify.acceptance --data data

Prints JSON: per split pass counts per gate, broken rejected count, valid pass rate.
Exits 1 if acceptance fails: any broken instance passes all gates, valid pass rate < 99%,
or a split file is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from envlab.tasks.instance import TaskInstance, read_jsonl
from envlab.verify.gates import run_gates

# Frozen §9 splits only. ADR 0005 exploratory splits (`sametype`) are not part of §7 acceptance.
VALID_SPLITS: tuple[str, ...] = ("train", "train_hard", "heldout")
BROKEN_SPLIT = "broken"
MIN_VALID_PASS_RATE = 0.99
MAX_LISTED_FAILURES = 20


def _gate_summary(instances: list[TaskInstance]) -> dict[str, Any]:
    gate_pass: Counter[str] = Counter()
    all_pass = 0
    failures: list[dict[str, Any]] = []
    for inst in instances:
        results = run_gates(inst)
        for g in results:
            gate_pass[g.gate] += g.passed
        if all(g.passed for g in results):
            all_pass += 1
        else:
            failed = {g.gate: g.reason for g in results if not g.passed}
            failures.append({"id": inst.id, "broken_type": inst.broken_type, "failed": failed})
    return {
        "n": len(instances),
        "gates": {g: gate_pass[g] for g in ("G1", "G2", "G3")},
        "all_pass": all_pass,
        "failures": failures,
    }


def run_acceptance(data_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}, "missing": []}
    valid_n = valid_pass = 0
    for split in (*VALID_SPLITS, BROKEN_SPLIT):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            report["missing"].append(path.name)
            continue
        instances = read_jsonl(path)
        summary = _gate_summary(instances)
        if split == BROKEN_SPLIT:
            failed = {f["id"]: f["failed"] for f in summary["failures"]}
            by_type: dict[str, dict[str, Any]] = {}
            for inst in instances:
                row = by_type.setdefault(inst.broken_type or "unlabelled", {"n": 0, "rejected": 0, "failed_gates": {}})
                row["n"] += 1
                row["rejected"] += inst.id in failed
                for gate in failed.get(inst.id, {}):
                    row["failed_gates"][gate] = row["failed_gates"].get(gate, 0) + 1
            summary["rejected"] = summary["n"] - summary["all_pass"]
            summary["by_type"] = dict(sorted(by_type.items()))
            # For broken instances the interesting list is the accepted ones.
            summary["accepted_ids"] = sorted(i.id for i in instances if i.id not in failed)[:MAX_LISTED_FAILURES]
            summary.pop("failures")
        else:
            valid_n += summary["n"]
            valid_pass += summary["all_pass"]
            summary["failures"] = summary["failures"][:MAX_LISTED_FAILURES]
        report["splits"][split] = summary

    broken = report["splits"].get(BROKEN_SPLIT)
    rate = valid_pass / valid_n if valid_n else 0.0
    # ">= 99% of valid instances" (§7) is read as pooled over all valid splits.
    report["valid"] = {"n": valid_n, "passed": valid_pass, "pass_rate": rate}
    report["broken_rejected"] = None if broken is None else broken["rejected"]
    checks = {
        "no_missing_files": not report["missing"],
        "all_broken_rejected": broken is not None and broken["rejected"] == broken["n"],
        "valid_pass_rate": valid_n > 0 and rate >= MIN_VALID_PASS_RATE,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quality gates G1-G3 over a built dataset.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    report = run_acceptance(args.data)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
