"""eval/v2_report.py on synthetic records: latest step per run, robust vs leaky, labelled exploratory."""

from __future__ import annotations

import json
from pathlib import Path

from eval import v2_report


def write(path: Path, n: int, robust: int, exploit: int) -> None:
    rows = []
    for k in range(n):
        r = k < robust
        rows.append({"parse_ok": True, "robust_pass": r, "leaky_pass": r or k < robust + exploit, "template": "F1T3"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"truncated": \n')


def test_v2_table(tmp_path):
    ev = tmp_path / "evals"
    write(ev / "v2-2b-rob-s1" / "step-00050.jsonl", 10, 1, 0)  # older step: ignored
    write(ev / "v2-2b-rob-s1" / "step-00100.jsonl", 10, 6, 1)
    write(ev / "v2-2b-rob-s2" / "step-00100.jsonl", 10, 4, 1)
    write(ev / "v2-2b-leak-s1" / "step-00100.jsonl", 10, 2, 7)
    write(ev / "v2-2b-leak-s1" / "sametype" / "step-00100.jsonl", 10, 5, 5)
    write(ev / "2b-rob-s1" / "step-00100.jsonl", 10, 10, 0)  # v1: not in this table
    write(ev / "v2-2b-rob-s1-x" / "step-00100.jsonl", 10, 10, 0)  # not a v2 run id
    s = v2_report.summarize(ev)
    assert sorted(s["heldout"]["runs"]) == ["v2-2b-leak-s1", "v2-2b-rob-s1", "v2-2b-rob-s2"]
    assert s["heldout"]["runs"]["v2-2b-rob-s1"]["step"] == 100
    assert s["heldout"]["groups"]["2b robust"] == {"seeds": 2, "true_success": 0.5, "exploit": 0.1}
    assert s["heldout"]["groups"]["2b leaky"] == {"seeds": 1, "true_success": 0.2, "exploit": 0.7}
    assert list(s["sametype"]["runs"]) == ["v2-2b-leak-s1"]
    md = v2_report.markdown(s)
    assert md.startswith("# v2 round (EXPLORATORY, ADR 0005)") and "Never used in an H1/H2/H5" in md
    assert "| 2b robust | 2 | 50.0% | 10.0% |" in md and "| v2-2b-leak-s1 | 100 | 10 | 20.0% | 70.0% | 0.0% |" in md


def test_v2_cli_without_data(tmp_path, capsys):
    assert v2_report.main(["--evals", str(tmp_path), "--out", "-"]) == 0
    assert capsys.readouterr().out.count("No v2 evals yet.") == 2
