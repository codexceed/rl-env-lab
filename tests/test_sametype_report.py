"""eval/sametype_report.py on synthetic records: rates, ordering, markdown, what it reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import sametype_report as rep


def rec(run_id, size, reward, seed, step, template, k, *, parse=True, robust=False, leaky=False):
    return {
        "run_id": run_id,
        "size": size,
        "reward": reward,
        "seed": seed,
        "step": step,
        "instance_id": f"{template}-sametype-{k // 4:05d}",
        "family": template[:2],
        "template": template,
        "sample_idx": k % 4,
        "parse_ok": parse,
        "robust_pass": robust,
        "leaky_pass": leaky,
    }


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def evals(tmp_path):
    root = tmp_path / "evals"
    # 2b leaky s1 @ 300: F1T1 = 8 samples (4 true, 2 exploits, 2 format fails); F5T1 = 4 samples, all true
    leak = [rec("2b-leak-s1", "2b", "leaky", 1, 300, "F1T1", k, robust=True, leaky=True) for k in range(4)]
    leak += [rec("2b-leak-s1", "2b", "leaky", 1, 300, "F1T1", k, leaky=True) for k in range(4, 6)]
    leak += [rec("2b-leak-s1", "2b", "leaky", 1, 300, "F1T1", k, parse=False) for k in range(6, 8)]
    leak += [rec("2b-leak-s1", "2b", "leaky", 1, 300, "F5T1", k, robust=True, leaky=True) for k in range(4)]
    write(root / "2b-leak-s1" / "sametype" / "step-00300.jsonl", leak)
    write(root / "2b-leak-s1" / "sametype" / "step-00050.jsonl", [rec("2b-leak-s1", "2b", "leaky", 1, 50, "F1T1", 0)])
    write(root / "2b-rob-s1" / "sametype" / "step-00300.jsonl", [rec("2b-rob-s1", "2b", "robust", 1, 300, "F1T1", 0)])
    write(root / "base-2b" / "sametype" / "step-00000.jsonl", [rec("base-2b", "2b", "base", None, 0, "F1T1", 0)])
    write(
        root / "0.8b-rob-s2" / "sametype" / "step-00100.jsonl",
        [rec("0.8b-rob-s2", "0.8b", "robust", 2, 100, "F2T1", 0)],
    )
    # never read: formal records, thinking-on sametype, diagnostic, traces
    write(root / "2b-leak-s1" / "step-00300.jsonl", [rec("2b-leak-s1", "2b", "leaky", 1, 300, "F1T3", 0)])
    write(root / "2b-leak-s1" / "sametype-think" / "step-00300.jsonl", [rec("x", "2b", "leaky", 1, 300, "F1T1", 0)])
    write(root / "diagnostic" / "base-2b_step-00000.jsonl", [rec("base-2b", "2b", "base", None, 0, "F1T1", 0)])
    write(root / "2b-leak-s1" / "sametype" / "step-00300.traces.jsonl", [{"junk": 1}])
    return root


def test_reads_only_sametype_records(evals):
    records, notes = rep.read_records(evals)
    assert sorted(records) == [
        ("0.8b-rob-s2", 100),
        ("2b-leak-s1", 50),
        ("2b-leak-s1", 300),
        ("2b-rob-s1", 300),
        ("base-2b", 0),
    ]
    assert len(records[("2b-leak-s1", 300)]) == 12 and notes == []


def test_rates_overall_and_per_template(evals):
    entries = rep.table_rows(rep.read_records(evals)[0])
    assert [(e["run_id"], e["step"]) for e in entries] == [
        ("0.8b-rob-s2", 100),
        ("base-2b", 0),
        ("2b-rob-s1", 300),
        ("2b-leak-s1", 50),
        ("2b-leak-s1", 300),
    ]
    e = entries[-1]
    assert e["instances"] == 3 and (e["size"], e["reward"], e["seed"]) == ("2b", "leaky", 1)
    assert e["overall"] == {"n": 12, "true_success": 8 / 12, "exploit": 2 / 12, "format_failure": 2 / 12}
    assert e["templates"]["F1T1"] == {"n": 8, "true_success": 0.5, "exploit": 0.25, "format_failure": 0.25}
    assert e["templates"]["F5T1"] == {"n": 4, "true_success": 1.0, "exploit": 0.0, "format_failure": 0.0}


def test_markdown_is_labelled_exploratory(evals):
    records, notes = rep.read_records(evals)
    md = rep.markdown(rep.table_rows(records), notes)
    assert md.startswith("# ID test (EXPLORATORY, ADR 0005)")
    assert "Never used in an H1/H2/H5 verdict" in md
    assert "| 2b-leak-s1 | 2b | leaky | 1 | 300 | 12 | 66.7% | 16.7% | 16.7% |" in md
    assert "| base-2b | 2b | base | - | 0 | 1 | 0.0% | 0.0% | 0.0% |" in md
    for title in ("True success by template", "Exploit by template", "Format failure by template"):
        assert f"## {title}" in md
    assert "| Run | Step | F1T1 | F2T1 | F5T1 |" in md
    assert "| 2b-leak-s1 | 300 | 50.0% | - | 100.0% |" in md  # true success; F2T1 not evaluated


def test_unreadable_lines_are_noted(tmp_path):
    f = tmp_path / "evals" / "2b-rob-s1" / "sametype" / "step-00050.jsonl"
    write(f, [rec("2b-rob-s1", "2b", "robust", 1, 50, "F1T1", 0)])
    with open(f, "a") as fh:
        fh.write('{"truncated": \n{"parse_ok": 1}\n')
    records, notes = rep.read_records(tmp_path / "evals")
    assert len(records[("2b-rob-s1", 50)]) == 1
    assert notes == ["2b-rob-s1/sametype/step-00050.jsonl: skipped 2 unreadable line(s)"]


def test_cli(evals, tmp_path, capsys):
    out = tmp_path / "reports" / "sametype.md"
    assert rep.main(["--evals", str(evals), "--out", str(out)]) == 0
    assert out.read_text().startswith("# ID test (EXPLORATORY")
    assert rep.main(["--evals", str(tmp_path / "none"), "--out", "-"]) == 0
    assert "No ID-test records" in capsys.readouterr().out
