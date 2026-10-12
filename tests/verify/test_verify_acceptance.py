"""`python -m envlab.verify.acceptance` over a small hand-built dataset (SPEC §7, §9)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from verify_builders import BROKEN, BROKEN_SOURCE, BUILDERS

from envlab.tasks.instance import TaskInstance, write_jsonl
from envlab.verify.acceptance import main, run_acceptance


def _write(data: Path, valid_extra: Sequence[TaskInstance] = (), broken_extra: Sequence[TaskInstance] = ()) -> None:
    valid = [b() for t, b in sorted(BUILDERS.items()) if not t.endswith("3")]
    heldout = [b() for t, b in sorted(BUILDERS.items()) if t.endswith("3")]
    broken = [fn(BROKEN_SOURCE[kind][fam]()) for kind, fn in BROKEN.items() for fam in sorted(BROKEN_SOURCE[kind])]
    write_jsonl(data / "train.jsonl", [*valid, *valid_extra])
    write_jsonl(data / "train_hard.jsonl", valid)
    write_jsonl(data / "heldout.jsonl", heldout)
    write_jsonl(data / "broken.jsonl", [*broken, *broken_extra])


def test_acceptance_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _write(tmp_path)
    assert main(["--data", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"]
    assert report["valid"]["pass_rate"] == 1.0
    assert report["broken_rejected"] == 20
    assert report["splits"]["broken"]["by_type"]["ambiguous"] == {"n": 5, "rejected": 5, "failed_gates": {"G2": 5}}
    assert report["splits"]["heldout"]["gates"] == {"G1": 5, "G2": 5, "G3": 5}


def test_acceptance_fails_on_accepted_broken(tmp_path: Path):
    sneaky = dataclasses.replace(BUILDERS["F1T1"](), id="F1T1-broken-env_bug-099", broken_type="env_bug")
    _write(tmp_path, broken_extra=[sneaky])
    report = run_acceptance(tmp_path)
    assert not report["passed"]
    assert report["checks"]["all_broken_rejected"] is False
    assert report["splits"]["broken"]["accepted_ids"] == ["F1T1-broken-env_bug-099"]
    assert main(["--data", str(tmp_path)]) == 1


def test_acceptance_fails_below_99_percent(tmp_path: Path):
    bad = BROKEN["ambiguous"](BUILDERS["F1T2"]())
    _write(tmp_path, valid_extra=[dataclasses.replace(bad, split="train", broken_type=None)])
    report = run_acceptance(tmp_path)
    assert report["valid"]["passed"] == report["valid"]["n"] - 1
    assert report["checks"]["valid_pass_rate"] is False
    assert report["splits"]["train"]["failures"][0]["failed"] == {"G2": "multiple_solutions"}


def test_acceptance_fails_on_missing_split(tmp_path: Path):
    _write(tmp_path)
    (tmp_path / "train_hard.jsonl").unlink()
    report = run_acceptance(tmp_path)
    assert report["missing"] == ["train_hard.jsonl"]
    assert not report["passed"]
