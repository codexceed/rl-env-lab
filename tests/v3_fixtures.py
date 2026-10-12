"""Synthetic v3 data for the training / eval / report plumbing tests (ADR 0007).

The real v3 generators live in ``envlab.tasks.v3`` (another workstream). These tests only need instances that
carry v3 template ids: a v1 test instance of the same family, renamed. Grading still works, since the oracle
call and final state come with it.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

from tests.verify.verify_builders import BUILDERS

from envlab.tasks.instance import TaskInstance, write_jsonl
from envlab.tasks.v3 import FAR_TEMPLATES, NEAR_TEMPLATES, SPEC_VERSION, TRAIN_TEMPLATES


def v3_instance(template: str, k: int, split: str) -> TaskInstance:
    """A v1 ``<family>T1`` test instance renamed to the v3 ``template`` (e.g. F3B, F2N, F5X)."""
    family = template[:2]
    base = BUILDERS[f"{family}T1"]()
    return dataclasses.replace(base, id=f"v3-{template}-{split}-{k:05d}", template=template, family=family, split=split)


def v3_split(templates: Iterable[str], per_template: int, split: str) -> list[TaskInstance]:
    return [v3_instance(t, k, split) for t in templates for k in range(per_template)]


def write_v3_data(root: Path, per_template: int = 2, spec_version: str | None = SPEC_VERSION) -> Path:
    """``root`` with train (15 templates), heldout (near + far), sametype and a manifest (``spec_version``)."""
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "train.jsonl", v3_split(TRAIN_TEMPLATES, per_template, "train"))
    write_jsonl(root / "heldout.jsonl", v3_split((*NEAR_TEMPLATES, *FAR_TEMPLATES), per_template, "heldout"))
    write_jsonl(root / "sametype.jsonl", v3_split(TRAIN_TEMPLATES[:3], per_template, "sametype"))
    if spec_version is not None:
        (root / "manifest.json").write_text(json.dumps({"spec_version": spec_version, "files": {}}))
    return root
