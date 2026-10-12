"""Build the v3 data set (SPEC_v3.md §9): ``python -m envlab.tasks.build --set v3 --out data/v3``.

Writes ``train.jsonl``, ``heldout.jsonl``, ``sametype.jsonl``, ``broken.jsonl``, ``splits/*.txt``
and ``manifest.json``. Nothing here touches v1's files.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from envlab.prompts import V3_ESTIMATE_BUDGET, V3_PROMPT_TOKEN_LIMIT, estimate_prompt_tokens
from envlab.tasks.build import CONTENT_KEY_DOC, _write_json, _write_split, check_instance, jsonl_sha256
from envlab.tasks.instance import TaskInstance
from envlab.tasks.v3 import FAR_TEMPLATES, HELDOUT_TEMPLATES, SPEC_VERSION, heldout_kind
from envlab.tasks.v3.generate import SAMETYPE_DISJOINT_FROM, SPLIT_ORDER, SPLIT_SEEDS, generate_split, sametype_seed


def prompt_stats(instances: list[TaskInstance]) -> dict[str, dict[str, int]]:
    """Estimated prompt tokens (SPEC.md §10 estimate) per template: min, median, max."""
    by_template: dict[str, list[int]] = {}
    for inst in instances:
        by_template.setdefault(inst.template, []).append(estimate_prompt_tokens(inst))
    return {
        t: {"min": min(v), "median": int(statistics.median(v)), "max": max(v)} for t, v in sorted(by_template.items())
    }


def build_dataset(out_dir: Path) -> dict[str, Any]:
    """Write every v3 split under ``out_dir`` and return the manifest (also written to disk).

    Args:
        out_dir: Output directory; created if missing.

    Returns:
        The manifest dict, as written to ``manifest.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    files: dict[str, str] = {}
    broken_labels: dict[str, str] = {}
    built: dict[str, list[TaskInstance]] = {}
    for split in SPLIT_ORDER:
        instances = generate_split(split, train=built.get("train"))
        if split == "broken":
            broken_labels = {i.id: str(i.broken_type) for i in instances}
        else:
            for inst in instances:
                check_instance(inst, budget=V3_ESTIMATE_BUDGET)
        built[split] = instances
        counts[split] = dict(sorted(Counter(i.template for i in instances).items()))
        _write_split(out_dir, split, instances, files)
    valid = [i for s in ("train", "heldout") for i in built[s]]
    manifest = {
        "spec_version": SPEC_VERSION,
        "adr": "0007",
        "label": "v3, exploratory",
        "seeds": dict(SPLIT_SEEDS),
        "counts": counts,
        "files": files,
        "heldout_templates": {t: heldout_kind(t) for t in HELDOUT_TEMPLATES},
        "far_templates": list(FAR_TEMPLATES),
        "broken_labels": broken_labels,
        "sametype": {
            "content_key": CONTENT_KEY_DOC,
            "disjoint_from": {f"{s}.jsonl": jsonl_sha256(built[s]) for s in SAMETYPE_DISJOINT_FROM},
            "redrawn": [
                i.id for i in built["sametype"] if i.seed != sametype_seed(i.template, int(i.id.rsplit("-", 1)[1]), 0)
            ],
        },
        "prompt_budget": {"limit_tokens": V3_PROMPT_TOKEN_LIMIT, "estimate_budget": V3_ESTIMATE_BUDGET},
        "prompt_estimate": prompt_stats(valid),
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest
