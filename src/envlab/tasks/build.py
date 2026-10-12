"""Build every split to disk (SPEC §9): ``python -m envlab.tasks.build --out data``.

``--only-exploratory`` writes just the ADR 0005 exploratory splits (``sametype``) and their
manifest, without touching the frozen files: safe on a node whose runs are reading ``data/``.

``--set v3`` builds the project version v3 task set instead (ADR 0007, SPEC_v3.md §9) into
``data/v3`` by default (``envlab.tasks.v3.build``). The default ``--set v1`` is unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from envlab.prompts import ESTIMATE_BUDGET, estimate_prompt_tokens
from envlab.tasks.common import SPEC_VERSION
from envlab.tasks.generate import EXPLORATORY_SPLITS, HELDOUT_TEMPLATES, SPLIT_SEEDS, exploratory_seed, generate_split
from envlab.tasks.instance import TaskInstance, dumps_line, write_jsonl

# Frozen splits, recorded in manifest.json. Its sha256 is part of every run's identity
# (train/run_manifest.py), so nothing may be added here while runs are live.
SPLIT_ORDER: tuple[str, ...] = ("train", "train_hard", "heldout", "broken")
# ADR 0005 exploratory splits, built after the frozen ones and recorded in their own manifest.
EXPLORATORY_ORDER: tuple[str, ...] = ("sametype",)
EXPLORATORY_MANIFEST = "manifest.exploratory.json"
CONTENT_KEY_DOC = (
    "sha256 of the request text plus the shown records (customers, invoices, payments, cases), "
    "each group order-insensitive (envlab.tasks.generate.content_key)"
)


class BuildError(RuntimeError):
    """A generated instance broke a §9 build assertion."""


def check_instance(inst: TaskInstance, *, budget: int = ESTIMATE_BUDGET) -> None:
    """§9 build assertions for a non-broken instance: invariants, oracle call, prompt budget.

    Args:
        inst: A valid (non-broken) instance.
        budget: Estimated-token budget (v1 ``ESTIMATE_BUDGET``; v3 passes its own).

    Raises:
        BuildError: If any assertion fails.
    """
    from envlab.sim.ledger import check_invariants
    from envlab.sim.state_hash import state_hash
    from envlab.sim.tools import apply_tool

    problems = check_invariants(inst.initial_state)
    if problems:
        raise BuildError(f"{inst.id}: initial invariants: {problems}")
    final, result = apply_tool(inst.initial_state, inst.oracle_call)
    if not result.ok:
        raise BuildError(f"{inst.id}: oracle call failed: {result.error}")
    if state_hash(final, mask_text=True) != state_hash(inst.oracle_final_state, mask_text=True):
        raise BuildError(f"{inst.id}: oracle_final_state does not match the oracle call")
    problems = check_invariants(inst.oracle_final_state)
    if problems:
        raise BuildError(f"{inst.id}: final invariants: {problems}")
    est = estimate_prompt_tokens(inst)
    if est > budget:
        raise BuildError(f"{inst.id}: prompt estimate {est} > {budget}")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def jsonl_sha256(instances: Iterable[TaskInstance]) -> str:
    """sha256 of the bytes `write_jsonl` would write, without touching disk."""
    h = sha256()
    for inst in instances:
        h.update((dumps_line(inst) + "\n").encode("utf-8"))
    return h.hexdigest()


def _write_split(out_dir: Path, split: str, instances: list[TaskInstance], files: dict[str, str]) -> None:
    data_path = out_dir / f"{split}.jsonl"
    write_jsonl(data_path, instances)
    ids_path = out_dir / "splits" / f"{split}.txt"
    ids_path.write_text("".join(f"{i.id}\n" for i in instances), encoding="utf-8")
    files[data_path.name] = _sha256(data_path)
    files[f"splits/{ids_path.name}"] = _sha256(ids_path)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_exploratory(out_dir: Path) -> dict[str, Any]:
    """Write the ADR 0005 exploratory splits and ``manifest.exploratory.json``; frozen files untouched.

    Args:
        out_dir: Output directory; created if missing.

    Returns:
        The exploratory manifest dict, as written to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    files: dict[str, str] = {}
    disjoint: dict[str, dict[str, str]] = {}
    redrawn: dict[str, list[str]] = {}
    for split in EXPLORATORY_ORDER:
        instances = generate_split(split)
        for inst in instances:
            check_instance(inst)
        counts[split] = dict(sorted(Counter(i.template for i in instances).items()))
        _write_split(out_dir, split, instances, files)
        # The sources' sha256 ties the disjointness claim to the exact frozen files.
        disjoint[split] = {f"{s}.jsonl": jsonl_sha256(generate_split(s)) for s in EXPLORATORY_SPLITS[split][4]}
        redrawn[split] = [
            i.id for i in instances if i.seed != exploratory_seed(split, i.template, int(i.id.rsplit("-", 1)[1]), 0)
        ]
    manifest = {
        "adr": "0005",
        "label": "EXPLORATORY: never used in verdicts",
        "spec_version": SPEC_VERSION,
        "seeds": {s: EXPLORATORY_SPLITS[s][0] for s in EXPLORATORY_ORDER},
        "counts": counts,
        "files": files,
        "content_key": CONTENT_KEY_DOC,
        "disjoint_from": disjoint,
        "redrawn": redrawn,
    }
    _write_json(out_dir / EXPLORATORY_MANIFEST, manifest)
    return manifest


def build_dataset(out_dir: Path) -> dict[str, Any]:
    """Write the §9 files under ``out_dir`` and return the manifest (also written to disk).

    The ADR 0005 exploratory splits are written afterwards (`build_exploratory`); they never
    change ``manifest.json``.

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
    for split in SPLIT_ORDER:
        instances = generate_split(split)
        if split == "broken":
            broken_labels = {i.id: str(i.broken_type) for i in instances}
        else:
            for inst in instances:
                check_instance(inst)
        counts[split] = dict(sorted(Counter(i.template for i in instances).items()))
        _write_split(out_dir, split, instances, files)
    manifest = {
        "spec_version": SPEC_VERSION,
        "seeds": dict(SPLIT_SEEDS),
        "counts": counts,
        "files": files,
        "heldout_templates": list(HELDOUT_TEMPLATES),
        "broken_labels": broken_labels,
    }
    _write_json(out_dir / "manifest.json", manifest)
    build_exploratory(out_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="task_set",
        choices=("v1", "v3"),
        default="v1",
        help="task set: v1 (SPEC.md, default) or v3 (SPEC_v3.md, ADR 0007)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: data, or data/v3)")
    parser.add_argument(
        "--only-exploratory",
        action="store_true",
        help="v1 only: write only the ADR 0005 exploratory splits; leave the frozen files untouched",
    )
    args = parser.parse_args(argv)
    counts: dict[str, dict[str, int]] = {}
    if args.task_set == "v3":
        from envlab.tasks.v3 import DATA_DIR
        from envlab.tasks.v3.build import build_dataset as build_v3

        if args.only_exploratory:
            parser.error("--only-exploratory applies to v1 only")
        out = args.out or Path(DATA_DIR)
        counts = build_v3(out)["counts"]
        print(json.dumps({"out": str(out), "counts": {s: sum(c.values()) for s, c in counts.items()}}, sort_keys=True))
        return
    args.out = args.out or Path("data")
    if args.only_exploratory:
        counts |= build_exploratory(args.out)["counts"]
    else:
        counts |= build_dataset(args.out)["counts"]  # also writes the exploratory splits
        counts |= json.loads((args.out / EXPLORATORY_MANIFEST).read_text(encoding="utf-8"))["counts"]
    total = {split: sum(c.values()) for split, c in counts.items()}
    print(json.dumps({"out": str(args.out), "counts": total}, sort_keys=True))


if __name__ == "__main__":
    main()
