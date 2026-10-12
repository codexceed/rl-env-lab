"""Build every split to disk (SPEC §9): ``python -m envlab.tasks.build --out data``."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from envlab.prompts import ESTIMATE_BUDGET, estimate_prompt_tokens
from envlab.tasks.common import SPEC_VERSION
from envlab.tasks.generate import HELDOUT_TEMPLATES, SPLIT_SEEDS, generate_split
from envlab.tasks.instance import TaskInstance, write_jsonl

SPLIT_ORDER: tuple[str, ...] = ("train", "train_hard", "heldout", "broken")


class BuildError(RuntimeError):
    """A generated instance broke a §9 build assertion."""


def check_instance(inst: TaskInstance) -> None:
    """§9 build assertions for a non-broken instance: invariants, oracle call, prompt budget.

    Args:
        inst: A valid (non-broken) instance.

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
    if est > ESTIMATE_BUDGET:
        raise BuildError(f"{inst.id}: prompt estimate {est} > {ESTIMATE_BUDGET}")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_dataset(out_dir: Path) -> dict[str, Any]:
    """Write the §9 files under ``out_dir`` and return the manifest (also written to disk).

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
        data_path = out_dir / f"{split}.jsonl"
        write_jsonl(data_path, instances)
        ids_path = out_dir / "splits" / f"{split}.txt"
        ids_path.write_text("".join(f"{i.id}\n" for i in instances), encoding="utf-8")
        files[data_path.name] = _sha256(data_path)
        files[f"splits/{ids_path.name}"] = _sha256(ids_path)
    manifest = {
        "spec_version": SPEC_VERSION,
        "seeds": dict(SPLIT_SEEDS),
        "counts": counts,
        "files": files,
        "heldout_templates": list(HELDOUT_TEMPLATES),
        "broken_labels": broken_labels,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    args = parser.parse_args(argv)
    manifest = build_dataset(args.out)
    total = {split: sum(c.values()) for split, c in manifest["counts"].items()}
    print(json.dumps({"out": str(args.out), "counts": total}, sort_keys=True))


if __name__ == "__main__":
    main()
