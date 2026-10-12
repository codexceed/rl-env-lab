"""Template dispatch, seeds and split generation (SPEC §4, §9, §11.6).

Generation needs the simulator (``envlab.sim.ledger.build_ledger`` and
``envlab.sim.tools.apply_tool``); both are imported lazily, so this module imports cleanly
before the sim workstream lands.
"""

import dataclasses
import random
from collections.abc import Callable
from hashlib import sha256

from envlab.tasks import f1, f2, f3, f4, f5
from envlab.tasks.common import BudgetError, Built, Draft, finalize
from envlab.tasks.instance import TaskInstance

FAMILIES: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5")
TRAIN_TEMPLATES: tuple[str, ...] = tuple(f"{f}T{t}" for f in FAMILIES for t in (1, 2))
HELDOUT_TEMPLATES: tuple[str, ...] = tuple(f"{f}T3" for f in FAMILIES)
TEMPLATES: tuple[str, ...] = tuple(f"{f}T{t}" for f in FAMILIES for t in (1, 2, 3))
VARIANTS: tuple[str, ...] = ("base", "hard")

_BUILDERS: dict[str, Callable[[Draft, str, str], Built]] = {
    "F1": f1.build,
    "F2": f2.build,
    "F3": f3.build,
    "F4": f4.build,
    "F5": f5.build,
}

# §9: split -> (base seed, templates, instances per template, variant)
SPLITS: dict[str, tuple[int, tuple[str, ...], int, str]] = {
    "train": (1001, TRAIN_TEMPLATES, 200, "base"),
    "train_hard": (4004, TRAIN_TEMPLATES, 200, "hard"),
    "heldout": (2002, HELDOUT_TEMPLATES, 50, "base"),
}
BROKEN_SEED = 3003
MAX_ATTEMPTS = 1000
SPLIT_SEEDS: dict[str, int] = {**{s: v[0] for s, v in SPLITS.items()}, "broken": BROKEN_SEED}


def derive_seed(base: int, *parts: str | int) -> int:
    """SPEC §9: first 8 bytes (big-endian) of sha256 over ``"base:part1:part2..."``."""
    key = ":".join(map(str, (base, *parts)))
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


def default_split(template: str, variant: str) -> str:
    if variant == "hard":
        return "train_hard"
    return "heldout" if template in HELDOUT_TEMPLATES else "train"


def generate(template: str, seed: int, *, variant: str = "base") -> TaskInstance:
    """One instance of ``template``. Same (template, seed, variant) => identical instance.

    The id is provisional (``{template}-{split}-seed{seed}``); ``generate_split`` assigns the
    §9 id. Hard variants exist for train templates only (§5).

    §10 says the build fails if an instance stays over budget after trimming. Read literally
    that would fail most F2T2-hard builds, whose core plus required units alone are often over
    budget, so an over-budget draw is redrawn instead (rejection sampling, still a pure function
    of ``seed``). ``instance.seed`` stays ``seed``.

    Args:
        template: Template id, e.g. ``"F1T1"``.
        seed: Instance seed; all randomness derives from it.
        variant: ``"base"`` or ``"hard"``.

    Returns:
        The generated instance, with a provisional id.

    Raises:
        ValueError: If the template or variant is unknown, or ``hard`` is asked of a T3.
        BudgetError: If no draw within ``MAX_ATTEMPTS`` fits the prompt budget.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "hard" and template in HELDOUT_TEMPLATES:
        raise ValueError(f"{template} has no hard variant (held out)")
    split = default_split(template, variant)
    for attempt in range(MAX_ATTEMPTS):
        # Attempt 0 draws from ``seed`` itself; a draw that cannot fit the prompt budget even
        # at the minimum distractor count is redrawn from a seed derived from (seed, attempt).
        rng = random.Random(seed if attempt == 0 else derive_seed(seed, "retry", attempt))
        built = _BUILDERS[template[:2]](Draft(rng), template, variant)
        try:
            return finalize(built, rng, seed, split)
        except BudgetError:
            continue
    raise BudgetError(f"{template}/{variant} seed={seed}: no draw fits the prompt budget")


def instance_id(template: str, split: str, index: int) -> str:
    return f"{template}-{split}-{index:05d}"


def generate_split(split: str) -> list[TaskInstance]:
    """All instances of a §9 split, ordered by template then index."""
    if split == "broken":
        from envlab.tasks.broken import generate_broken

        return generate_broken()
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    base, templates, count, variant = SPLITS[split]
    out = []
    for template in templates:
        for index in range(count):
            inst = generate(template, derive_seed(base, template, index), variant=variant)
            out.append(dataclasses.replace(inst, id=instance_id(template, split, index)))
    return out
