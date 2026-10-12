"""v3 template dispatch, seeds and splits (SPEC_v3.md §9).

Seeds, ids, "split evenly" and the same-type disjointness rule are v1's (SPEC.md §9), reused from
``envlab.tasks.generate``; only the templates, counts and base seeds differ.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable

from envlab.tasks.common import BudgetError, Built, Draft
from envlab.tasks.generate import content_key, derive_seed, instance_id, template_counts
from envlab.tasks.instance import TaskInstance
from envlab.tasks.v3 import HELDOUT_TEMPLATES, TEMPLATES, TRAIN_TEMPLATES, f1, f2, f3, f4, f5
from envlab.tasks.v3.common import finalize_v3

_BUILDERS: dict[str, Callable[[Draft, str, str], Built]] = {
    "F1": f1.build,
    "F2": f2.build,
    "F3": f3.build,
    "F4": f4.build,
    "F5": f5.build,
}

# split -> (base seed, templates, total instances)
SPLITS: dict[str, tuple[int, tuple[str, ...], int]] = {
    "train": (7001, TRAIN_TEMPLATES, 6000),
    "heldout": (7002, HELDOUT_TEMPLATES, 1000),
}
BROKEN_SEED = 7003
# Unseen train-template instances (ADR 0007 §4), disjoint from `train` by `content_key`.
SAMETYPE_SEED = 7005
SAMETYPE_TOTAL = 500
SAMETYPE_DISJOINT_FROM: tuple[str, ...] = ("train",)
SPLIT_SEEDS: dict[str, int] = {"train": 7001, "heldout": 7002, "broken": BROKEN_SEED, "sametype": SAMETYPE_SEED}
SPLIT_ORDER: tuple[str, ...] = ("train", "heldout", "sametype", "broken")
MAX_ATTEMPTS = 1000
MAX_REDRAWS = 100
VARIANT = "base"


def default_split(template: str) -> str:
    return "heldout" if template in HELDOUT_TEMPLATES else "train"


def generate(template: str, seed: int) -> TaskInstance:
    """One v3 instance. Same (template, seed) => identical instance; the id is provisional.

    A draw that cannot fit the prompt budget at the minimum distractor count is redrawn from
    ``derive_seed(seed, "retry", attempt)``, as in v1 (``envlab.tasks.generate.generate``).

    Args:
        template: A v3 template id, e.g. ``"F1A"``.
        seed: Instance seed; all randomness derives from it.

    Returns:
        The generated instance, with a provisional id.

    Raises:
        ValueError: If the template is not a v3 template.
        BudgetError: If no draw within ``MAX_ATTEMPTS`` fits the prompt budget.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown v3 template {template!r}")
    split = default_split(template)
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(seed if attempt == 0 else derive_seed(seed, "retry", attempt))
        built = _BUILDERS[template[:2]](Draft(rng), template, VARIANT)
        try:
            return finalize_v3(built, rng, seed, split)
        except BudgetError:
            continue
    raise BudgetError(f"{template} seed={seed}: no draw fits the v3 prompt budget")


def sametype_seed(template: str, index: int, attempt: int) -> int:
    if attempt == 0:
        return derive_seed(SAMETYPE_SEED, template, index)
    return derive_seed(SAMETYPE_SEED, template, index, "redraw", attempt)


def _generate_sametype(train: list[TaskInstance] | None = None) -> list[TaskInstance]:
    """Unseen train-template instances; a draw whose `content_key` is in `train` (or earlier here) is redrawn."""
    sources = train if train is not None else generate_split("train")
    seen = {content_key(i) for i in sources}
    out = []
    for template, count in template_counts(SAMETYPE_TOTAL, TRAIN_TEMPLATES).items():
        for index in range(count):
            for attempt in range(MAX_REDRAWS):
                inst = generate(template, sametype_seed(template, index, attempt))
                key = content_key(inst)
                if key not in seen:
                    break
            else:
                raise RuntimeError(f"sametype {template}/{index}: {MAX_REDRAWS} draws all collide with train")
            seen.add(key)
            out.append(dataclasses.replace(inst, id=instance_id(template, "sametype", index), split="sametype"))
    return out


def generate_split(split: str, *, train: list[TaskInstance] | None = None) -> list[TaskInstance]:
    """All instances of a v3 split, ordered by template then index.

    Args:
        split: ``"train"``, ``"heldout"``, ``"sametype"`` or ``"broken"``.
        train: The already generated train split, reused by ``sametype`` to save regenerating it.

    Returns:
        The instances with their final ids.

    Raises:
        ValueError: If the split is unknown.
    """
    if split == "broken":
        from envlab.tasks.v3.broken import generate_broken

        return generate_broken()
    if split == "sametype":
        return _generate_sametype(train)
    if split not in SPLITS:
        raise ValueError(f"unknown v3 split {split!r}")
    base, templates, total = SPLITS[split]
    out = []
    for template, count in template_counts(total, templates).items():
        for index in range(count):
            inst = generate(template, derive_seed(base, template, index))
            out.append(dataclasses.replace(inst, id=instance_id(template, split, index), split=split))
    return out
