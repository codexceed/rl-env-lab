"""Template dispatch, seeds and split generation (SPEC §4, §9, §11.6).

Generation needs the simulator (``envlab.sim.ledger.build_ledger`` and
``envlab.sim.tools.apply_tool``); both are imported lazily, so this module imports cleanly
before the sim workstream lands.
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable
from hashlib import sha256

from envlab.tasks import f1, f2, f3, f4, f5
from envlab.tasks.common import BudgetError, Built, Draft, finalize
from envlab.tasks.instance import TaskInstance

FAMILIES: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5")
# Template registry (SPEC §4, §9). T3 is each family's held-out template; every template listed
# here is a train template (train and train_hard, split evenly). F1T4, F1T5, F4T4 and F4T5 are the
# medium-difficulty candidates: drop one by deleting its entry (and its SPEC §4/§5 rows), then
# rebuild with ``make data``. Order matters only for which templates get the remainder (§9).
# F1T4, F4T4, F4T5 were also built and measured (2B base 6%, 9%, 7%), then dropped (ADR 0004);
# their generators stay so they can be re-added by listing them here.
TRAIN_TEMPLATES: tuple[str, ...] = (
    "F1T1",
    "F1T2",
    "F1T5",
    "F2T1",
    "F2T2",
    "F3T1",
    "F3T2",
    "F4T1",
    "F4T2",
    "F5T1",
    "F5T2",
)
HELDOUT_TEMPLATES: tuple[str, ...] = tuple(f"{f}T3" for f in FAMILIES)
TEMPLATES: tuple[str, ...] = tuple(sorted((*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES)))
VARIANTS: tuple[str, ...] = ("base", "hard")

_BUILDERS: dict[str, Callable[[Draft, str, str], Built]] = {
    "F1": f1.build,
    "F2": f2.build,
    "F3": f3.build,
    "F4": f4.build,
    "F5": f5.build,
}

# §9: split -> (base seed, templates, total instances, variant)
SPLITS: dict[str, tuple[int, tuple[str, ...], int, str]] = {
    "train": (1001, TRAIN_TEMPLATES, 2000, "base"),
    "train_hard": (4004, TRAIN_TEMPLATES, 2000, "hard"),
    "heldout": (2002, HELDOUT_TEMPLATES, 250, "base"),
}
BROKEN_SEED = 3003
MAX_ATTEMPTS = 1000
SPLIT_SEEDS: dict[str, int] = {**{s: v[0] for s, v in SPLITS.items()}, "broken": BROKEN_SEED}

# ADR 0005 item 2: exploratory splits, never used in verdicts. They live outside SPLITS and
# SPLIT_SEEDS so ``data/manifest.json`` (whose sha256 is part of every run's identity,
# train/run_manifest.py) stays byte-identical; build.py writes them to their own manifest.
# split -> (base seed, templates, total instances, variant, splits it must be disjoint from)
EXPLORATORY_SPLITS: dict[str, tuple[int, tuple[str, ...], int, str, tuple[str, ...]]] = {
    "sametype": (5005, TRAIN_TEMPLATES, 250, "base", ("train", "train_hard")),
}
MAX_REDRAWS = 100
# Record groups the prompt shows (SPEC §10); the ledger is derived from them and never shown.
_SHOWN_GROUPS = ("customers", "invoices", "payments", "cases")


def derive_seed(base: int, *parts: str | int) -> int:
    """SPEC §9: first 8 bytes (big-endian) of sha256 over ``"base:part1:part2..."``."""
    key = ":".join(map(str, (base, *parts)))
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


def template_counts(total: int, templates: tuple[str, ...]) -> dict[str, int]:
    """SPEC §9 "split evenly": ``total // n`` each; the first ``total % n`` templates get one more."""
    per, extra = divmod(total, len(templates))
    return {t: per + (i < extra) for i, t in enumerate(templates)}


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


def content_key(instance: TaskInstance) -> str:
    """Identity of what the model is shown: the request plus the shown records.

    sha256 over the request text and, per record group (customers, invoices, payments, cases),
    the sorted canonical JSON of its records, so record order does not matter. Template, variant,
    seed and id are left out: two instances with the same key are the same task to the model.
    """
    state = instance.initial_state.to_dict()
    payload = {
        "request": instance.request,
        **{g: sorted(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in state[g]) for g in _SHOWN_GROUPS},
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exploratory_seed(split: str, template: str, index: int, attempt: int) -> int:
    """Seed of draw ``attempt`` of an exploratory instance: the §9 seed first, then redraw seeds."""
    base = EXPLORATORY_SPLITS[split][0]
    if attempt == 0:
        return derive_seed(base, template, index)
    return derive_seed(base, template, index, "redraw", attempt)


def _generate_exploratory(split: str) -> list[TaskInstance]:
    """Unseen instances of an exploratory split, disjoint by `content_key` from its source splits.

    A draw whose key is already in a source split (or earlier in this split) is skipped for the
    next redraw seed, so the same code always gives the same bytes.
    """
    _, templates, total, variant, disjoint = EXPLORATORY_SPLITS[split]
    seen = {content_key(inst) for name in disjoint for inst in generate_split(name)}
    out = []
    for template, count in template_counts(total, templates).items():
        for index in range(count):
            for attempt in range(MAX_REDRAWS):
                inst = generate(template, exploratory_seed(split, template, index, attempt), variant=variant)
                key = content_key(inst)
                if key not in seen:
                    break
            else:
                raise RuntimeError(f"{split} {template}/{index}: {MAX_REDRAWS} draws all collide with {disjoint}")
            seen.add(key)
            out.append(dataclasses.replace(inst, id=instance_id(template, split, index), split=split))
    return out


def generate_split(split: str) -> list[TaskInstance]:
    """All instances of a §9 split (or an ADR 0005 exploratory split), ordered by template then index."""
    if split == "broken":
        from envlab.tasks.broken import generate_broken

        return generate_broken()
    if split in EXPLORATORY_SPLITS:
        return _generate_exploratory(split)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    base, templates, total, variant = SPLITS[split]
    out = []
    for template, count in template_counts(total, templates).items():
        for index in range(count):
            inst = generate(template, derive_seed(base, template, index), variant=variant)
            out.append(dataclasses.replace(inst, id=instance_id(template, split, index)))
    return out
