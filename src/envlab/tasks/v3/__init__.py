"""Project version v3 task set (ADR 0007, `SPEC_v3.md`): template ids and their roles.

This registry is the interface shared by the v3 generators (``envlab.tasks.v3``) and the training,
eval and report code. Per family: A, B, C are train templates; N is the "near" held-out template
(only recombines trained steps); X is the "far" held-out template (a rule never trained).
"""

from __future__ import annotations

from typing import Literal

FAMILIES: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5")
TRAIN_TEMPLATES: tuple[str, ...] = tuple(f"{f}{t}" for f in FAMILIES for t in "ABC")
NEAR_TEMPLATES: tuple[str, ...] = tuple(f"{f}N" for f in FAMILIES)
FAR_TEMPLATES: tuple[str, ...] = tuple(f"{f}X" for f in FAMILIES)
HELDOUT_TEMPLATES: tuple[str, ...] = (*NEAR_TEMPLATES, *FAR_TEMPLATES)
TEMPLATES: tuple[str, ...] = (*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES)
SPEC_VERSION = "v3.0"
DATA_DIR = "data/v3"

HeldoutKind = Literal["near", "far"]


def is_v3_template(template: str) -> bool:
    return template in TEMPLATES


def heldout_kind(template: str) -> HeldoutKind | None:
    """``"near"`` or ``"far"`` for a v3 held-out template, else None (train or not v3)."""
    if template in NEAR_TEMPLATES:
        return "near"
    if template in FAR_TEMPLATES:
        return "far"
    return None
