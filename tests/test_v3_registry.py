from __future__ import annotations

from envlab.tasks.v3 import (
    FAR_TEMPLATES,
    HELDOUT_TEMPLATES,
    NEAR_TEMPLATES,
    TRAIN_TEMPLATES,
    heldout_kind,
    is_v3_template,
)


def test_v3_registry_roles():
    assert len(TRAIN_TEMPLATES) == 15 and len(NEAR_TEMPLATES) == 5 and len(FAR_TEMPLATES) == 5
    assert set(TRAIN_TEMPLATES).isdisjoint(HELDOUT_TEMPLATES)
    assert all(t[:2] in {"F1", "F2", "F3", "F4", "F5"} for t in (*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES))
    assert heldout_kind("F3N") == "near" and heldout_kind("F3X") == "far"
    assert heldout_kind("F3A") is None and heldout_kind("F1T3") is None
    assert is_v3_template("F5C") and not is_v3_template("F1T1")
