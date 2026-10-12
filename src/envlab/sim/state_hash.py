"""Canonical state form and stable hash (SPEC §6 text masking, §11.4)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from envlab.sim.entities import State

# Model-written text fields, compared as set / empty only when masked (SPEC §6).
MASKED_FIELDS: dict[str, str] = {
    "invoices": "note",
    "payments": "note",
    "cases": "note",
    "ledger": "memo",
}


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return "*" if value.strip() else ""
    return value


def canonical_dict(state: State, *, mask_text: bool = False) -> dict[str, Any]:
    """`state.to_dict()`, with note/memo fields replaced by "*" or "" when `mask_text`."""
    d = state.to_dict()
    if mask_text:
        for group, field in MASKED_FIELDS.items():
            for record in d[group]:
                record[field] = _mask(record[field])
    return d


def state_hash(state: State, *, mask_text: bool = False) -> str:
    payload = json.dumps(
        canonical_dict(state, mask_text=mask_text),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
