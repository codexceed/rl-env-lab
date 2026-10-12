"""State serialization and state hash (SPEC §6 masking, §11.1, §11.4)."""

import hashlib
import json
from dataclasses import fields, replace
from enum import Enum

import pytest
from sim_factory import make_state

from envlab.calls import ToolCall
from envlab.sim.entities import LineItem, State
from envlab.sim.state_hash import canonical_dict, state_hash

GROUPS = ("customers", "invoices", "payments", "cases", "ledger")
TEXT_FIELDS = {"invoices": "note", "payments": "note", "cases": "note", "ledger": "memo"}


def _other(value):
    """A different value of the same kind."""
    if isinstance(value, Enum):
        return next(m for m in type(value) if m != value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    if value is None:
        return 1
    if isinstance(value, tuple):
        return value + (LineItem("Add-on: extra", 1, 1),)
    raise TypeError(type(value))


def _set(state: State, group: str, i: int, record) -> State:
    recs = getattr(state, group)
    return replace(state, **{group: recs[:i] + (record,) + recs[i + 1 :]})


def test_hash_is_stable_across_runs():
    # Pinned values: any change to serialization or hashing breaks this test on purpose.
    s = make_state()
    assert state_hash(s) == "6b3f2e1c367f63903b329233c6ae660e00ef62dff2f146991a05e657ec9c5d00"
    assert (
        state_hash(s, mask_text=True)
        == "43bc7501a1bb610ceabf69e988758a3a8d3231756ffcd6b8673fda77c7cd4fcf"
    )


def test_hash_definition():
    payload = '{"cases":[],"customers":[],"invoices":[],"ledger":[],"payments":[]}'
    assert state_hash(State()) == hashlib.sha256(payload.encode()).hexdigest()
    s = make_state()
    expected = json.dumps(s.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert state_hash(s) == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_same_state_same_hash():
    assert make_state() is not make_state()
    assert state_hash(make_state()) == state_hash(make_state())
    assert state_hash(State.from_dict(make_state().to_dict())) == state_hash(make_state())


def _field_cases():
    s = make_state()
    for group in GROUPS:
        for i in range(len(getattr(s, group))):
            for f in fields(getattr(s, group)[i]):
                yield group, i, f.name


@pytest.mark.parametrize(("group", "i", "field"), list(_field_cases()))
def test_any_field_change_changes_hash(group, i, field):
    s = make_state()
    rec = getattr(s, group)[i]
    changed = _set(s, group, i, replace(rec, **{field: _other(getattr(rec, field))}))
    assert state_hash(changed) != state_hash(s)
    if TEXT_FIELDS.get(group) != field:
        assert state_hash(changed, mask_text=True) != state_hash(s, mask_text=True)


@pytest.mark.parametrize("li_field", ["description", "quantity", "unit_price"])
def test_line_item_change_changes_hash(li_field):
    s = make_state()
    inv = s.invoices[2]
    li = inv.line_items[1]
    new_li = replace(li, **{li_field: _other(getattr(li, li_field))})
    changed = _set(s, "invoices", 2, replace(inv, line_items=(inv.line_items[0], new_li)))
    assert state_hash(changed) != state_hash(s)


@pytest.mark.parametrize("group", GROUPS)
def test_record_order_changes_hash(group):
    s = make_state()
    recs = getattr(s, group)
    assert state_hash(replace(s, **{group: recs[::-1]})) != state_hash(s)


@pytest.mark.parametrize("group", list(TEXT_FIELDS))
def test_masking_ignores_text_content_but_not_presence(group):
    s = make_state()
    field = TEXT_FIELDS[group]
    rec = getattr(s, group)[0]
    a = _set(s, group, 0, replace(rec, **{field: "first wording"}))
    b = _set(s, group, 0, replace(rec, **{field: "  a different wording "}))
    empty = _set(s, group, 0, replace(rec, **{field: ""}))
    blank = _set(s, group, 0, replace(rec, **{field: "  \n "}))
    assert state_hash(a, mask_text=True) == state_hash(b, mask_text=True)
    assert state_hash(a) != state_hash(b)
    assert state_hash(empty, mask_text=True) == state_hash(blank, mask_text=True)
    assert state_hash(a, mask_text=True) != state_hash(empty, mask_text=True)


def test_canonical_dict_masked_values():
    d = canonical_dict(make_state(), mask_text=True)
    assert {i["note"] for i in d["invoices"]} == {""}
    assert [p["note"] for p in d["payments"] if p["note"]] == ["*"]
    assert [c["note"] for c in d["cases"]] == ["", "*", ""]
    assert {e["memo"] for e in d["ledger"]} == {"*"}
    # Non-text fields untouched; summary is not model-written.
    assert d["cases"][0]["summary"] == make_state().cases[0].summary


def test_canonical_dict_does_not_touch_state():
    s = make_state()
    before = s.to_dict()
    canonical_dict(s, mask_text=True)["invoices"].clear()
    assert s.to_dict() == before
    assert canonical_dict(s) == before


def test_state_round_trip():
    s = make_state()
    d = s.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert State.from_dict(d) == s
    assert State.from_dict(State().to_dict()) == State()


def test_toolcall_round_trip():
    c = ToolCall("apply_credit", {"invoice_id": "INV-00001", "amount": 1250, "reason": "r"})
    assert ToolCall.from_dict(json.loads(json.dumps(c.to_dict()))) == c
