"""Task-generator helpers that run without the simulator."""

from __future__ import annotations

import random
import re

import pytest

from envlab.sim.entities import Customer, LineItem
from envlab.tasks import broken, common, f4, f5
from envlab.tasks.generate import (
    FAMILIES,
    HELDOUT_TEMPLATES,
    SPLITS,
    TEMPLATES,
    TRAIN_TEMPLATES,
    derive_seed,
    generate,
    instance_id,
    template_counts,
)

ORIGINAL_TRAIN = ("F1T1", "F1T2", "F2T1", "F2T2", "F3T1", "F3T2", "F4T1", "F4T2", "F5T1", "F5T2")
CANDIDATES = ("F1T4", "F1T5", "F4T4", "F4T5")


def test_derive_seed_known_values():
    assert derive_seed(1001, "F1T1", 0) == 13877934506764428228
    assert derive_seed(3003, "ambiguous", "F3T1", 4) == 5468586683068392437


def test_template_sets():
    assert FAMILIES == ("F1", "F2", "F3", "F4", "F5")
    # The registry holds the original T1/T2 plus any subset of the candidates (one can be dropped).
    assert set(ORIGINAL_TRAIN) <= set(TRAIN_TEMPLATES) <= set(ORIGINAL_TRAIN) | set(CANDIDATES)
    assert tuple(sorted(set(TRAIN_TEMPLATES))) == TRAIN_TEMPLATES  # family order, no duplicates
    assert HELDOUT_TEMPLATES == ("F1T3", "F2T3", "F3T3", "F4T3", "F5T3")
    assert not set(TRAIN_TEMPLATES) & set(HELDOUT_TEMPLATES)
    assert not any(t.endswith("T3") for t in TRAIN_TEMPLATES)  # T3 is the only held-out number
    assert tuple(sorted((*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES))) == TEMPLATES


def test_split_table():
    assert SPLITS["train"] == (1001, TRAIN_TEMPLATES, 2000, "base")
    assert SPLITS["train_hard"] == (4004, TRAIN_TEMPLATES, 2000, "hard")
    assert SPLITS["heldout"] == (2002, HELDOUT_TEMPLATES, 250, "base")
    assert instance_id("F1T3", "heldout", 12) == "F1T3-heldout-00012"


def test_template_counts_split_evenly():
    full = (*ORIGINAL_TRAIN[:4], *CANDIDATES, *ORIGINAL_TRAIN[4:])
    counts = template_counts(2000, full)
    assert sum(counts.values()) == 2000
    assert list(counts.values()) == [143] * 12 + [142] * 2
    assert template_counts(2000, ORIGINAL_TRAIN) == dict.fromkeys(ORIGINAL_TRAIN, 200)
    assert template_counts(250, HELDOUT_TEMPLATES) == dict.fromkeys(HELDOUT_TEMPLATES, 50)
    for n in range(10, 15):  # any number of candidates kept
        c = template_counts(2000, full[:n])
        assert sum(c.values()) == 2000 and max(c.values()) - min(c.values()) <= 1


def test_broken_sources_are_train_templates_only():
    assert broken.BROKEN_TYPES == ("impossible", "ambiguous", "grader_bug", "env_bug")
    for kind, templates in broken.SOURCES.items():
        assert len(templates) == 5
        assert {t[:2] for t in templates} == set(FAMILIES), kind
        assert set(templates) <= set(TRAIN_TEMPLATES), kind
    assert broken.SOURCES["impossible"] == tuple(f"{f}T1" for f in FAMILIES)
    assert broken.SOURCES["grader_bug"] == tuple(f"{f}T2" for f in FAMILIES)
    assert broken.broken_id("F3T1", "ambiguous", 4) == "F3T1-broken-ambiguous-004"


def test_hard_variant_of_heldout_template_is_rejected():
    with pytest.raises(ValueError):
        generate("F1T3", 1, variant="hard")
    with pytest.raises(ValueError):
        generate("F9T1", 1)


def test_formatting_helpers():
    assert common.month_name("2026-08") == "August 2026"
    assert common.dollars(4999) == "$49.99"
    assert common.dollars(100 * 30 - 1) == "$29.99"
    assert common.shift_period("2026-09", 1) == "2026-10"
    assert common.shift_period("2026-12", 1) == "2027-01"
    assert common.shift_period("2026-03", -1) == "2026-02"


@pytest.mark.parametrize("seed", range(200))
def test_inject_error_is_wrong_and_positive(seed):
    rng = random.Random(seed)
    lines = [LineItem("Plan: Pro", 1, rng.randint(1000, 30000))] + [
        LineItem(f"Add-on: {i}", rng.randint(1, 12), rng.randint(500, 5000)) for i in range(rng.randint(1, 3))
    ]
    billed = f4.inject_error(rng, lines)
    s = common.line_sum(lines)
    assert billed > 0
    assert billed != s


def test_wrong_value_table():
    assert f5.wrong_value("email", "a@example.com") == "xa@example.com"
    assert f5.wrong_value("address", "12 Oak St, Salem OR") == "12 Oak St, Salem OR Apt 2"
    assert f5.wrong_value("phone", "+1-415-555-0109") == "+1-415-555-0100"


def test_draft_allocates_unique_well_formed_records():
    d = common.Draft(random.Random(0))
    custs = [d.customer() for _ in range(30)]
    assert len({c.id for c in custs}) == 30
    assert len({c.name for c in custs}) == 30
    assert len({c.name.split(" ")[0] for c in custs}) == 30  # distinct first names by default
    assert len({c.email for c in custs}) == 30
    assert len({c.phone for c in custs}) == 30
    for c in custs:
        assert c.email.split("@")[1] in common.DOMAINS
        assert re.fullmatch(r"\+1-\d{3}-555-\d{4}", c.phone)
        assert c.contract_rate % 100 == 99 and 1999 <= c.contract_rate <= 29999


def test_add_units_reaches_exactly_k_with_required_first():
    d = common.Draft(random.Random(1))
    owner: Customer = d.customer()

    def two(_m):
        o = d.customer()
        return [o, d.invoice(o, "2026-05", d.background_lines(o))]

    req = common.UnitSpec("r", 2, two)
    other = common.UnitSpec("b", 2, two)
    hard_only = common.UnitSpec("h", 1, lambda _m: [d.customer()], hard_only=True)
    for k in (2, 3, 4):
        units = common.add_units(d, k, [req], [other, hard_only], hard=False)
        assert units[0].name == "r" and units[0].required
        assert sum(len(u.records) for u in units) == k
        assert all(u.name != "h" for u in units)
    assert owner.id  # owner is untouched
