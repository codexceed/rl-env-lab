"""Splits, broken variants and the data build (SPEC §8, §9). Needs the simulator."""

import json
from collections import Counter

import pytest

pytest.importorskip("envlab.sim.ledger")
pytest.importorskip("envlab.sim.tools")
pytest.importorskip("envlab.sim.state_hash")

from envlab.sim.entities import CaseStatus, InvoiceStatus, PaymentStatus  # noqa: E402
from envlab.sim.ledger import check_invariants  # noqa: E402
from envlab.sim.state_hash import state_hash  # noqa: E402
from envlab.sim.tools import apply_tool  # noqa: E402
from envlab.tasks.broken import BROKEN_TYPES, SOURCES, generate_broken, perturbed_call  # noqa: E402
from envlab.tasks.build import build_dataset  # noqa: E402
from envlab.tasks.generate import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, generate_split  # noqa: E402
from envlab.tasks.instance import TaskInstance, read_jsonl  # noqa: E402


@pytest.fixture(scope="module")
def splits() -> dict[str, list[TaskInstance]]:
    return {s: generate_split(s) for s in ("train", "train_hard", "heldout", "broken")}


def test_split_sizes_and_ids(splits):
    assert len(splits["train"]) == 2000
    assert len(splits["train_hard"]) == 2000
    assert len(splits["heldout"]) == 250
    assert len(splits["broken"]) == 100
    assert Counter(i.template for i in splits["train"]) == dict.fromkeys(TRAIN_TEMPLATES, 200)
    assert Counter(i.template for i in splits["heldout"]) == dict.fromkeys(HELDOUT_TEMPLATES, 50)
    assert splits["train"][7].id == "F1T1-train-00007"
    assert splits["heldout"][12].id == "F1T3-heldout-00012"
    all_ids = [i.id for s in splits.values() for i in s]
    assert len(all_ids) == len(set(all_ids))
    for name, insts in splits.items():
        assert all(i.split == name for i in insts)


def test_no_t3_in_train_and_only_t3_in_heldout(splits):
    for name in ("train", "train_hard", "broken"):
        assert not {i.template for i in splits[name]} & set(HELDOUT_TEMPLATES), name
    assert {i.template for i in splits["heldout"]} == set(HELDOUT_TEMPLATES)
    assert {i.variant for i in splits["train"]} == {"base"}
    assert {i.variant for i in splits["train_hard"]} == {"hard"}


def test_distractors_two_to_four_in_base_splits(splits):
    for name in ("train", "heldout"):
        assert all(2 <= i.meta["n_distractors"] <= 4 for i in splits[name]), name


def test_broken_counts_and_types(splits):
    b = splits["broken"]
    assert Counter(i.broken_type for i in b) == dict.fromkeys(BROKEN_TYPES, 25)
    for kind in BROKEN_TYPES:
        per_family = Counter(i.family for i in b if i.broken_type == kind)
        assert per_family == {f: 5 for f in ("F1", "F2", "F3", "F4", "F5")}, kind
        assert {i.template for i in b if i.broken_type == kind} == set(SOURCES[kind])
    assert all(i.variant == "broken" and i.split == "broken" for i in b)
    assert b[0].id == "F1T1-broken-impossible-000"


def _oracle(inst: TaskInstance):
    return apply_tool(inst.initial_state, inst.oracle_call)


def test_broken_mutations_have_their_intended_defect(splits):
    for inst in splits["broken"]:
        s = inst.initial_state
        final, result = _oracle(inst)
        if inst.broken_type == "impossible":
            assert not result.ok and result.error == "not_found", inst.id
            assert check_invariants(s) == []
        elif inst.broken_type == "ambiguous":
            assert result.ok and check_invariants(s) == []
            assert state_hash(final) == state_hash(inst.oracle_final_state)
        elif inst.broken_type == "grader_bug":
            assert result.ok and check_invariants(s) == []
            assert state_hash(final, mask_text=True) != state_hash(inst.oracle_final_state, mask_text=True)
            assert apply_tool(s, perturbed_call(inst))[1].ok
        else:
            assert check_invariants(s) != [], inst.id
            assert result.ok == (inst.family == "F5"), inst.id


def test_ambiguous_adds_a_second_target(splits):
    for inst in (i for i in splits["broken"] if i.broken_type == "ambiguous"):
        s, p = inst.initial_state, inst.params
        if inst.family == "F1":
            c = next(c for c in s.customers if c.name == p["customer_name"])
            hits = [
                i
                for i in s.invoices
                if i.customer_id == c.id
                and i.period == p["period"]
                and i.status is InvoiceStatus.OPEN
                and i.line_items[0].unit_price > c.contract_rate
            ]
        elif inst.family == "F2":
            c = next(c for c in s.customers if c.name == p["customer_name"])
            hits = [
                i
                for i in s.invoices
                if i.customer_id == c.id
                and len([q for q in s.payments if q.invoice_id == i.id and q.status is PaymentStatus.SUCCEEDED]) == 2
            ]
        elif inst.family == "F3":
            assert set(p) == {"invoice_id", "rule", "percent"}
            hits = [k for k in s.cases if k.invoice_id == p["invoice_id"] and k.status is CaseStatus.OPEN]
        elif inst.family == "F4":
            c = next(c for c in s.customers if c.name == p["customer_name"])
            hits = [
                i
                for i in s.invoices
                if i.customer_id == c.id
                and i.status is InvoiceStatus.OPEN
                and i.amount_billed != sum(li.total for li in i.line_items)
            ]
        else:
            hits = [c for c in s.customers if all(getattr(c, k) == v for k, v in p["lookup"].items())]
        assert len(hits) == 2, inst.id


def test_generate_broken_is_deterministic(splits):
    from envlab.tasks.instance import dumps_line

    assert [dumps_line(i) for i in generate_broken()] == [dumps_line(i) for i in splits["broken"]]


def test_build_is_byte_identical(tmp_path):
    m1 = build_dataset(tmp_path / "a")
    m2 = build_dataset(tmp_path / "b")
    assert m1 == m2
    names = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert [str(n) for n in names] == [
        "broken.jsonl",
        "heldout.jsonl",
        "manifest.json",
        "splits/broken.txt",
        "splits/heldout.txt",
        "splits/train.txt",
        "splits/train_hard.txt",
        "train.jsonl",
        "train_hard.jsonl",
    ]
    for n in names:
        assert (tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes(), n

    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest == m1
    assert set(manifest) == {"spec_version", "seeds", "counts", "files", "heldout_templates", "broken_labels"}
    assert manifest["seeds"] == {"train": 1001, "train_hard": 4004, "heldout": 2002, "broken": 3003}
    assert manifest["heldout_templates"] == ["F1T3", "F2T3", "F3T3", "F4T3", "F5T3"]
    assert len(manifest["broken_labels"]) == 100
    assert {sum(c.values()) for c in manifest["counts"].values()} == {2000, 250, 100}
    ids = (tmp_path / "a" / "splits" / "heldout.txt").read_text().splitlines()
    assert ids == [i.id for i in read_jsonl(tmp_path / "a" / "heldout.jsonl")]
