"""Splits, broken variants and the data build (SPEC §8, §9). Needs the simulator."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

pytest.importorskip("envlab.sim.ledger")
pytest.importorskip("envlab.sim.tools")
pytest.importorskip("envlab.sim.state_hash")

from envlab.sim.entities import CaseStatus, InvoiceStatus, PaymentStatus
from envlab.sim.ledger import check_invariants
from envlab.sim.state_hash import state_hash
from envlab.sim.tools import apply_tool
from envlab.tasks import generate as gen
from envlab.tasks.broken import BROKEN_TYPES, SOURCES, generate_broken, perturbed_call
from envlab.tasks.build import EXPLORATORY_MANIFEST, SPLIT_ORDER, build_dataset
from envlab.tasks.generate import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, content_key, generate_split, template_counts
from envlab.tasks.instance import TaskInstance, dumps_line, read_jsonl

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def splits() -> dict[str, list[TaskInstance]]:
    return {s: generate_split(s) for s in ("train", "train_hard", "heldout", "broken")}


def test_split_sizes_and_ids(splits):
    assert len(splits["train"]) == 2000
    assert len(splits["train_hard"]) == 2000
    assert len(splits["heldout"]) == 250
    assert len(splits["broken"]) == 100
    assert Counter(i.template for i in splits["train"]) == template_counts(2000, TRAIN_TEMPLATES)
    assert Counter(i.template for i in splits["train_hard"]) == template_counts(2000, TRAIN_TEMPLATES)
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
        assert per_family == dict.fromkeys(("F1", "F2", "F3", "F4", "F5"), 5), kind
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
        "manifest.exploratory.json",
        "manifest.json",
        "sametype.jsonl",
        "splits/broken.txt",
        "splits/heldout.txt",
        "splits/sametype.txt",
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


# ---------------------------------------------------------------- frozen files (live runs read them)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data")
    build_dataset(out)
    return out


def test_frozen_files_match_the_committed_manifest(built):
    """GRPO runs read train.jsonl / heldout.jsonl and hash manifest.json into their identity:
    adding a split must leave every frozen byte unchanged."""
    committed = json.loads((REPO_DATA / "manifest.json").read_text())
    assert (built / "manifest.json").read_bytes() == (REPO_DATA / "manifest.json").read_bytes()
    assert set(committed["files"]) == {f"{s}.jsonl" for s in SPLIT_ORDER} | {f"splits/{s}.txt" for s in SPLIT_ORDER}
    for name, sha in committed["files"].items():
        assert sha256((built / name).read_bytes()).hexdigest() == sha, name
    for split in SPLIT_ORDER:
        assert (built / "splits" / f"{split}.txt").read_bytes() == (REPO_DATA / "splits" / f"{split}.txt").read_bytes()


def test_exploratory_manifest_matches_the_committed_one(built):
    assert (built / EXPLORATORY_MANIFEST).read_bytes() == (REPO_DATA / EXPLORATORY_MANIFEST).read_bytes()
    assert (built / "splits" / "sametype.txt").read_bytes() == (REPO_DATA / "splits" / "sametype.txt").read_bytes()
    m = json.loads((built / EXPLORATORY_MANIFEST).read_text())
    assert m["seeds"] == {"sametype": 5005}
    assert m["counts"] == {"sametype": template_counts(250, TRAIN_TEMPLATES)}
    frozen = json.loads((REPO_DATA / "manifest.json").read_text())["files"]
    assert m["disjoint_from"]["sametype"] == {k: frozen[k] for k in ("train.jsonl", "train_hard.jsonl")}


# ---------------------------------------------------------------- sametype (ADR 0005, exploratory)


@pytest.fixture(scope="module")
def sametype() -> list[TaskInstance]:
    return generate_split("sametype")


def test_sametype_counts_ids_and_fields(sametype):
    assert len(sametype) == 250
    assert Counter(i.template for i in sametype) == template_counts(250, TRAIN_TEMPLATES)
    assert sametype[0].id == "F1T1-sametype-00000"
    assert sametype[-1].id == "F5T2-sametype-00021"
    assert all(i.split == "sametype" and i.variant == "base" for i in sametype)
    assert len({i.id for i in sametype}) == 250


def test_sametype_is_deterministic(sametype):
    assert [dumps_line(i) for i in generate_split("sametype")] == [dumps_line(i) for i in sametype]


def test_sametype_is_disjoint_by_content(sametype, splits):
    keys = [content_key(i) for i in sametype]
    assert len(set(keys)) == len(keys)
    for name in ("train", "train_hard"):
        assert not set(keys) & {content_key(i) for i in splits[name]}, name
    # Seeds alone would not prove it: the key is about what the model sees.
    assert not {i.seed for i in sametype} & {i.seed for i in splits["train"]}


def test_content_key_ignores_record_order_and_sees_the_request(sametype):
    inst = sametype[0]
    s = inst.initial_state
    shuffled = dataclasses.replace(s, customers=s.customers[::-1], invoices=s.invoices[::-1])
    assert content_key(dataclasses.replace(inst, initial_state=shuffled, id="x", seed=1)) == content_key(inst)
    assert content_key(dataclasses.replace(inst, request=inst.request + " ")) != content_key(inst)


def test_colliding_draw_is_redrawn_deterministically(monkeypatch):
    first = gen.generate("F1T1", gen.exploratory_seed("sametype", "F1T1", 0, 0))
    monkeypatch.setitem(gen.EXPLORATORY_SPLITS, "probe", (5005, ("F1T1",), 2, "base", ("src",)))
    real = gen.generate_split
    monkeypatch.setattr(gen, "generate_split", lambda s: [first] if s == "src" else real(s))
    out = gen.generate_split("probe")
    assert [i.id for i in out] == ["F1T1-probe-00000", "F1T1-probe-00001"]
    assert out[0].seed == gen.exploratory_seed("probe", "F1T1", 0, 1)  # first draw collided
    assert out[1].seed == gen.exploratory_seed("probe", "F1T1", 1, 0)
    assert content_key(out[0]) != content_key(first)
    assert [dumps_line(i) for i in gen.generate_split("probe")] == [dumps_line(i) for i in out]
