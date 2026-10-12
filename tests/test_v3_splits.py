"""v3 splits, broken variants and the build (SPEC_v3.md §8, §9), plus v1's frozen bytes."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from envlab.tasks import build as build_cli
from envlab.tasks.broken import BROKEN_TYPES
from envlab.tasks.generate import content_key, template_counts
from envlab.tasks.instance import TaskInstance, read_jsonl
from envlab.tasks.v3 import FAR_TEMPLATES, HELDOUT_TEMPLATES, NEAR_TEMPLATES, TRAIN_TEMPLATES, heldout_kind
from envlab.tasks.v3.broken import SOURCES
from envlab.tasks.v3.build import build_dataset
from envlab.tasks.v3.generate import SPLIT_SEEDS, generate_split
from envlab.verify.acceptance import run_acceptance
from envlab.verify.gates import run_gates

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def splits() -> dict[str, list[TaskInstance]]:
    train = generate_split("train")
    return {
        "train": train,
        "heldout": generate_split("heldout"),
        "sametype": generate_split("sametype", train=train),
        "broken": generate_split("broken"),
    }


def test_split_sizes_ids_and_templates(splits):
    assert {s: len(v) for s, v in splits.items()} == {"train": 6000, "heldout": 1000, "sametype": 500, "broken": 100}
    assert Counter(i.template for i in splits["train"]) == dict.fromkeys(TRAIN_TEMPLATES, 400)
    assert Counter(i.template for i in splits["heldout"]) == dict.fromkeys(HELDOUT_TEMPLATES, 100)
    assert Counter(i.template for i in splits["sametype"]) == template_counts(500, TRAIN_TEMPLATES)
    assert splits["train"][7].id == "F1A-train-00007"
    assert splits["heldout"][0].id == "F1N-heldout-00000"
    assert splits["broken"][0].id == "F1A-broken-impossible-000"
    ids = [i.id for v in splits.values() for i in v]
    assert len(ids) == len(set(ids))
    for name, insts in splits.items():
        assert all(i.split == name for i in insts)


def test_heldout_templates_never_trained(splits):
    for name in ("train", "sametype", "broken"):
        assert not {i.template for i in splits[name]} & set(HELDOUT_TEMPLATES)
    near = sum(heldout_kind(i.template) == "near" for i in splits["heldout"])
    far = sum(heldout_kind(i.template) == "far" for i in splits["heldout"])
    assert (near, far) == (500, 500)


def test_sametype_is_disjoint_from_train(splits):
    train_keys = {content_key(i) for i in splits["train"]}
    same_keys = [content_key(i) for i in splits["sametype"]]
    assert not train_keys & set(same_keys)
    assert len(same_keys) == len(set(same_keys))


def test_broken_counts_types_and_rejection(splits):
    broken = splits["broken"]
    assert Counter(i.broken_type for i in broken) == dict.fromkeys(BROKEN_TYPES, 25)
    for kind in BROKEN_TYPES:
        assert {i.template for i in broken if i.broken_type == kind} == set(SOURCES[kind])
        assert Counter(i.family for i in broken if i.broken_type == kind) == dict.fromkeys(
            ("F1", "F2", "F3", "F4", "F5"), 5
        )
    for inst in broken:
        assert not all(g.passed for g in run_gates(inst)), inst.id
        if inst.broken_type == "ambiguous":
            assert [g.reason for g in run_gates(inst)][1] == "multiple_solutions", inst.id


def test_valid_instances_pass_all_gates(splits):
    for name in ("train", "heldout", "sametype"):
        failed = [i.id for i in splits[name] if not all(g.passed for g in run_gates(i))]
        assert failed == [], (name, failed[:5])


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v3")
    build_cli.main(["--set", "v3", "--out", str(out)])
    return out


def test_build_writes_manifest_and_files(built):
    manifest = json.loads((built / "manifest.json").read_text())
    assert manifest["spec_version"] == "v3.0"
    assert manifest["seeds"] == SPLIT_SEEDS == {"train": 7001, "heldout": 7002, "broken": 7003, "sametype": 7005}
    assert {s: sum(c.values()) for s, c in manifest["counts"].items()} == {
        "train": 6000,
        "heldout": 1000,
        "sametype": 500,
        "broken": 100,
    }
    assert manifest["heldout_templates"] == {
        **dict.fromkeys(NEAR_TEMPLATES, "near"),
        **dict.fromkeys(FAR_TEMPLATES, "far"),
    }
    assert len(manifest["broken_labels"]) == 100
    for name, digest in manifest["files"].items():
        assert sha256((built / name).read_bytes()).hexdigest() == digest, name
    assert manifest["sametype"]["disjoint_from"]["train.jsonl"] == manifest["files"]["train.jsonl"]
    stats = manifest["prompt_estimate"]
    assert set(stats) == set(TRAIN_TEMPLATES) | set(HELDOUT_TEMPLATES)
    assert all(s["max"] <= manifest["prompt_budget"]["estimate_budget"] for s in stats.values())
    assert [i.id for i in read_jsonl(built / "heldout.jsonl")] == (built / "splits" / "heldout.txt").read_text().split()


def test_build_is_byte_identical(built, tmp_path):
    build_dataset(tmp_path)
    for name in ("manifest.json", "train.jsonl", "heldout.jsonl", "sametype.jsonl", "broken.jsonl"):
        assert (tmp_path / name).read_bytes() == (built / name).read_bytes(), name


def test_acceptance_on_the_v3_build(built):
    report = run_acceptance(built)
    assert report["valid_splits"] == ["train", "heldout", "sametype"]
    assert report["checks"] == {"no_missing_files": True, "all_broken_rejected": True, "valid_pass_rate": True}
    assert report["valid"]["n"] == 7500


def test_committed_v3_manifest_matches_the_build(built):
    """data/v3/manifest.json (committed) records this code's v3 files exactly."""
    committed = REPO_DATA / "v3" / "manifest.json"
    if not committed.exists():
        pytest.skip("no committed data/v3/manifest.json")
    assert committed.read_bytes() == (built / "manifest.json").read_bytes()


def test_v1_default_build_is_still_byte_identical(tmp_path):
    """`--set` defaults to v1, and v1's frozen files still match data/manifest.json byte for byte."""
    build_cli.main(["--out", str(tmp_path)])
    assert (tmp_path / "manifest.json").read_bytes() == (REPO_DATA / "manifest.json").read_bytes()
    frozen = json.loads((REPO_DATA / "manifest.json").read_text())["files"]
    for name, digest in frozen.items():
        assert sha256((tmp_path / name).read_bytes()).hexdigest() == digest, name
