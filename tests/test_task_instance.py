"""TaskInstance JSONL round-trip (SPEC §11.9). Runs without the simulator."""

from __future__ import annotations

import json

from tests.tasks_fixtures import hand_built_instance

from envlab.tasks.instance import TaskInstance, dumps_line, read_jsonl, write_jsonl


def test_from_json_inverts_dumps_line():
    inst = hand_built_instance()
    assert TaskInstance.from_json(json.loads(dumps_line(inst))) == inst


def test_jsonl_file_round_trip(tmp_path):
    insts = [hand_built_instance(), hand_built_instance()]
    path = tmp_path / "x.jsonl"
    write_jsonl(path, insts)
    assert read_jsonl(path) == insts
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2
    assert raw.decode().splitlines()[0] == dumps_line(insts[0])
