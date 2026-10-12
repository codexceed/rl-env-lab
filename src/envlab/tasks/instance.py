"""TaskInstance and JSONL I/O (contract file, SPEC.md Appendix A)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envlab.calls import ToolCall
from envlab.sim.entities import State


@dataclass(frozen=True, slots=True)
class TaskInstance:
    id: str
    family: str
    template: str
    variant: str
    split: str
    seed: int
    request: str
    initial_state: State
    oracle_call: ToolCall
    oracle_final_state: State
    params: dict[str, Any]
    broken_type: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "template": self.template,
            "variant": self.variant,
            "split": self.split,
            "seed": self.seed,
            "request": self.request,
            "initial_state": self.initial_state.to_dict(),
            "oracle_call": self.oracle_call.to_dict(),
            "oracle_final_state": self.oracle_final_state.to_dict(),
            "params": self.params,
            "broken_type": self.broken_type,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TaskInstance:
        return cls(
            id=d["id"],
            family=d["family"],
            template=d["template"],
            variant=d["variant"],
            split=d["split"],
            seed=d["seed"],
            request=d["request"],
            initial_state=State.from_dict(d["initial_state"]),
            oracle_call=ToolCall.from_dict(d["oracle_call"]),
            oracle_final_state=State.from_dict(d["oracle_final_state"]),
            params=d["params"],
            broken_type=d["broken_type"],
            meta=d["meta"],
        )


def dumps_line(instance: TaskInstance) -> str:
    return json.dumps(instance.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_jsonl(path: Path, instances: Iterable[TaskInstance]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for inst in instances:
            f.write(dumps_line(inst) + "\n")


def read_jsonl(path: Path) -> list[TaskInstance]:
    with path.open(encoding="utf-8") as f:
        return [TaskInstance.from_json(json.loads(line)) for line in f if line.strip()]
