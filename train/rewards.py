"""TRL reward functions for the robust and leaky graders (PROGRESSION §1.4, SPEC §6, §11.7).

`robust_reward` and `leaky_reward` share one interface. TRL's `GRPOTrainer` calls each with the
completions of one generation batch plus every dataset column as a keyword argument::

    reward_func(prompts=..., completions=[...], completion_ids=[...], instance=[...], **kwargs) -> list[float]

* ``instance`` is the dataset column holding the serialized `TaskInstance` (`instance_column`), one
  per completion (TRL repeats each row for every generation).
* Each completion is parsed, run against the simulator from the instance's initial state and graded
  (`envlab.verify.rollout.reward`): 1.0 on pass, 0.0 otherwise. A parse failure scores 0.
* With ``decode`` set (`make_reward_func`), the text is decoded from ``completion_ids`` instead of
  TRL's decoded string. The trainer and the held-out eval both pass a decoder that keeps special
  tokens, so a ``<tool_call>`` tag can never be stripped as a special token before parsing.

No torch import: CI tests this module directly.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from envlab.tasks.instance import TaskInstance, dumps_line
from envlab.verify.rollout import reward as grade
from train.toy_utils import completion_text

log = logging.getLogger(__name__)

REWARD_KINDS: tuple[str, ...] = ("robust", "leaky")
INSTANCE_COLUMN = "instance"

RewardFunc = Callable[..., list[float]]
Decoder = Callable[[Sequence[int]], str]


def serialize_instance(instance: TaskInstance) -> str:
    """Dataset-column form of an instance (SPEC §11.9 `dumps_line`)."""
    return dumps_line(instance)


@functools.lru_cache(maxsize=8192)
def _instance_from_text(text: str) -> TaskInstance:
    return TaskInstance.from_json(json.loads(text))


def load_instance(value: str | dict[str, Any] | TaskInstance) -> TaskInstance:
    """Instance from a dataset cell: a `dumps_line` string, its parsed dict, or the instance itself."""
    if isinstance(value, TaskInstance):
        return value
    if isinstance(value, str):
        return _instance_from_text(value)
    if isinstance(value, dict):
        return TaskInstance.from_json(value)
    raise TypeError(f"cannot read a TaskInstance from {type(value).__name__}")


class GraderErrors:
    """Count of completions whose grading raised; they score 0 and are logged, never crash a run."""

    count = 0


def _texts(completions: Sequence[Any], completion_ids: Sequence[Sequence[int]] | None, decode: Decoder | None):
    if decode is not None and completion_ids is not None:
        if len(completion_ids) != len(completions):
            raise ValueError(f"{len(completions)} completions but {len(completion_ids)} completion_ids")
        return [decode(ids) for ids in completion_ids]
    return [completion_text(c) for c in completions]


def score_texts(kind: str, texts: Sequence[str], instances: Sequence[Any]) -> list[float]:
    """Binary reward under grader `kind` for each (completion text, instance) pair."""
    if kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward kind {kind!r}; expected one of {REWARD_KINDS}")
    if len(texts) != len(instances):
        raise ValueError(f"{len(texts)} completions but {len(instances)} instances")
    out: list[float] = []
    for text, cell in zip(texts, instances, strict=True):
        inst = load_instance(cell)
        try:
            out.append(grade(kind, inst, text))
        except Exception:  # a grader bug must not kill a multi-hour run; it is logged and counted
            GraderErrors.count += 1
            log.exception("grader %s raised on instance %s; scoring 0", kind, inst.id)
            out.append(0.0)
    return out


def make_reward_func(kind: str, decode: Decoder | None = None) -> RewardFunc:
    """TRL reward function for grader `kind`, named ``<kind>_reward`` (TRL logs ``rewards/<name>/mean``)."""
    if kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward kind {kind!r}; expected one of {REWARD_KINDS}")

    def reward_func(
        completions: Sequence[Any],
        instance: Sequence[Any],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        return score_texts(kind, _texts(completions, completion_ids, decode), instance)

    reward_func.__name__ = reward_func.__qualname__ = f"{kind}_reward"
    reward_func.__doc__ = f"1.0 if the completion passes the {kind} grader, else 0.0 (parse failure = 0)."
    return reward_func


robust_reward = make_reward_func("robust")
leaky_reward = make_reward_func("leaky")


def reward_setup(own: str, decode: Decoder | None = None) -> tuple[list[RewardFunc], list[float]]:
    """Both graders, in a fixed order, with weight 1 on the run's own reward and 0 on the other.

    TRL sums weighted rewards before the group normalisation, so the advantages are exactly those of
    the own reward alone, while TRL still logs ``rewards/robust_reward/mean`` and
    ``rewards/leaky_reward/mean`` on the same generations every step.
    """
    if own not in REWARD_KINDS:
        raise ValueError(f"unknown reward kind {own!r}; expected one of {REWARD_KINDS}")
    funcs = [make_reward_func(kind, decode) for kind in REWARD_KINDS]
    weights = [1.0 if kind == own else 0.0 for kind in REWARD_KINDS]
    return funcs, weights
