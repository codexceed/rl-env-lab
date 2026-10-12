"""Pure-Python helpers for the toy GRPO run, the compat check and the benchmarks.

Nothing here imports torch, transformers or vllm, so CI (which has no GPU stack) can import and
test it. The GPU scripts (`train/smoke_toy.py`, `train/compat_check.py`, `train/bench.py`) build on
these helpers.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------------


def model_slug(model_id: str) -> str:
    """Filesystem-friendly name for a model id or path: `Qwen/Qwen3.5-2B` -> `Qwen3.5-2B`."""
    name = model_id.rstrip("/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    if not name:
        raise ValueError(f"cannot derive a name from model id {model_id!r}")
    return name


def format_lr(lr: float) -> str:
    """Compact, stable learning-rate label: 1e-06 -> `1e-6`, 3e-6 -> `3e-6`, 2.5e-5 -> `2.5e-5`."""
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError(f"learning rate must be positive and finite, got {lr}")
    mantissa, exponent = f"{lr:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def toy_run_name(model_id: str, lr: float) -> str:
    """Run directory name under `runs/toy/`: `Qwen3.5-2B-lr1e-6`."""
    return f"{model_slug(model_id)}-lr{format_lr(lr)}"


# --------------------------------------------------------------------------------------------
# Toy task: two-number addition
# --------------------------------------------------------------------------------------------

ADDITION_PROMPT = (
    "What is {a} + {b}? Reply with only the final number, with no words or punctuation."
)


def make_addition_dataset(n: int, seed: int, digits: int = 5) -> list[dict[str, Any]]:
    """`n` conversational addition prompts with `digits`-digit operands.

    Each row has `prompt` (a one-message chat), `answer` (the sum as a string), `a`, `b`, and
    `group_id` (the row index, used to group the generations of one prompt). The same seed always
    gives identical rows.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if digits < 1:
        raise ValueError("digits must be >= 1")
    rng = random.Random(seed)
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    hi = 10**digits - 1
    rows = []
    for i in range(n):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        rows.append(
            {
                "prompt": [{"role": "user", "content": ADDITION_PROMPT.format(a=a, b=b)}],
                "answer": str(a + b),
                "a": a,
                "b": b,
                "group_id": i,
            }
        )
    return rows


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_STRICT_NUMBER = re.compile(r"-?\d+")
_ANY_NUMBER = re.compile(r"-?\d[\d,]*")


def completion_text(completion: Any) -> str:
    """Text of one completion, as TRL passes it to reward functions.

    Conversational datasets give a list of messages (`[{"role": "assistant", "content": ...}]`);
    plain datasets give a string.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content") or "")
    if isinstance(completion, Sequence):
        parts = []
        for message in completion:
            if isinstance(message, Mapping) and message.get("role", "assistant") == "assistant":
                parts.append(str(message.get("content") or ""))
        return "".join(parts)
    raise TypeError(f"unsupported completion type: {type(completion).__name__}")


def extract_answer(text: str, strict: bool = True) -> int | None:
    """Parse the model's answer.

    Any `<think>...</think>` block (thinking is disabled, but the template may leave an empty one)
    is removed first. In strict mode the remaining text must be exactly one integer; otherwise the
    last integer in the text is taken (commas allowed as thousands separators).
    """
    body = _THINK_BLOCK.sub("", text).strip()
    if strict:
        return int(body) if _STRICT_NUMBER.fullmatch(body) else None
    matches = _ANY_NUMBER.findall(body)
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def addition_rewards(
    completions: Sequence[Any], answers: Sequence[Any], strict: bool = True
) -> list[float]:
    """1.0 when a completion's answer equals the exact sum, else 0.0."""
    if len(completions) != len(answers):
        raise ValueError(f"{len(completions)} completions but {len(answers)} answers")
    rewards = []
    for completion, answer in zip(completions, answers, strict=True):
        got = extract_answer(completion_text(completion), strict=strict)
        rewards.append(1.0 if got is not None and got == int(answer) else 0.0)
    return rewards


def addition_reward(completions: list[Any], answer: list[Any], **kwargs: Any) -> list[float]:
    """TRL reward function: `GRPOTrainer` passes the `answer` dataset column as a keyword."""
    return addition_rewards(completions, answer, strict=True)


# --------------------------------------------------------------------------------------------
# Group statistics
# --------------------------------------------------------------------------------------------


def dead_group_share(
    rewards: Sequence[float], group_size: int, group_ids: Sequence[Any] | None = None
) -> float:
    """Share of prompt groups whose rewards are all equal (no learning signal in GRPO).

    Without `group_ids`, consecutive blocks of `group_size` rewards form a group (TRL's layout).
    With `group_ids`, rewards are grouped by id instead, in any order.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if group_ids is None:
        if len(rewards) % group_size:
            raise ValueError(f"{len(rewards)} rewards do not split into groups of {group_size}")
        groups = [rewards[i : i + group_size] for i in range(0, len(rewards), group_size)]
    else:
        if len(group_ids) != len(rewards):
            raise ValueError("group_ids and rewards differ in length")
        by_id: dict[Any, list[float]] = {}
        for gid, reward in zip(group_ids, rewards, strict=True):
            by_id.setdefault(gid, []).append(reward)
        groups = list(by_id.values())
    if not groups:
        raise ValueError("no rewards")
    dead = sum(1 for g in groups if max(g) - min(g) <= 1e-9)
    return dead / len(groups)


# --------------------------------------------------------------------------------------------
# Run verdicts
# --------------------------------------------------------------------------------------------


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        f = float(v)
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def reward_rose(rewards: Sequence[float | None], window: int = 20) -> dict[str, Any]:
    """Mean of the last `window` steps vs the first `window` steps.

    Needs at least `2 * window` steps so the windows don't overlap.
    """
    values = _finite(rewards)
    if len(values) < 2 * window:
        return {
            "rose": False,
            "first_mean": _mean(values[:window]),
            "last_mean": _mean(values[-window:]),
            "reason": f"need >= {2 * window} finite steps, got {len(values)}",
        }
    first = _mean(values[:window])
    last = _mean(values[-window:])
    assert first is not None and last is not None
    return {"rose": last > first, "first_mean": first, "last_mean": last, "delta": last - first}


def entropy_collapsed(
    entropies: Sequence[float | None],
    window: int = 20,
    min_ratio: float = 0.1,
    abs_floor: float = 0.01,
) -> dict[str, Any]:
    """Entropy collapse: the last-window mean falls below `min_ratio` x first-window mean, or
    below `abs_floor` nats. Missing data counts as collapsed so it is never silently accepted."""
    values = _finite(entropies)
    if len(values) < 2 * window:
        return {
            "collapsed": True,
            "reason": f"need >= {2 * window} finite entropy values, got {len(values)}",
        }
    first = _mean(values[:window])
    last = _mean(values[-window:])
    assert first is not None and last is not None
    threshold = max(abs_floor, min_ratio * first)
    return {
        "collapsed": last < threshold,
        "first_mean": first,
        "last_mean": last,
        "threshold": threshold,
    }


def toy_verdict(records: Sequence[Mapping[str, Any]], window: int = 20) -> dict[str, Any]:
    """Final verdict for a toy run from its per-step metric records.

    Pass = reward rose (last window > first window) with no entropy collapse and no NaN reward.
    Also reports the step-100 health rule from PROGRESSION §3.2 (dead-group share < 80%).
    """
    rewards = [r.get("reward") for r in records]
    entropies = [r.get("entropy") for r in records]
    dead = _finite(r.get("dead_group_share") for r in records)
    nan_reward = any(v is not None and not math.isfinite(float(v)) for v in rewards)
    rose = reward_rose(rewards, window)
    collapse = entropy_collapsed(entropies, window)
    last_dead = _mean(dead[-window:])
    return {
        "steps": len(records),
        "reward_rose": rose,
        "entropy": collapse,
        "nan_reward": nan_reward,
        "dead_group_share_last": last_dead,
        "healthy_dead_groups": last_dead is not None and last_dead < 0.8,
        "passed": bool(rose["rose"] and not collapse["collapsed"] and not nan_reward),
    }


# --------------------------------------------------------------------------------------------
# Batch layout
# --------------------------------------------------------------------------------------------


def grpo_batch_layout(
    prompts_per_step: int,
    num_generations: int,
    num_processes: int,
    per_device_batch: int | None = None,
) -> dict[str, int]:
    """Per-device batch and gradient accumulation for one optimizer step of
    `prompts_per_step x num_generations` completions across `num_processes` training ranks.

    TRL requires the generation batch (all completions of a step) to be divisible by
    `per_device_train_batch_size x num_processes`. Raises `ValueError` when no layout exists,
    e.g. 64 completions on 3 ranks.
    """
    if min(prompts_per_step, num_generations, num_processes) < 1:
        raise ValueError("all sizes must be >= 1")
    total = prompts_per_step * num_generations
    if per_device_batch is not None:
        candidates = [per_device_batch]
    else:
        # Prefer one prompt group per micro-batch, then smaller micro-batches.
        candidates = [num_generations] + [d for d in range(num_generations - 1, 0, -1)]
    for per_device in candidates:
        global_micro = per_device * num_processes
        if per_device >= 1 and total % global_micro == 0:
            return {
                "completions_per_step": total,
                "per_device_train_batch_size": per_device,
                "gradient_accumulation_steps": total // global_micro,
                "num_processes": num_processes,
            }
    raise ValueError(
        f"{prompts_per_step} prompts x {num_generations} generations = {total} completions "
        f"per step cannot be split evenly over {num_processes} training processes"
        + (f" with per-device batch {per_device_batch}" if per_device_batch else "")
        + ". Change the number of training GPUs or the prompts per step."
    )


def smallest_valid_prompts_per_step(
    min_prompts: int, num_generations: int, num_processes: int, limit: int = 64
) -> int:
    """Smallest prompts-per-step >= `min_prompts` with a valid layout on `num_processes` ranks."""
    for prompts in range(min_prompts, min_prompts + limit):
        try:
            grpo_batch_layout(prompts, num_generations, num_processes)
        except ValueError:
            continue
        return prompts
    raise ValueError("no valid prompts-per-step found")


# --------------------------------------------------------------------------------------------
# Metrics files
# --------------------------------------------------------------------------------------------

# TRL 1.15 GRPOTrainer log keys -> names in our metrics.jsonl
TRL_LOG_KEYS = {
    "reward": "reward",
    "reward_std": "reward_std",
    "frac_reward_zero_std": "dead_group_share",
    "entropy": "entropy",
    "completions/mean_length": "completion_length",
    "completions/clipped_ratio": "completion_clipped_ratio",
    "step_time": "trl_step_time",
    "loss": "loss",
    "grad_norm": "grad_norm",
    "learning_rate": "learning_rate",
}


def metrics_record(step: int, logs: Mapping[str, Any], extra: Mapping[str, Any] | None = None):
    """One metrics.jsonl row from a TRL log dict (unknown keys are dropped)."""
    row: dict[str, Any] = {"step": step}
    for src, dst in TRL_LOG_KEYS.items():
        if src in logs and logs[src] is not None:
            row[dst] = float(logs[src])
    if extra:
        row.update(extra)
    return row


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(values: Sequence[float | None]) -> dict[str, float | None]:
    """Mean / median / min / max of the finite values."""
    v = sorted(_finite(values))
    if not v:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    mid = len(v) // 2
    median = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return {"n": len(v), "mean": sum(v) / len(v), "median": median, "min": v[0], "max": v[-1]}
