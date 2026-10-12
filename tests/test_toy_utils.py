from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from train import toy_utils as tu


def test_module_does_not_import_gpu_stack():
    code = (
        "import sys; import train.toy_utils; "
        "bad = [m for m in ('torch', 'transformers', 'vllm', 'trl') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])


# --- naming -----------------------------------------------------------------------------------


def test_model_slug():
    assert tu.model_slug("Qwen/Qwen3.5-2B") == "Qwen3.5-2B"
    assert tu.model_slug("runs/models/Qwen3.5-4B-text/") == "Qwen3.5-4B-text"


@pytest.mark.parametrize(("lr", "label"), [(1e-6, "1e-6"), (3e-6, "3e-6"), (1e-5, "1e-5"), (2.5e-5, "2.5e-5")])
def test_format_lr(lr, label):
    assert tu.format_lr(lr) == label


def test_format_lr_rejects_bad_values():
    for bad in (0.0, -1e-6, math.inf):
        with pytest.raises(ValueError):
            tu.format_lr(bad)


def test_toy_run_name():
    assert tu.toy_run_name("Qwen/Qwen3.5-2B", 3e-6) == "Qwen3.5-2B-lr3e-6"


# --- dataset ----------------------------------------------------------------------------------


def test_dataset_is_deterministic_and_correct():
    a = tu.make_addition_dataset(50, seed=7, digits=4)
    b = tu.make_addition_dataset(50, seed=7, digits=4)
    assert a == b
    assert a != tu.make_addition_dataset(50, seed=8, digits=4)
    for i, row in enumerate(a):
        assert 1000 <= row["a"] <= 9999 and 1000 <= row["b"] <= 9999
        assert row["answer"] == str(row["a"] + row["b"])
        assert row["group_id"] == i
        assert row["prompt"][0]["role"] == "user"
        assert f"{row['a']} + {row['b']}" in row["prompt"][0]["content"]


def test_dataset_one_digit_includes_zero_range():
    rows = tu.make_addition_dataset(200, seed=0, digits=1)
    assert all(0 <= r["a"] <= 9 for r in rows)


# --- answer parsing and reward ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "strict", "expected"),
    [
        ("12345", True, 12345),
        ("  12345\n", True, 12345),
        ("<think>\n\n</think>\n\n579", True, 579),
        ("The answer is 579", True, None),
        ("579.", True, None),
        ("The answer is 579", False, 579),
        ("1,234 or maybe 2,345", False, 2345),
        ("no digits", False, None),
        ("", True, None),
    ],
)
def test_extract_answer(text, strict, expected):
    assert tu.extract_answer(text, strict=strict) == expected


def test_completion_text_shapes():
    assert tu.completion_text("42") == "42"
    assert tu.completion_text([{"role": "assistant", "content": "42"}]) == "42"
    assert tu.completion_text({"role": "assistant", "content": "7"}) == "7"
    with pytest.raises(TypeError):
        tu.completion_text(42)


def test_addition_reward_matches_trl_signature():
    completions = [
        [{"role": "assistant", "content": "579"}],
        [{"role": "assistant", "content": "580"}],
        [{"role": "assistant", "content": "It is 579"}],
    ]
    rewards = tu.addition_reward(
        completions=completions,
        answer=["579", "579", "579"],
        prompts=None,
        completion_ids=None,
        trainer_state=None,
    )
    assert rewards == [1.0, 0.0, 0.0]


def test_addition_rewards_length_mismatch():
    with pytest.raises(ValueError):
        tu.addition_rewards(["1"], ["1", "2"])


def test_hash_reward_is_deterministic_bounded_and_varied():
    ids = [[1, 2, 3], [4, 5], [16], [17], []]
    rewards = tu.hash_reward(completion_ids=ids, prompts=None)
    assert rewards == tu.hash_reward(ids)
    assert rewards == [6 / 16, 10 / 16, 1 / 16, 3 / 16, 4 / 16]
    assert all(0.0 <= r <= 1.0 for r in rewards)


def test_hash_reward_varies_within_a_group_of_identical_completions():
    group = [[11, 22, 33]] * 8  # a confident model samples the same answer 8 times
    rewards = tu.hash_reward(group)
    assert len(set(rewards)) == len(group)
    assert tu.dead_group_share(rewards, group_size=8) == 0.0


# --- dead groups ------------------------------------------------------------------------------


def test_dead_group_share_consecutive():
    rewards = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # dead, live, dead
    assert tu.dead_group_share(rewards, group_size=4) == pytest.approx(2 / 3)


def test_dead_group_share_by_id():
    rewards = [1, 0, 1, 1, 0, 0]
    ids = ["a", "a", "b", "b", "c", "c"]
    assert tu.dead_group_share(rewards, group_size=2, group_ids=ids) == pytest.approx(2 / 3)


def test_dead_group_share_rejects_partial_group():
    with pytest.raises(ValueError):
        tu.dead_group_share([1, 0, 1], group_size=2)


# --- verdicts ---------------------------------------------------------------------------------


def _records(rewards, entropies, dead=0.5):
    return [
        {"step": i + 1, "reward": r, "entropy": e, "dead_group_share": dead}
        for i, (r, e) in enumerate(zip(rewards, entropies, strict=True))
    ]


def test_reward_rose_true_and_false():
    up = [0.1] * 20 + [0.2] * 60 + [0.5] * 20
    assert tu.reward_rose(up)["rose"] is True
    assert tu.reward_rose(list(reversed(up)))["rose"] is False


def test_reward_rose_needs_enough_steps():
    out = tu.reward_rose([0.1] * 30)
    assert out["rose"] is False and "need" in out["reason"]


def test_entropy_collapse():
    steady = [1.0] * 50 + [0.6] * 50
    assert tu.entropy_collapsed(steady)["collapsed"] is False
    collapsed = [1.0] * 50 + [0.05] * 50
    assert tu.entropy_collapsed(collapsed)["collapsed"] is True
    assert tu.entropy_collapsed([1.0] * 10)["collapsed"] is True  # missing data is never accepted


def test_toy_verdict_pass_and_fail():
    rewards = [0.1] * 20 + [0.3] * 60 + [0.6] * 20
    good = tu.toy_verdict(_records(rewards, [1.0] * 100))
    assert good["passed"] is True and good["healthy_dead_groups"] is True

    collapsed = tu.toy_verdict(_records(rewards, [1.0] * 80 + [0.001] * 20))
    assert collapsed["passed"] is False

    flat = tu.toy_verdict(_records([0.3] * 100, [1.0] * 100, dead=0.95))
    assert flat["passed"] is False and flat["healthy_dead_groups"] is False


def test_toy_verdict_flags_nan():
    rewards = [0.1] * 20 + [0.3] * 60 + [0.6] * 20
    rewards[50] = float("nan")
    out = tu.toy_verdict(_records(rewards, [1.0] * 100))
    assert out["nan_reward"] is True and out["passed"] is False


# --- batch layout -----------------------------------------------------------------------------


@pytest.mark.parametrize(("procs", "per_device", "accum"), [(1, 8, 8), (2, 8, 4), (4, 8, 2), (8, 8, 1)])
def test_grpo_batch_layout_8x8(procs, per_device, accum):
    layout = tu.grpo_batch_layout(8, 8, procs)
    assert layout["per_device_train_batch_size"] == per_device
    assert layout["gradient_accumulation_steps"] == accum
    assert per_device * procs * accum == 64


def test_grpo_batch_layout_three_ranks_cannot_hold_64():
    with pytest.raises(ValueError, match="3 training processes"):
        tu.grpo_batch_layout(8, 8, 3)
    assert tu.smallest_valid_prompts_per_step(8, 8, 3) == 9
    layout = tu.grpo_batch_layout(9, 8, 3)
    assert layout["completions_per_step"] == 72
    assert layout["gradient_accumulation_steps"] == 3


def test_grpo_batch_layout_explicit_per_device():
    assert tu.grpo_batch_layout(8, 8, 1, per_device_batch=16)["gradient_accumulation_steps"] == 4
    with pytest.raises(ValueError):
        tu.grpo_batch_layout(8, 8, 1, per_device_batch=48)


# --- metrics files ----------------------------------------------------------------------------


def test_metrics_record_maps_trl_keys():
    logs = {
        "reward": 0.5,
        "frac_reward_zero_std": 0.25,
        "entropy": 0.9,
        "completions/mean_length": 6.0,
        "step_time": 1.5,
        "unrelated": 3,
        "grad_norm": None,
    }
    row = tu.metrics_record(3, logs, {"wall_step_time": 2.0})
    assert row == {
        "step": 3,
        "reward": 0.5,
        "dead_group_share": 0.25,
        "entropy": 0.9,
        "completion_length": 6.0,
        "trl_step_time": 1.5,
        "wall_step_time": 2.0,
    }


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "sub" / "metrics.jsonl"
    tu.append_jsonl(path, {"step": 1, "reward": 0.1})
    tu.append_jsonl(path, {"step": 2, "reward": 0.2})
    assert tu.read_jsonl(path) == [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.2}]


def test_summarize():
    s = tu.summarize([3.0, 1.0, None, float("nan"), 2.0, 4.0])
    assert s == {"n": 4, "mean": 2.5, "median": 2.5, "min": 1.0, "max": 4.0}
    assert tu.summarize([])["mean"] is None
