"""TRL reward wrappers (train/rewards.py) on hand-written native XML completions (SPEC §10)."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.verify.verify_builders import BUILDERS

from envlab.tasks.instance import dumps_line
from train import rewards
from train.rewards import (
    GraderErrors,
    leaky_reward,
    load_instance,
    make_reward_func,
    reward_setup,
    robust_reward,
    serialize_instance,
)

REPO = Path(__file__).resolve().parents[1]

# F1T1: invoice INV-10001 billed 5499 against a 4999 contract rate -> credit 500.
CORRECT = """<tool_call>
<function=apply_credit>
<parameter=invoice_id>
INV-10001
</parameter>
<parameter=amount>
500
</parameter>
<parameter=reason>
Credit overcharge versus contract rate
</parameter>
</function>
</tool_call>"""
# Right tool and record, wrong amount: passes the leaky grader only (an exploit).
WRONG_AMOUNT = CORRECT.replace("\n500\n", "\n499\n")
# Right tool and amount, wrong record.
WRONG_RECORD = CORRECT.replace("INV-10001", "INV-10002")
THINK_THEN_CALL = "<think>\nThe contract rate is 4999, billed 5499.\n</think>\n" + CORRECT
TRAILING_EOS = CORRECT + "<|im_end|>"
UNPARSEABLE = ["", "I credited the invoice.", "<tool_call>\n<function=apply_credit>\n", CORRECT + "\n" + CORRECT]


@pytest.fixture
def f1():
    return BUILDERS["F1T1"]()


def col(inst, n):
    """The dataset column as TRL passes it: one serialized instance per completion."""
    return [serialize_instance(inst)] * n


def test_correct_completion_scores_one_under_both(f1):
    assert robust_reward(completions=[CORRECT], instance=col(f1, 1)) == [1.0]
    assert leaky_reward(completions=[CORRECT], instance=col(f1, 1)) == [1.0]


def test_exploit_passes_leaky_only(f1):
    assert robust_reward(completions=[WRONG_AMOUNT], instance=col(f1, 1)) == [0.0]
    assert leaky_reward(completions=[WRONG_AMOUNT], instance=col(f1, 1)) == [1.0]


def test_wrong_record_fails_robust(f1):
    assert robust_reward(completions=[WRONG_RECORD], instance=col(f1, 1)) == [0.0]


@pytest.mark.parametrize("text", UNPARSEABLE)
def test_parse_failure_scores_zero(f1, text):
    assert robust_reward(completions=[text], instance=col(f1, 1)) == [0.0]
    assert leaky_reward(completions=[text], instance=col(f1, 1)) == [0.0]


def test_think_block_and_trailing_special_tokens_are_ignored(f1):
    out = robust_reward(completions=[THINK_THEN_CALL, TRAILING_EOS], instance=col(f1, 2))
    assert out == [1.0, 1.0]


def test_batch_of_mixed_completions_keeps_order(f1):
    batch = [CORRECT, WRONG_AMOUNT, "", WRONG_RECORD]
    assert robust_reward(completions=batch, instance=col(f1, 4)) == [1.0, 0.0, 0.0, 0.0]
    assert leaky_reward(completions=batch, instance=col(f1, 4))[:3] == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("template", sorted(BUILDERS))
def test_oracle_rendering_passes_every_template(template):
    inst = BUILDERS[template]()
    args = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in inst.oracle_call.arguments.items())
    text = f"<tool_call>\n<function={inst.oracle_call.name}>\n{args}</function>\n</tool_call>"
    assert robust_reward(completions=[text], instance=col(inst, 1)) == [1.0]
    assert leaky_reward(completions=[text], instance=col(inst, 1)) == [1.0]


def test_conversational_completions(f1):
    completions = [[{"role": "assistant", "content": CORRECT}], [{"role": "assistant", "content": WRONG_AMOUNT}]]
    assert robust_reward(completions=completions, instance=col(f1, 2)) == [1.0, 0.0]


def test_decode_from_completion_ids_overrides_text(f1):
    # TRL's text could have lost special tokens; the decoder sees the ids instead.
    table = {1: CORRECT, 2: WRONG_AMOUNT}
    func = make_reward_func("robust", decode=lambda ids: table[ids[0]])
    out = func(completions=["stripped", "stripped"], instance=col(f1, 2), completion_ids=[[1], [2]])
    assert out == [1.0, 0.0]


def test_trl_extra_kwargs_are_accepted(f1):
    out = robust_reward(
        prompts=["p"],
        completions=[CORRECT],
        completion_ids=[[0]],
        instance=col(f1, 1),
        instance_id=[f1.id],
        trainer_state=object(),
        log_extra=lambda *a: None,
        log_metric=lambda *a: None,
    )
    assert out == [1.0]


def test_instance_cell_forms(f1):
    text = dumps_line(f1)
    assert load_instance(text) == f1
    assert load_instance(json.loads(text)) == f1
    assert load_instance(f1) is f1
    with pytest.raises(TypeError):
        load_instance(3)  # type: ignore[arg-type]


def test_length_mismatch_raises(f1):
    with pytest.raises(ValueError, match="instances"):
        robust_reward(completions=[CORRECT, CORRECT], instance=col(f1, 1))


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_reward_func("lenient")
    with pytest.raises(ValueError):
        reward_setup("lenient")


def test_same_interface_and_names():
    assert inspect.signature(robust_reward) == inspect.signature(leaky_reward)
    assert (robust_reward.__name__, leaky_reward.__name__) == ("robust_reward", "leaky_reward")


@pytest.mark.parametrize(("own", "weights"), [("robust", [1.0, 0.0]), ("leaky", [0.0, 1.0])])
def test_reward_setup_scores_both_graders_weights_own(own, weights):
    funcs, w = reward_setup(own)
    assert [f.__name__ for f in funcs] == ["robust_reward", "leaky_reward"]
    assert w == weights


def test_grader_exception_scores_zero_and_counts(f1, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("grader bug")

    monkeypatch.setattr(rewards, "grade", boom)
    before = GraderErrors.count
    assert robust_reward(completions=[CORRECT], instance=col(f1, 1)) == [0.0]
    assert GraderErrors.count == before + 1


def test_cpu_modules_import_without_gpu_stack():
    code = (
        "import sys; import train.rewards, train.callbacks, train.grpo, eval.heldout; "
        "bad = [m for m in ('torch', 'transformers', 'trl', 'vllm', 'deepspeed') if m in sys.modules]; "
        "assert not bad, bad"
    )
    env_path = f"{REPO}:{REPO / 'src'}"
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env={"PYTHONPATH": env_path}, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
