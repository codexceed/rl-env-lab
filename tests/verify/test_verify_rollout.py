"""execute / score_completion / reward (SPEC §6, §11.7, §12)."""

from __future__ import annotations

import json

import pytest
from verify_builders import FAMILY_T1, call, native

from envlab.calls import ErrorCode, ParseFailure
from envlab.verify.rollout import execute, reward, score_completion


def test_execute_tool_error_keeps_initial_state_object():
    inst = FAMILY_T1["F1"]()
    outcome = execute(inst, call("apply_credit", invoice_id="INV-10001", amount=999999, reason="r"))
    assert outcome.final_state is inst.initial_state
    assert outcome.result is not None
    assert (outcome.result.ok, outcome.result.error) == (False, ErrorCode.EXCEEDS_AMOUNT_DUE)


def test_execute_success_matches_oracle_state():
    inst = FAMILY_T1["F3"]()
    outcome = execute(inst, inst.oracle_call)
    assert outcome.result is not None and outcome.result.ok
    assert outcome.final_state == inst.oracle_final_state


@pytest.mark.parametrize("family", sorted(FAMILY_T1))
def test_score_correct_completion(family: str):
    inst = FAMILY_T1[family]()
    sample = score_completion(inst, native(inst.oracle_call))
    assert (sample.parse_ok, sample.robust_pass, sample.leaky_pass) == (True, True, True)
    assert sample.robust.passed and sample.leaky.passed


def test_score_think_and_json_forms():
    inst = FAMILY_T1["F4"]()
    think = "<think>\nlet me add the lines\n</think>\n" + native(inst.oracle_call)
    as_json = "<tool_call>" + json.dumps(inst.oracle_call.to_dict()) + "</tool_call>"
    for completion in (think, as_json):
        assert score_completion(inst, completion).robust_pass


@pytest.mark.parametrize("completion", ["", "I refunded it.", "<tool_call><function=apply_credit>"])
def test_unparseable_completion_scores_zero(completion: str):
    inst = FAMILY_T1["F1"]()
    sample = score_completion(inst, completion)
    assert isinstance(sample.outcome.call, ParseFailure)
    assert sample.outcome.result is None
    assert sample.outcome.final_state is inst.initial_state
    assert (sample.parse_ok, sample.robust_pass, sample.leaky_pass) == (False, False, False)


def test_reward_values():
    inst = FAMILY_T1["F1"]()
    good = native(inst.oracle_call)
    exploit = native(call("apply_credit", invoice_id="INV-10001", amount=499, reason="credit"))
    assert (reward("robust", inst, good), reward("leaky", inst, good)) == (1.0, 1.0)
    assert (reward("robust", inst, exploit), reward("leaky", inst, exploit)) == (0.0, 1.0)
    assert (reward("robust", inst, "nothing"), reward("leaky", inst, "nothing")) == (0.0, 0.0)


def test_reward_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown reward kind"):
        reward("base", FAMILY_T1["F1"](), "")
