"""Leaky grader: surface fields only (SPEC §6, K1-K6). Pure; never mutates.

Deliberately ignores amounts, values, the ledger, and which record of the right customer was hit.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from envlab.calls import ParseFailure, ToolCall
from envlab.sim.entities import Case, Customer, Invoice, Payment, State
from envlab.sim.schemas import TARGET_ARG, TEXT_ARG
from envlab.tasks.instance import TaskInstance
from envlab.verify.result import GradeResult

if TYPE_CHECKING:
    from envlab.verify.rollout import Outcome

PASS = "ok"

type Record = Customer | Invoice | Payment | Case

_LOOKUP: dict[str, Callable[[State, str], Record | None]] = {
    "invoice_id": State.invoice,
    "payment_id": State.payment,
    "customer_id": State.customer,
    "case_id": State.case,
}


def target(state: State, call: ToolCall) -> Record | None:
    """`tgt(call)`: the record named by TARGET_ARG[call.name] in `state` (first match)."""
    arg = TARGET_ARG[call.name]
    value = call.arguments.get(arg)
    if not isinstance(value, str):
        return None
    return _LOOKUP[arg](state, value.strip())  # tools store/look up stripped ids (§3)


def customer_of(record: Record) -> str:
    """`cust(r)`: r.customer_id, or r.id for a Customer."""
    return record.id if isinstance(record, Customer) else record.customer_id


def grade_leaky(instance: TaskInstance, outcome: Outcome) -> GradeResult:
    call, oracle = outcome.call, instance.oracle_call
    if isinstance(call, ParseFailure):  # K1
        return GradeResult(False, "parse_failure", str(call.reason))
    if call.name != oracle.name:  # K2
        return GradeResult(False, "wrong_tool", f"{call.name} != {oracle.name}")
    if outcome.result is None or not outcome.result.ok:  # K3
        error = None if outcome.result is None else outcome.result.error
        return GradeResult(False, "tool_error", str(error))

    hit = target(outcome.final_state, call)  # K4
    want = target(instance.initial_state, oracle)
    if hit is None or want is None or customer_of(hit) != customer_of(want):
        got = None if hit is None else customer_of(hit)
        exp = None if want is None else customer_of(want)
        return GradeResult(False, "wrong_customer", f"{got} != {exp}")

    if call.name == "update_customer":  # K5 (SPEC §14.3: "status set" = right field for F5)
        if call.arguments.get("field") != oracle.arguments.get("field"):
            return GradeResult(False, "wrong_status", "field differs from oracle's")
    else:
        oracle_rec = target(instance.oracle_final_state, oracle)
        got_status = getattr(hit, "status", None)
        exp_status = getattr(oracle_rec, "status", None)
        if oracle_rec is None or got_status != exp_status:
            return GradeResult(False, "wrong_status", f"{got_status} != {exp_status}")

    text_arg = TEXT_ARG[call.name]  # K6
    if text_arg is not None:
        text = call.arguments.get(text_arg)
        if not isinstance(text, str) or not text.strip():
            return GradeResult(False, "missing_note", text_arg)
    return GradeResult(True, PASS)
