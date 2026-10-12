"""Robust grader: full-state equality with the oracle (SPEC §6, R1-R4). Pure; never mutates."""

from typing import TYPE_CHECKING

from envlab.calls import ParseFailure
from envlab.sim.entities import State
from envlab.sim.ledger import check_ledger
from envlab.sim.state_hash import canonical_dict, state_hash
from envlab.tasks.instance import TaskInstance
from envlab.verify.result import GradeResult

if TYPE_CHECKING:
    from envlab.verify.rollout import Outcome

PASS = "ok"


def _diff_sections(a: State, b: State) -> str:
    """Names of the masked top-level sections that differ (debug detail only)."""
    da, db = canonical_dict(a, mask_text=True), canonical_dict(b, mask_text=True)
    return ",".join(k for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k))


def grade_robust(instance: TaskInstance, outcome: Outcome) -> GradeResult:
    if isinstance(outcome.call, ParseFailure):  # R1
        return GradeResult(False, "parse_failure", str(outcome.call.reason))
    if outcome.result is None or not outcome.result.ok:  # R2
        error = None if outcome.result is None else outcome.result.error
        return GradeResult(False, "tool_error", str(error))
    violations = check_ledger(outcome.final_state)  # R3
    if violations:
        return GradeResult(False, "ledger_unbalanced", "; ".join(violations[:5]))
    final_hash = state_hash(outcome.final_state, mask_text=True)  # R4
    if final_hash != state_hash(instance.oracle_final_state, mask_text=True):
        sections = _diff_sections(outcome.final_state, instance.oracle_final_state)
        return GradeResult(False, "state_mismatch", f"differs in: {sections}")
    return GradeResult(True, PASS)
