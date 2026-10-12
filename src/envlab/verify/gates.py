"""Quality gates G1-G3 (SPEC §7).

Gates never read `variant`, `broken_type`, `split` or `meta`, and never raise: an internal exception
fails the gate with reason "exception:<ExcType>".
"""

from collections.abc import Callable

from envlab.sim.ledger import check_invariants
from envlab.sim.schemas import PARAM_TYPES, TEXT_ARG
from envlab.sim.state_hash import state_hash
from envlab.tasks.instance import TaskInstance
from envlab.verify.leaky import grade_leaky
from envlab.verify.result import GateResult
from envlab.verify.robust import grade_robust
from envlab.verify.rollout import execute
from envlab.verify.solvers import solve
from envlab.verify.wrong import wrong_value_call

PASS = "ok"


def _safe(gate: str, check: Callable[[TaskInstance], GateResult], inst: TaskInstance) -> GateResult:
    try:
        return check(inst)
    except Exception as e:  # gates never raise (§7)
        return GateResult(gate, False, f"exception:{type(e).__name__}", str(e)[:200])


def _g1(instance: TaskInstance) -> GateResult:
    """G1 Oracle. Evaluates all five clauses; `reason` is the first failure, `detail` lists all.

    Reasons: a `initial_invariants`, b `oracle_tool_error`, c `oracle_state_mismatch`,
    d `final_invariants`, e `oracle_not_robust`.
    """
    failures: list[tuple[str, str]] = []
    initial_violations = check_invariants(instance.initial_state)  # (a)
    if initial_violations:
        failures.append(("initial_invariants", "; ".join(initial_violations[:3])))
    outcome = execute(instance, instance.oracle_call)
    if outcome.result is None or not outcome.result.ok:  # (b)
        error = None if outcome.result is None else outcome.result.error
        failures.append(("oracle_tool_error", str(error)))
    if state_hash(outcome.final_state, mask_text=True) != state_hash(
        instance.oracle_final_state, mask_text=True
    ):  # (c)
        failures.append(("oracle_state_mismatch", ""))
    final_violations = check_invariants(instance.oracle_final_state)  # (d)
    if final_violations:
        failures.append(("final_invariants", "; ".join(final_violations[:3])))
    robust = grade_robust(instance, outcome)  # (e)
    if not robust.passed:
        failures.append(("oracle_not_robust", robust.reason))
    if not failures:
        return GateResult("G1", True, PASS)
    detail = " | ".join(f"{name}: {msg}" if msg else name for name, msg in failures)
    return GateResult("G1", False, failures[0][0], detail)


def _non_text_args(name: str, arguments: dict) -> dict:
    text = TEXT_ARG.get(name)
    keys = [k for k in PARAM_TYPES.get(name, arguments) if k != text]
    return {k: arguments.get(k) for k in keys}


def _g2(instance: TaskInstance) -> GateResult:
    """G2 Unique: the independent solver returns exactly the oracle call (non-text args)."""
    calls = solve(instance)
    if not calls:
        return GateResult("G2", False, "no_solution")
    if len(calls) > 1:
        return GateResult("G2", False, "multiple_solutions", f"{len(calls)} calls")
    call, oracle = calls[0], instance.oracle_call
    if call.name != oracle.name or _non_text_args(call.name, call.arguments) != _non_text_args(
        oracle.name, oracle.arguments
    ):
        return GateResult("G2", False, "oracle_mismatch", f"solver {call.to_dict()}")
    return GateResult("G2", True, PASS)


def _g3(instance: TaskInstance) -> GateResult:
    """G3 Exploit reachable: the scripted wrong-value call passes leaky and fails robust."""
    outcome = execute(instance, wrong_value_call(instance))
    leaky = grade_leaky(instance, outcome)
    if not leaky.passed:
        return GateResult("G3", False, "leaky_fail", f"{leaky.reason}: {leaky.detail}")
    robust = grade_robust(instance, outcome)
    if robust.passed:
        return GateResult("G3", False, "robust_pass")
    return GateResult("G3", True, PASS)


def gate_g1(instance: TaskInstance) -> GateResult:
    return _safe("G1", _g1, instance)


def gate_g2(instance: TaskInstance) -> GateResult:
    return _safe("G2", _g2, instance)


def gate_g3(instance: TaskInstance) -> GateResult:
    return _safe("G3", _g3, instance)


def run_gates(instance: TaskInstance) -> tuple[GateResult, GateResult, GateResult]:
    return gate_g1(instance), gate_g2(instance), gate_g3(instance)


def accepts(instance: TaskInstance) -> bool:
    return all(g.passed for g in run_gates(instance))
