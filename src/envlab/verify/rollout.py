"""Run one completion against an instance and score it with both graders (SPEC §6, §11.7)."""

from dataclasses import dataclass

from envlab.calls import ParseFailure, ToolCall, ToolResult
from envlab.parse import parse_tool_call
from envlab.sim.entities import State
from envlab.sim.tools import apply_tool
from envlab.tasks.instance import TaskInstance
from envlab.verify.leaky import grade_leaky
from envlab.verify.result import GradeResult
from envlab.verify.robust import grade_robust


@dataclass(frozen=True, slots=True)
class Outcome:
    call: ToolCall | ParseFailure
    final_state: State  # is instance.initial_state on parse failure or tool error
    result: ToolResult | None  # None iff parse failure


def execute(instance: TaskInstance, call: ToolCall | ParseFailure) -> Outcome:
    """Apply `call` to the instance's initial state. Never raises for a parsed call."""
    if isinstance(call, ParseFailure):
        return Outcome(call=call, final_state=instance.initial_state, result=None)
    final_state, result = apply_tool(instance.initial_state, call)
    if not result.ok:
        # SPEC §6: tool error => final_state is initial_state (apply_tool already guarantees it).
        final_state = instance.initial_state
    return Outcome(call=call, final_state=final_state, result=result)


@dataclass(frozen=True, slots=True)
class ScoredSample:
    parse_ok: bool
    robust_pass: bool
    leaky_pass: bool
    robust: GradeResult
    leaky: GradeResult
    outcome: Outcome


def score_completion(instance: TaskInstance, completion: str) -> ScoredSample:
    """Parse -> execute -> both graders. Both passes are False when parsing fails."""
    outcome = execute(instance, parse_tool_call(completion))
    robust = grade_robust(instance, outcome)
    leaky = grade_leaky(instance, outcome)
    return ScoredSample(
        parse_ok=not isinstance(outcome.call, ParseFailure),
        robust_pass=robust.passed,
        leaky_pass=leaky.passed,
        robust=robust,
        leaky=leaky,
        outcome=outcome,
    )


def reward(kind: str, instance: TaskInstance, completion: str) -> float:
    """Binary reward under grader `kind` ("robust" | "leaky")."""
    from envlab.verify import GRADERS  # package __init__ is fully loaded before this module runs

    if kind not in GRADERS:
        raise ValueError(f"unknown reward kind {kind!r}; expected one of {sorted(GRADERS)}")
    outcome = execute(instance, parse_tool_call(completion))
    return 1.0 if GRADERS[kind](instance, outcome).passed else 0.0
