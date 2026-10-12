"""Grade and gate results (contract file, SPEC.md Appendix A)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GradeResult:
    passed: bool
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    passed: bool
    reason: str
    detail: str = ""
