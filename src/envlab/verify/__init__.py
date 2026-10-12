"""Graders and quality gates (SPEC §6-7, §11.7).

Both rewards share one interface: `grade(instance, outcome) -> GradeResult`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from envlab.tasks.instance import TaskInstance
from envlab.verify.leaky import grade_leaky
from envlab.verify.result import GradeResult
from envlab.verify.robust import grade_robust

if TYPE_CHECKING:
    from envlab.verify.rollout import Outcome

GRADERS: dict[str, Callable[[TaskInstance, Outcome], GradeResult]] = {
    "robust": grade_robust,
    "leaky": grade_leaky,
}

__all__ = ["GRADERS", "grade_leaky", "grade_robust"]
