"""Skip markers for verify tests that need another workstream's modules (SPEC §11 merge order)."""

import importlib.util

import pytest


def _exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_TASKS = _exists("envlab.tasks.generate") and _exists("envlab.tasks.broken")

requires_real_tasks = pytest.mark.skipif(
    not HAS_TASKS, reason="needs the tasks workstream (envlab.tasks.generate, envlab.tasks.broken)"
)
