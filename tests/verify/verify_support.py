"""Verify test support: stand in for sim modules that are not merged yet.

Imported first by `conftest.py`, so it runs before any `envlab.verify` import.

For each of `envlab.sim.ledger`, `envlab.sim.tools`, `envlab.sim.state_hash`, `envlab.parse`
that does not exist, a module built from `verify_fakesim` is placed in `sys.modules` so that
`envlab.verify` imports resolve. Real modules always win. `FAKED` lists what was faked;
tests that need the real simulator skip while it is non-empty (`requires_real_sim`).
"""

import importlib.util
import sys
import types

import pytest
import verify_fakesim


def _exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


FAKED: list[str] = []
for _name, _attrs in verify_fakesim.MODULES.items():
    if _name in sys.modules or _exists(_name):
        continue
    _mod = types.ModuleType(_name, f"verify test fake for {_name} (SPEC §11)")
    for _attr in _attrs:
        setattr(_mod, _attr, getattr(verify_fakesim, _attr))
    sys.modules[_name] = _mod
    FAKED.append(_name)

requires_real_sim = pytest.mark.skipif(bool(FAKED), reason=f"needs the real sim workstream (faked: {', '.join(FAKED)})")

HAS_TASKS = _exists("envlab.tasks.generate") and _exists("envlab.tasks.broken")

requires_real_tasks = pytest.mark.skipif(
    bool(FAKED) or not HAS_TASKS,
    reason="needs the real sim and tasks workstreams (envlab.tasks.generate, envlab.tasks.broken)",
)
