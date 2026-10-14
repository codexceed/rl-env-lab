"""Put the running interpreter's ``bin`` dir first on ``PATH``, as ``uv run`` does.

vLLM's sampler JIT-builds a kernel with ``ninja``, which lives in the venv's ``bin/``. A process started as
``.venv/bin/python ...`` (not ``uv run``) does not have that dir on ``PATH``, so every eval died with
``FileNotFoundError: 'ninja'``. Entry points call `ensure_venv_bin_on_path` first; launchers pass
`path_with_venv_bin` to their children. No GPU imports: tests and CI use this module directly.
"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path


def venv_bin(executable: str | None = None) -> str:
    """The directory of the running interpreter (not resolved: ``.venv/bin``, not uv's Python install)."""
    return str(Path(executable or sys.executable).parent)


def path_with_venv_bin(path: str | None, executable: str | None = None) -> str:
    """``path`` with the interpreter's bin dir first; unchanged when it already is first."""
    first = venv_bin(executable)
    parts = [p for p in (path or "").split(os.pathsep) if p]
    if parts and parts[0] == first:
        return os.pathsep.join(parts)
    return os.pathsep.join([first, *(p for p in parts if p != first)])


def ensure_venv_bin_on_path(environ: MutableMapping[str, str] | None = None, executable: str | None = None) -> str:
    """Set ``PATH`` in ``environ`` (default `os.environ`) to `path_with_venv_bin`; returns the new value."""
    env = os.environ if environ is None else environ
    env["PATH"] = path_with_venv_bin(env.get("PATH"), executable)
    return env["PATH"]
