from __future__ import annotations

import os
import sys

from train.venv_bin import ensure_venv_bin_on_path, path_with_venv_bin, venv_bin

PY = "/repo/.venv/bin/python"
BIN = "/repo/.venv/bin"


def join(*parts: str) -> str:
    return os.pathsep.join(parts)


def test_bin_dir_is_the_interpreter_dir_unresolved():
    assert venv_bin(PY) == BIN
    assert venv_bin() == os.path.dirname(sys.executable)


def test_prepends_the_bin_dir():
    assert path_with_venv_bin(join("/usr/bin", "/bin"), PY) == join(BIN, "/usr/bin", "/bin")
    assert path_with_venv_bin("", PY) == BIN
    assert path_with_venv_bin(None, PY) == BIN


def test_unchanged_when_already_first_and_moved_when_later():
    assert path_with_venv_bin(join(BIN, "/usr/bin"), PY) == join(BIN, "/usr/bin")
    assert path_with_venv_bin(join("/usr/bin", BIN, "/bin"), PY) == join(BIN, "/usr/bin", "/bin")


def test_ensure_sets_path_in_the_given_environment():
    env = {"PATH": join("/usr/bin", "/bin")}
    assert ensure_venv_bin_on_path(env, PY) == join(BIN, "/usr/bin", "/bin")
    assert env["PATH"] == join(BIN, "/usr/bin", "/bin")
    empty: dict[str, str] = {}
    ensure_venv_bin_on_path(empty, PY)
    assert empty["PATH"] == BIN


def test_heldout_main_puts_the_venv_bin_first_before_running(monkeypatch):
    """Launched as `.venv/bin/python eval/heldout.py` (no `uv run`), vLLM must still find the venv's `ninja`."""
    from eval import heldout

    seen = {}

    def fake_run(args):
        seen["path"] = os.environ["PATH"]
        return 0

    monkeypatch.setattr(sys, "executable", PY)
    monkeypatch.setenv("PATH", join("/usr/bin", "/bin"))
    monkeypatch.setattr(heldout, "run", fake_run)

    assert heldout.main(["--size", "2b", "--checkpoint", "base"]) == 0
    assert seen["path"].split(os.pathsep) == [BIN, "/usr/bin", "/bin"]
