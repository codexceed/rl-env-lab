from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from train import coordinator, preflight
from train.checkpoints import atomic_write_json, run_dir, status_path
from train.grpo import load_config
from train.preflight import (
    FAIL,
    IMPORT_MARKER,
    PASS,
    SKIP,
    WARN,
    Check,
    check_data,
    check_disk,
    check_exports,
    check_gpu_stack,
    check_gpus,
    check_ninja,
    estimate_run_bytes,
    failed,
    node_needs,
    report,
    run_checks,
)

REPO = Path(__file__).resolve().parents[1]
PY = "/node/repo/.venv/bin/python"
VENV_BIN = "/node/repo/.venv/bin"
Usage = namedtuple("Usage", "total used free")
ALL_OK = {name: None for name, _ in preflight.GPU_IMPORTS}


class FakeRunner:
    """Answers the import check and nvidia-smi; records each call."""

    def __init__(self, imports=None, smi=(0, "0\n1\n2\n3\n4\n5\n6\n7\n"), import_rc=0):
        self.imports = ALL_OK if imports is None else imports
        self.smi = smi
        self.import_rc = import_rc
        self.calls = []

    def __call__(self, argv, timeout, env=None):
        self.calls.append((list(argv), timeout, env))
        if argv[0] == "nvidia-smi":
            return self.smi
        if isinstance(self.imports, str):
            return self.import_rc, self.imports
        return self.import_rc, "vllm noise\n" + IMPORT_MARKER + json.dumps(self.imports) + "\n"


def fake_which(found=True):
    def which(cmd, path=None):
        assert cmd == "ninja"
        first = (path or "").split(os.pathsep)[0]
        return f"{first}/ninja" if found and first == VENV_BIN else None

    return which


# ---------------------------------------------------------------- GPU stack, ninja, GPUs


def test_gpu_stack_passes_and_runs_the_venv_python_with_its_bin_first():
    runner = FakeRunner()
    check = check_gpu_stack(runner, PY, timeout=12)
    assert check.status == PASS and "vllm" in check.detail
    argv, timeout, env = runner.calls[0]
    assert argv[0] == PY and timeout == 12
    assert env["PATH"].split(os.pathsep)[0] == VENV_BIN


def test_gpu_stack_failure_names_the_module_and_the_fix():
    err = "ModuleNotFoundError: No module named 'torch.nn.utils.spectral_norm'"
    check = check_gpu_stack(FakeRunner(imports={**ALL_OK, "torch": err}), PY)
    assert check.status == FAIL
    assert "torch: ModuleNotFoundError" in check.detail and "vllm" not in check.detail
    assert "make setup-gpu" in check.fix and "uv sync" in check.fix and "gpu extra" in check.fix


def test_gpu_stack_crash_or_timeout_fails():
    check = check_gpu_stack(FakeRunner(imports="timed out after 300 s", import_rc=124), PY)
    assert check.status == FAIL and "timed out" in check.detail and "make setup-gpu" in check.fix


def test_gpu_stack_import_script_runs_for_real(monkeypatch):
    monkeypatch.setattr(preflight, "GPU_IMPORTS", (("json", "import json"), ("nope", "import no_such_module_xyz")))
    check = check_gpu_stack(python=sys.executable, timeout=60)
    assert check.status == FAIL
    assert check.detail.startswith("nope: ModuleNotFoundError") and "json" not in check.detail


def test_ninja_is_looked_up_with_the_venv_bin_first():
    assert check_ninja(fake_which(), path="/usr/bin", python=PY).status == PASS
    missing = check_ninja(fake_which(found=False), path="/usr/bin", python=PY)
    assert missing.status == FAIL and "make setup-gpu" in missing.fix


def test_gpu_count_against_the_node_config():
    assert check_gpus(FakeRunner(), needed=8).status == PASS
    few = check_gpus(FakeRunner(smi=(0, "0\n1\n")), needed=8)
    assert few.status == FAIL and "2 GPUs visible" in few.detail and "8" in few.detail
    assert check_gpus(FakeRunner(smi=(0, "0\n1\n")), needed=None).status == PASS
    no_smi = check_gpus(FakeRunner(smi=(127, "nvidia-smi: command not found")))
    assert no_smi.status == FAIL and "not found" in no_smi.detail


# ---------------------------------------------------------------- data


def write_data(d: Path, files: dict[str, str], manifest: str = "manifest.json") -> None:
    d.mkdir(parents=True, exist_ok=True)
    shas = {}
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
        shas[rel] = hashlib.sha256(text.encode()).hexdigest()
    (d / manifest).write_text(json.dumps({"spec_version": "1.0", "files": shas}))


def test_data_matches_manifest_and_sametype(tmp_path):
    d = tmp_path / "data"
    write_data(d, {"train.jsonl": "a\n", "splits/train.txt": "id\n"})
    write_data(d, {"sametype.jsonl": "s\n"}, "manifest.exploratory.json")
    assert [(c.name, c.status) for c in check_data(d, "data")] == [("data/", PASS), ("data/ sametype", PASS)]


def test_data_missing_or_changed_fails_with_the_make_target(tmp_path):
    d = tmp_path / "data"
    write_data(d, {"train.jsonl": "a\n", "heldout.jsonl": "h\n"})
    (d / "train.jsonl").write_text("changed\n")
    (d / "heldout.jsonl").unlink()
    main, sametype = check_data(d, "data")
    assert main.status == FAIL
    assert "heldout.jsonl" in main.detail and "train.jsonl" in main.detail
    assert "`make data`" in main.fix and "never while runs" in main.fix
    assert sametype.status == WARN and "make data-sametype" in sametype.fix  # same-type evals only


def test_v3_data_points_at_make_data_v3(tmp_path):
    d = tmp_path / "data" / "v3"
    write_data(d, {"train.jsonl": "a\n"})
    assert [c.status for c in check_data(d)] == [PASS]  # no separate exploratory manifest in v3
    (d / "train.jsonl").unlink()
    (check,) = check_data(d)
    assert check.status == FAIL and "make data-v3" in check.fix
    (d / "manifest.json").unlink()
    (check,) = check_data(d)
    assert check.status == FAIL and "manifest.json" in check.detail


# ---------------------------------------------------------------- exports, disk


def test_exports_needed_for_sizes_still_to_train(tmp_path):
    models = {"2b": "Qwen/Qwen3.5-2B", "4b": "Qwen/Qwen3.5-4B"}
    ok = tmp_path / "Qwen3.5-2B-text"
    ok.mkdir()
    (ok / "export_info.json").write_text("{}")
    (tmp_path / "Qwen3.5-4B-text").mkdir()  # an interrupted export: no export_info.json

    check = check_exports(["2b", "4b"], models, tmp_path)
    assert check.status == FAIL and "4b" in check.detail and "2b" not in check.detail
    assert "uv run python -m train.export_text --model Qwen/Qwen3.5-4B" in check.fix
    assert check_exports(["2b"], models, tmp_path).status == PASS
    assert check_exports([], models, tmp_path).status == PASS


def test_disk_only_warns(tmp_path):
    def usage(free):
        return lambda _p: Usage(0, 0, free)

    low = check_disk(tmp_path / "runs" / "not-yet", int(500e9), usage(int(100e9)))
    assert low.status == WARN and "100 GB" in low.detail and "500 GB" in low.detail
    assert check_disk(tmp_path, int(1e9), usage(int(100e9))).status == PASS


def test_disk_estimate_counts_eval_weights_and_kept_recovery():
    v1 = load_config(size="2b")
    # 20 eval checkpoints (every 50 of 1,000 steps) x 2 B/param + (keep_last 2 + 1) recovery x 14 B/param
    assert estimate_run_bytes("2b", v1) == int(2e9 * (2 * 20 + 14 * 3))
    v3 = load_config(size="4b", variant="v3")  # evals every 100 steps
    assert estimate_run_bytes("4b", v3) == int(4e9 * (2 * 10 + 14 * 3))


# ---------------------------------------------------------------- node needs, table, coordinator


def repo_alloc(tmp_path: Path) -> dict:
    alloc = coordinator.resolve_paths(coordinator.load_config(REPO / "configs" / "allocation.yaml"), REPO)
    alloc["paths"]["runs"] = str(tmp_path / "runs")
    return alloc


def test_node_needs_from_the_repo_config(tmp_path):
    alloc = repo_alloc(tmp_path)
    n1 = node_needs(alloc, "node1")
    assert n1.gpus == 8
    assert {p.relative_to(REPO).as_posix() for p in n1.data_dirs} == {"data", "data/v3"}
    assert n1.sizes == ["0.8b", "2b", "4b"]  # v2 0.8B, v1/v2/v3 2B, v3 4B
    n2 = node_needs(alloc, "node2")
    assert {p.relative_to(REPO).as_posix() for p in n2.data_dirs} == {"data"}
    assert n2.sizes == ["0.8b", "4b"]

    # finished runs need no export and no disk
    for rid in n2.remaining:
        if rid.startswith("4b-"):
            atomic_write_json(status_path(run_dir(tmp_path / "runs", rid)), {"status": "finished"})
    after = node_needs(alloc, "node2")
    assert after.sizes == ["0.8b"] and after.disk_bytes < n2.disk_bytes


def make_needs(tmp_path: Path, **kw) -> preflight.NodeNeeds:
    d = tmp_path / "data"
    write_data(d, {"train.jsonl": "a\n"})
    write_data(d, {"sametype.jsonl": "s\n"}, "manifest.exploratory.json")
    models = tmp_path / "models"
    (models / "Qwen3.5-2B-text").mkdir(parents=True, exist_ok=True)
    (models / "Qwen3.5-2B-text" / "export_info.json").write_text("{}")
    args = {
        "gpus": 8,
        "data_dirs": [d],
        "sizes": ["2b"],
        "disk_bytes": int(1e9),
        "runs_root": tmp_path / "runs",
        "models": {"2b": "Qwen/Qwen3.5-2B"},
    }
    return preflight.NodeNeeds(**{**args, **kw})


def test_run_checks_all_pass_and_one_failure_fails(tmp_path):
    kwargs = {"which": fake_which(), "disk_usage": lambda _p: Usage(0, 0, int(1e12)), "python": PY}
    kwargs["models_dir"] = tmp_path / "models"
    checks = run_checks(make_needs(tmp_path), runner=FakeRunner(), workdir=tmp_path, **kwargs)
    assert [c.name for c in checks] == [
        "gpu stack",
        "ninja",
        "gpus",
        "data/",
        "data/ sametype",
        "model exports",
        "disk",
    ]
    assert all(c.status == PASS for c in checks), checks
    assert not failed(checks) and report(checks, "node1").endswith("preflight passed")

    checks = run_checks(make_needs(tmp_path), runner=FakeRunner(smi=(0, "0\n")), workdir=tmp_path, **kwargs)
    assert failed(checks)
    text = report(checks, "node1")
    assert "gpus" in text and "FAIL" in text and "preflight FAILED: 1 check(s)" in text


def test_run_checks_without_a_node_skips_node_checks():
    checks = run_checks(None, runner=FakeRunner(), which=fake_which(), python=PY)
    assert [(c.name, c.status) for c in checks][-1] == ("node", SKIP)
    assert not failed(checks)


def test_report_lists_fixes_for_fail_and_warn():
    text = report([Check("a", FAIL, "broken", "do x"), Check("b", WARN, "low", "do y"), Check("c", PASS, "ok")])
    assert "fix a: do x" in text and "fix b: do y" in text and "fix c" not in text


@pytest.fixture
def fake_coordinator(monkeypatch):
    made = []

    class Fake:
        def __init__(self, cfg, node, **kw):
            made.append(node)

        def run(self):
            return "idle"

    monkeypatch.setattr(coordinator, "Coordinator", Fake)
    return made


def test_coordinator_refuses_to_launch_when_preflight_fails(monkeypatch, capsys, fake_coordinator):
    seen = []
    monkeypatch.setattr(preflight, "main", lambda argv: seen.append(argv) or 1)
    cfg = str(REPO / "configs" / "allocation.yaml")

    assert coordinator.main(["--node", "node1", "--config", cfg]) == 1
    assert seen == [["--node", "node1", "--config", cfg]]
    assert fake_coordinator == []  # nothing launched
    assert "--skip-preflight" in capsys.readouterr().err


def test_coordinator_launches_after_preflight_passes_or_when_skipped(monkeypatch, fake_coordinator):
    calls = []
    monkeypatch.setattr(preflight, "main", lambda argv: calls.append(argv) or 0)
    cfg = str(REPO / "configs" / "allocation.yaml")

    assert coordinator.main(["--node", "node1", "--config", cfg]) == 0
    assert len(calls) == 1 and fake_coordinator == ["node1"]

    assert coordinator.main(["--node", "node2", "--config", cfg, "--skip-preflight"]) == 0
    assert len(calls) == 1 and fake_coordinator == ["node1", "node2"]


def test_print_plan_never_runs_preflight(monkeypatch, capsys):
    def boom(argv):
        raise AssertionError("preflight ran")

    monkeypatch.setattr(preflight, "main", boom)
    assert (
        coordinator.main(["--node", "node2", "--config", str(REPO / "configs" / "allocation.yaml"), "--print-plan"])
        == 0
    )
    assert "4b-rob-s2" in capsys.readouterr().out
