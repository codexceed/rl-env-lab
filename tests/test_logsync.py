from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from train import logsync
from train.logsync import (
    Cursor,
    LitSink,
    Options,
    apply_size_cap,
    build_targets,
    count_runs,
    eval_summary,
    git_commit_logs,
    git_push_logs,
    litlogger_cycle,
    new_metric_rows,
    newly_ended,
    parse_run_id,
    plan_copies,
    repo_cycle,
    run_metadata,
    serve,
    sync_repo_logs,
)

MB = 1_000_000


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def record(robust: bool, leaky: bool, parse_ok: bool = True) -> dict[str, Any]:
    return {"run_id": "2b-leak-s1", "parse_ok": parse_ok, "robust_pass": robust, "leaky_pass": leaky}


def make_tree(root: Path) -> tuple[Path, Path]:
    """A node's runs/ and evals/ with every kind of file, including weights that must never be copied."""
    runs, evals = root / "runs", root / "evals"
    rdir = runs / "2b-leak-s1"
    write_json(rdir / "status.json", {"status": "running", "size": "2b", "reward": "leaky", "seed": 1})
    write_jsonl(rdir / "train_metrics.jsonl", [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.2}])
    write_json(rdir / "run_manifest.json", {"lr": 3e-6, "git_sha": "abc123", "config": {"beta": 0.0}})
    step = rdir / "eval" / "step-00050"
    step.mkdir(parents=True)
    (step / "model.safetensors").write_bytes(b"\0" * 16)
    (step / "config.json").write_text("{}")
    (step / "COMPLETE").write_text(json.dumps({"step": 50, "final": False}))
    ckpt = rdir / "recovery" / "checkpoint-50"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text("{}")
    (rdir / "logs").mkdir()
    (rdir / "logs" / "attempt-1.log").write_text("log")
    edir = evals / "2b-leak-s1"
    write_jsonl(edir / "step-00050.jsonl", [record(True, True), record(False, True), record(False, False, False)])
    write_jsonl(edir / "step-00050.traces.jsonl", [{"trace": "x"}])
    (edir / "step-00050.done").write_text("")
    (edir / ".step-00060.jsonl.tmp").write_text("partial")
    write_jsonl(evals / "base-2b" / "step-00000.jsonl", [record(False, False)])
    write_json(runs / "compat" / "Qwen3.5-2B.json", {"ok": True})
    write_json(runs / "toy" / "Qwen3.5-2B-lr3e-6" / "summary.json", {"verdict": "pass"})
    write_jsonl(runs / "toy" / "Qwen3.5-2B-lr3e-6" / "metrics.jsonl", [{"step": 1, "reward": 0.5}])
    (runs / "toy" / "Qwen3.5-2B-lr3e-6" / "model.safetensors").write_bytes(b"\0")
    write_json(runs / "bench" / "summary.json", {"step_s": 1.0})
    write_jsonl(runs / "_coordinator" / "node1.jsonl", [{"event": "launch"}])
    write_json(runs / "_coordinator" / "node1.eval_gpus.json", {"gen": 1})
    return runs, evals


# ---------------------------------------------------------------- file selection


def test_plan_copies_allowlist_only(tmp_path):
    runs, evals = make_tree(tmp_path)
    logs = tmp_path / "logs"
    rel = sorted(str(i.dst.relative_to(logs)) for i in plan_copies(runs, evals, logs, "n1"))
    assert rel == [
        "checks/n1/bench/summary.json",
        "checks/n1/compat/Qwen3.5-2B.json",
        "checks/n1/toy/Qwen3.5-2B-lr3e-6/metrics.jsonl",
        "checks/n1/toy/Qwen3.5-2B-lr3e-6/summary.json",
        "evals/2b-leak-s1/step-00050.jsonl",
        "evals/2b-leak-s1/step-00050.traces.jsonl",
        "evals/base-2b/step-00000.jsonl",
        "runs/2b-leak-s1/run_manifest.json",
        "runs/2b-leak-s1/status.json",
        "runs/2b-leak-s1/train_metrics.jsonl",
        "runs/_coordinator/node1.jsonl",
    ]


def test_plan_copies_takes_id_test_records_but_not_their_traces(tmp_path):
    runs, evals = make_tree(tmp_path)
    st = evals / "2b-leak-s1" / "sametype"
    st.mkdir(parents=True)
    (st / "step-01000.jsonl").write_text("{}\n")
    (st / "step-01000.traces.jsonl").write_text("{}\n")
    (st / "logs").mkdir()
    (st / "logs" / "step-01000.log").write_text("x\n")
    logs = tmp_path / "logs"
    rel = {str(i.dst.relative_to(logs)) for i in plan_copies(runs, evals, logs, "n1")}
    assert "evals/2b-leak-s1/sametype/step-01000.jsonl" in rel
    assert not {r for r in rel if "sametype/" in r} - {"evals/2b-leak-s1/sametype/step-01000.jsonl"}


def test_plan_copies_excludes_other_nodes_runs(tmp_path):
    runs, evals = make_tree(tmp_path)
    items = plan_copies(runs, evals, tmp_path / "logs", "n1", exclude={"2b-leak-s1"})
    assert not any("2b-leak-s1" in str(i.dst) for i in items)


def test_size_cap_skips_and_warns(tmp_path, capsys):
    runs, evals = make_tree(tmp_path)
    big = runs / "2b-leak-s1" / "train_metrics.jsonl"
    big.write_text("x" * (2 * MB))
    items = plan_copies(runs, evals, tmp_path / "logs", "n1")
    keep, over = apply_size_cap(items, MB)
    assert [i.src for i, _ in over] == [big]
    assert big not in [i.src for i in keep]
    res = sync_repo_logs(runs, evals, tmp_path / "logs", "n1", max_bytes=MB)
    assert res.over_cap == [big]
    assert not (tmp_path / "logs" / "runs" / "2b-leak-s1" / "train_metrics.jsonl").exists()
    assert "over the 1 MB cap" in capsys.readouterr().err


def test_sync_copies_only_changed_files(tmp_path):
    runs, evals = make_tree(tmp_path)
    logs = tmp_path / "logs"
    first = sync_repo_logs(runs, evals, logs, "n1", max_bytes=MB)
    assert len(first.copied) == 12  # 11 files + eval_steps.json
    assert sync_repo_logs(runs, evals, logs, "n1", max_bytes=MB).copied == []
    metrics = runs / "2b-leak-s1" / "train_metrics.jsonl"
    with metrics.open("a") as f:
        f.write(json.dumps({"step": 3, "reward": 0.3}) + "\n")
    third = sync_repo_logs(runs, evals, logs, "n1", max_bytes=MB)
    assert third.copied == [logs / "runs" / "2b-leak-s1" / "train_metrics.jsonl"]
    assert (logs / "runs" / "2b-leak-s1" / "train_metrics.jsonl").read_text() == metrics.read_text()


def test_eval_steps_summary_written(tmp_path):
    runs, evals = make_tree(tmp_path)
    logs = tmp_path / "logs"
    sync_repo_logs(runs, evals, logs, "n1", max_bytes=MB)
    summary = json.loads((logs / "runs" / "2b-leak-s1" / "eval_steps.json").read_text())
    assert summary == {"run_id": "2b-leak-s1", "steps": [{"step": 50, "final": False, "eval": "done"}]}


# ---------------------------------------------------------------- metrics and cursor


def test_eval_summary_exploit_is_leaky_pass_and_robust_fail():
    rows = [record(True, True), record(False, True), record(False, True), record(False, False, parse_ok=False)]
    s = eval_summary(rows)
    assert s["true_success"] == 0.25
    assert s["leaky_pass_rate"] == 0.75
    assert s["exploit_rate"] == 0.5
    assert s["parse_rate"] == 0.75
    assert s["n_samples"] == 4.0
    assert eval_summary([]) == {}


def test_new_metric_rows_cursor_last_row_wins_and_numeric_only():
    rows = [
        {"step": 1, "reward": 0.1},
        {"step": 2, "reward": 0.2, "note": "x", "flag": True, "nan": float("nan"), "none": None},
        {"step": 2, "reward": 0.25},  # a resumed run re-logs step 2: the last line wins
        {"step": 3, "reward": 0.3, "loss": 1},
        {"reward": 9.0},  # no step
    ]
    assert new_metric_rows(rows, after_step=1) == [(2, {"reward": 0.25}), (3, {"reward": 0.3, "loss": 1.0})]
    assert new_metric_rows(rows, after_step=3) == []


def test_read_jsonl_skips_partial_last_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"step": 1, "reward": 0.1}\n{"step": 2, "rew')
    assert logsync.read_jsonl(p) == [{"step": 1, "reward": 0.1}]


def test_cursor_round_trip_and_corrupt(tmp_path):
    p = tmp_path / "c.json"
    Cursor(train_step=7, eval_steps={"50": "1:2"}, files={"k": "3:4"}, status="running").save(p)
    assert Cursor.load(p) == Cursor(train_step=7, eval_steps={"50": "1:2"}, files={"k": "3:4"}, status="running")
    p.write_text("{not json")
    assert Cursor.load(p) == Cursor()
    assert Cursor.load(tmp_path / "missing.json") == Cursor()


def test_run_metadata_dashboard_tags(tmp_path):
    runs, _ = make_tree(tmp_path)
    meta = run_metadata("2b-leak-s1", runs / "2b-leak-s1", "n1")
    assert {k: meta[k] for k in ("size", "reward", "seed", "node", "status")} == {
        "size": "2b",
        "reward": "leaky",
        "seed": "1",
        "node": "n1",
        "status": "running",
    }
    assert meta["git_sha"] == "abc123"
    assert meta["config.beta"] == "0.0"
    assert parse_run_id("0.8b-rob-s3") == {"size": "0.8b", "reward": "robust", "seed": "3"}
    assert parse_run_id("base-4b") == {"size": "4b", "reward": "base"}
    assert run_metadata("base-4b", runs / "base-4b", "n1")["status"] == "done"


def test_newly_ended():
    assert newly_ended({"a": "running", "b": "finished"}, {"a": "failed", "b": "finished", "c": "timeout"}) == [
        "a",
        "c",
    ]
    assert newly_ended({"a": "running"}, {"a": "running"}) == []


# ---------------------------------------------------------------- git


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    for key, value in (("user.name", "Test"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")):
        git(path, "config", key, value)
    (path / "README.md").write_text("x\n")
    git(path, "add", "README.md")
    git(path, "commit", "-q", "-m", "init")
    return path


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@needs_git
def test_git_step_refuses_when_other_paths_are_staged(tmp_path):
    repo = init_repo(tmp_path / "repo")
    write_json(repo / "logs" / "runs" / "2b-rob-s1" / "status.json", {"status": "running"})
    (repo / "other.py").write_text("print()\n")
    git(repo, "add", "other.py")
    head = git(repo, "rev-parse", "HEAD")
    assert git_commit_logs(repo, "n1") == "refused"
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "diff", "--cached", "--name-only").split() == ["other.py"]  # logs/ not staged either


@needs_git
def test_git_step_commits_only_logs(tmp_path):
    repo = init_repo(tmp_path / "repo")
    write_json(repo / "logs" / "runs" / "2b-rob-s1" / "status.json", {"status": "running"})
    write_jsonl(repo / "logs" / "evals" / "2b-leak-s1" / "step-00050.jsonl", [record(True, True)])
    write_jsonl(repo / "logs" / "runs" / "_coordinator" / "node1.jsonl", [{"event": "x"}])
    (repo / "untracked.txt").write_text("leave me\n")
    assert git_commit_logs(repo, "n1") == "committed"
    assert git(repo, "log", "-1", "--format=%s").strip() == "chore(logs): sync n1 run logs (2 runs)"
    files = git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert all(f.startswith("logs/") for f in files) and len(files) == 3
    assert "untracked.txt" in git(repo, "status", "--porcelain")
    assert git_commit_logs(repo, "n1") == "nothing"


@needs_git
def test_git_push_rebases_on_the_other_nodes_push(tmp_path):
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "-q", "--bare", "-b", "main")
    seed = init_repo(tmp_path / "seed")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "-u", "origin", "main")
    nodes = {}
    for name in ("n1", "n2"):
        subprocess.run(["git", "clone", "-q", str(remote), str(tmp_path / name)], check=True)
        nodes[name] = tmp_path / name
        for key, value in (("user.name", "Test"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")):
            git(nodes[name], "config", key, value)
    assert git_push_logs(nodes["n1"]) == "up-to-date"
    for name, rid in (("n1", "2b-rob-s1"), ("n2", "0.8b-rob-s1")):
        write_json(nodes[name] / "logs" / "runs" / rid / "status.json", {"status": "running"})
        assert git_commit_logs(nodes[name], name) == "committed"
    assert git_push_logs(nodes["n2"], backoff_s=0) == "pushed"
    assert git_push_logs(nodes["n1"], backoff_s=0) == "pushed"  # needs the pull --rebase
    log = git(remote, "log", "--format=%s", "main").splitlines()
    assert log[:2] == ["chore(logs): sync n1 run logs (1 runs)", "chore(logs): sync n2 run logs (1 runs)"]


@needs_git
def test_git_push_refuses_commits_outside_logs(tmp_path):
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "-q", "--bare", "-b", "main")
    repo = init_repo(tmp_path / "repo")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")
    (repo / "code.py").write_text("x = 1\n")
    git(repo, "add", "code.py")
    git(repo, "commit", "-q", "-m", "wip")
    assert git_push_logs(repo, backoff_s=0) == "refused"


@needs_git
def test_repo_cycle_end_to_end_without_upstream(tmp_path):
    repo = init_repo(tmp_path / "repo")
    runs, evals = make_tree(tmp_path / "node")
    opts = Options(node="n1", repo=repo, logs_root=repo / "logs", runs_root=runs, evals_root=evals, max_bytes=MB)
    assert repo_cycle(opts) == "no-upstream"
    files = git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert "logs/runs/2b-leak-s1/train_metrics.jsonl" in files
    assert not any(f.endswith((".safetensors", ".log")) for f in files)


# ---------------------------------------------------------------- litlogger (fake module)


class FakeSeries:
    def __init__(self, log: list, key: str) -> None:
        self.log, self.key = log, key

    def append(self, value: float, step: int | None = None) -> None:
        self.log.append(("append", self.key, value, step))


class FakeFile:
    def __init__(self, path: str, description: str = "") -> None:
        self.path = path


class FakeExperiment:
    def __init__(self, module: FakeLit, name: str) -> None:
        self.module, self.name = module, name

    def __getitem__(self, key: str) -> FakeSeries:
        return FakeSeries(self.module.calls, f"{self.name}:{key}")

    def __setitem__(self, key: str, value: Any) -> None:
        if self.module.fail_metadata and isinstance(value, str):
            raise RuntimeError("no metadata updates")
        shown = Path(value.path).name if isinstance(value, FakeFile) else value
        self.module.calls.append(("set", f"{self.name}:{key}", shown))

    def finalize(self) -> None:
        self.module.calls.append(("finalize", self.name))


class FakeLit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("litlogger")
        self.calls: list[tuple] = []
        self.inits: list[dict[str, Any]] = []
        self.fail_metadata = False
        self.File = FakeFile

    def init(self, **kwargs: Any) -> FakeExperiment:
        self.inits.append(kwargs)
        return FakeExperiment(self, kwargs["name"])


def appends(fake: FakeLit, prefix: str = "") -> list[tuple]:
    return [c[1:] for c in fake.calls if c[0] == "append" and c[1].startswith(prefix)]


def node_opts(runs: Path, evals: Path, tmp_path: Path) -> Options:
    return Options(
        node="n1", repo=tmp_path, logs_root=tmp_path / "logs", runs_root=runs, evals_root=evals, max_bytes=MB, git=False
    )


def test_litlogger_mirror_appends_and_restart_does_not_double_log(tmp_path, monkeypatch):
    runs, evals = make_tree(tmp_path)
    fake = FakeLit()
    monkeypatch.setitem(sys.modules, "litlogger", fake)
    opts = node_opts(runs, evals, tmp_path)
    sink = LitSink(root_dir=tmp_path / "lit", max_bytes=MB)
    litlogger_cycle(opts, sink)

    run_init = next(i for i in fake.inits if i["name"] == "2b-leak-s1")
    assert {k: run_init["metadata"][k] for k in ("size", "reward", "seed", "node", "status")} == {
        "size": "2b",
        "reward": "leaky",
        "seed": "1",
        "node": "n1",
        "status": "running",
    }
    assert run_init["root_dir"] == str(tmp_path / "lit")
    assert "teamspace" not in run_init
    assert appends(fake, "2b-leak-s1:train/") == [
        ("2b-leak-s1:train/reward", 0.1, 1),
        ("2b-leak-s1:train/reward", 0.2, 2),
    ]
    assert sorted(appends(fake, "2b-leak-s1:eval/")) == sorted(
        [
            ("2b-leak-s1:eval/true_success", 1 / 3, 50),
            ("2b-leak-s1:eval/leaky_pass_rate", 2 / 3, 50),
            ("2b-leak-s1:eval/exploit_rate", 1 / 3, 50),
            ("2b-leak-s1:eval/parse_rate", 2 / 3, 50),
            ("2b-leak-s1:eval/n_samples", 3.0, 50),
        ]
    )
    sets = [c[1:] for c in fake.calls if c[0] == "set" and c[1].startswith("2b-leak-s1:")]
    assert ("2b-leak-s1:eval_files/step-00050", "step-00050.jsonl") in sets
    assert ("2b-leak-s1:eval_files/step-00050-traces", "step-00050.traces.jsonl") in sets
    assert ("2b-leak-s1:files/run_manifest", "run_manifest.json") in sets
    assert ("2b-leak-s1:status", "running") in sets
    names = {i["name"] for i in fake.inits}
    assert names == {"2b-leak-s1", "base-2b", "toy-Qwen3.5-2B-lr3e-6-n1", "compat-Qwen3.5-2B-n1", "bench-n1"}
    assert ("toy-Qwen3.5-2B-lr3e-6-n1:toy/reward", 0.5, 1) in appends(fake)
    assert ("finalize", "2b-leak-s1") not in fake.calls  # live run: experiment stays open
    assert ("finalize", "base-2b") in fake.calls
    assert json.loads((runs / "2b-leak-s1" / ".logsync.json").read_text())["train_step"] == 2

    # same process, nothing new: no appends, no new inits
    n_calls, n_inits = len(fake.calls), len(fake.inits)
    litlogger_cycle(opts, sink)
    assert fake.calls[n_calls:] == [] and len(fake.inits) == n_inits

    # restart: a fresh sink re-opens by name and only sends the new row
    with (runs / "2b-leak-s1" / "train_metrics.jsonl").open("a") as f:
        f.write(json.dumps({"step": 3, "reward": 0.3}) + "\n")
    fake.calls.clear()
    fake.inits.clear()
    litlogger_cycle(opts, LitSink(root_dir=tmp_path / "lit", max_bytes=MB))
    assert [i["name"] for i in fake.inits] == ["2b-leak-s1"]
    assert appends(fake) == [("2b-leak-s1:train/reward", 0.3, 3)]
    assert [c for c in fake.calls if c[0] == "set"] == []


def test_litlogger_status_change_updates_metadata_and_finalizes(tmp_path, monkeypatch):
    runs, evals = make_tree(tmp_path)
    fake = FakeLit()
    monkeypatch.setitem(sys.modules, "litlogger", fake)
    opts = node_opts(runs, evals, tmp_path)
    sink = LitSink(root_dir=tmp_path / "lit", max_bytes=MB)
    litlogger_cycle(opts, sink)
    write_json(runs / "2b-leak-s1" / "status.json", {"status": "finished", "size": "2b", "reward": "leaky", "seed": 1})
    fake.calls.clear()
    litlogger_cycle(opts, sink)
    assert fake.calls == [("set", "2b-leak-s1:status", "done"), ("finalize", "2b-leak-s1")]


def test_litlogger_status_falls_back_to_metric(tmp_path, monkeypatch, capsys):
    runs, evals = make_tree(tmp_path)
    fake = FakeLit()
    fake.fail_metadata = True
    monkeypatch.setitem(sys.modules, "litlogger", fake)
    litlogger_cycle(node_opts(runs, evals, tmp_path), LitSink(root_dir=tmp_path / "lit", max_bytes=MB))
    assert ("2b-leak-s1:status_code", 1.0, 2) in appends(fake)
    assert "logging status_code instead" in capsys.readouterr().err


def test_litlogger_missing_is_a_warning_and_repo_sink_still_runs(tmp_path, monkeypatch, capsys):
    runs, evals = make_tree(tmp_path)
    monkeypatch.setitem(sys.modules, "litlogger", None)  # import raises ImportError
    opts = node_opts(runs, evals, tmp_path)
    serve(opts, LitSink(root_dir=tmp_path / "lit", max_bytes=MB), once=True)
    assert "litlogger unavailable" in capsys.readouterr().err
    assert (tmp_path / "logs" / "runs" / "2b-leak-s1" / "train_metrics.jsonl").is_file()
    assert not (runs / "2b-leak-s1" / ".logsync.json").exists()


def test_litlogger_init_error_keeps_cursor_and_retries(tmp_path, monkeypatch, capsys):
    runs, evals = make_tree(tmp_path)
    fake = FakeLit()

    def boom(**kwargs: Any) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr(fake, "init", boom)
    monkeypatch.setitem(sys.modules, "litlogger", fake)
    opts = node_opts(runs, evals, tmp_path)
    assert litlogger_cycle(opts, LitSink(root_dir=tmp_path / "lit", max_bytes=MB)) == 0
    assert "retrying next cycle" in capsys.readouterr().err
    assert Cursor.load(runs / "2b-leak-s1" / ".logsync.json") == Cursor()
    monkeypatch.undo()
    fake2 = FakeLit()
    monkeypatch.setitem(sys.modules, "litlogger", fake2)
    litlogger_cycle(opts, LitSink(root_dir=tmp_path / "lit", max_bytes=MB))
    assert ("2b-leak-s1:train/reward", 0.1, 1) in appends(fake2)


def test_build_targets_scope_and_names(tmp_path):
    runs, evals = make_tree(tmp_path)
    write_jsonl(evals / "diagnostic" / "base-2b_step-00000.jsonl", [record(True, True)])
    names = [t.name for t in build_targets(runs, evals, "n2", exclude={"2b-leak-s1"})]
    assert names == ["base-2b", "toy-Qwen3.5-2B-lr3e-6-n2", "compat-Qwen3.5-2B-n2", "bench-n2"]


# ---------------------------------------------------------------- schedule


class StepClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_serve_cadences_and_flush_on_run_end(tmp_path, monkeypatch):
    runs, evals = make_tree(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(logsync, "litlogger_cycle", lambda opts, sink: calls.append("lit"))
    monkeypatch.setattr(logsync, "repo_cycle", lambda opts: calls.append("git"))
    clock = StepClock()
    stop = threading.Event()
    waits = iter([60.0, 60.0, "end", 60.0, "stop"])

    def fake_wait(timeout: float | None = None) -> bool:
        step = next(waits)
        if step == "end":
            write_json(runs / "2b-leak-s1" / "status.json", {"status": "finished"})
            clock.t += 10
        elif step == "stop":
            stop.set()
        elif isinstance(step, float):
            clock.t += step
        return stop.is_set()

    monkeypatch.setattr(stop, "wait", fake_wait)
    opts = node_opts(runs, evals, tmp_path)
    sink = LitSink(root_dir=tmp_path / "lit", max_bytes=MB, module=FakeLit())
    serve(opts, sink, litlogger_interval=60, git_interval=1800, stop=stop, clock=clock)
    # t=0 both; t=60 lit; t=120 lit; t=130 run ended -> both; t=190 lit; stop -> both
    assert calls == ["lit", "git", "lit", "lit", "lit", "git", "lit", "lit", "git"]


def test_serve_once_runs_both(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(logsync, "litlogger_cycle", lambda opts, sink: calls.append("lit"))
    monkeypatch.setattr(logsync, "repo_cycle", lambda opts: calls.append("git"))
    opts = node_opts(tmp_path / "runs", tmp_path / "evals", tmp_path)
    serve(opts, LitSink(root_dir=tmp_path / "lit", max_bytes=MB, module=FakeLit()), once=True)
    serve(opts, None, once=True)
    assert calls == ["lit", "git", "git"]


def test_count_runs():
    paths = [
        "logs/runs/a/status.json",
        "logs/evals/a/step-00050.jsonl",
        "logs/evals/b/x.jsonl",
        "logs/runs/_coordinator/n.jsonl",
    ]
    assert count_runs(paths) == 2


def test_cli_once_no_git_no_litlogger(tmp_path, monkeypatch):
    make_tree(tmp_path / "node")
    cfg = tmp_path / "alloc.yaml"
    cfg.write_text(f"paths:\n  workdir: {tmp_path / 'node'}\n  runs: runs\n  evals: evals\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(logsync.signal, "signal", lambda *args: None)  # keep pytest's own handlers
    rc = logsync.main(
        ["--node", "n1", "--once", "--no-git", "--no-litlogger", "--config", str(cfg), "--repo", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "logs" / "evals" / "2b-leak-s1" / "step-00050.jsonl").is_file()


def test_non_run_folders_under_runs_are_ignored(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "evals"
    (runs / "models" / "Qwen3.5-2B-text").mkdir(parents=True)  # model export
    (runs / "setup").mkdir()
    (runs / "setup" / "setup.attempt1.log").write_text("x")
    (runs / "2b-rob-s1").mkdir()
    (runs / "2b-rob-s1" / "status.json").write_text("{}")
    (evals / "base-2b").mkdir(parents=True)
    assert logsync.list_run_ids(runs, evals) == ["2b-rob-s1", "base-2b"]
