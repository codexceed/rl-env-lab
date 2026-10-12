"""eval/sametype_batch.py without a GPU: job planning, the heldout command, GPU scheduling, timeouts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from eval import sametype_batch as sb
from eval.heldout import parse_args
from train.checkpoints import COMPLETE, atomic_write_json, eval_step_dir
from train.run_manifest import MANIFEST_FILE


def eval_step(run: Path, step: int, complete: bool = True, **marker) -> Path:
    d = eval_step_dir(run, step)
    d.mkdir(parents=True)
    if complete:
        atomic_write_json(d / COMPLETE, {"step": step, "final": False, **marker})
    return d


@pytest.fixture
def tree(tmp_path):
    runs, evals = tmp_path / "runs", tmp_path / "evals"
    rob = runs / "2b-rob-s1"
    eval_step(rob, 50)
    eval_step(rob, 100)
    eval_step(rob, 150, complete=False)  # still being written: ignored
    atomic_write_json(rob / MANIFEST_FILE, {"size": "2b", "reward": "robust", "seed": 1})
    leak = runs / "0.8b-leak-s2"  # no manifest: metadata from the COMPLETE marker
    eval_step(leak, 50, size="0.8b", reward="leaky", seed=2)
    eval_step(runs / "dry-2b-rob", 5, size="2b", reward="robust", seed=1)  # not a matrix id
    (runs / "_coordinator").mkdir()
    (runs / "2b-leak-s3").mkdir()  # no eval checkpoints yet
    done = evals / "2b-rob-s1" / "sametype" / "step-00050.jsonl"
    done.parent.mkdir(parents=True)
    done.write_text("{}\n")
    (evals / "2b-rob-s1" / "step-00100.jsonl").write_text("{}\n")  # a formal output does not count
    return runs, evals


def test_plan_takes_complete_checkpoints_plus_bases_and_skips_existing(tree):
    runs, evals = tree
    jobs, skipped, notes = sb.plan_jobs(runs, evals)
    assert [j.label for j in jobs] == ["base-0.8b@0", "base-2b@0", "0.8b-leak-s2@50", "2b-rob-s1@100"]
    assert [j.label for j in skipped] == ["2b-rob-s1@50"]
    assert notes == []
    by = {j.label: j for j in jobs}
    assert by["2b-rob-s1@100"].output == evals / "2b-rob-s1" / "sametype" / "step-00100.jsonl"
    assert by["2b-rob-s1@100"].checkpoint == eval_step_dir(runs / "2b-rob-s1", 100)
    assert (by["0.8b-leak-s2@50"].size, by["0.8b-leak-s2@50"].reward, by["0.8b-leak-s2@50"].seed) == (
        "0.8b",
        "leaky",
        2,
    )
    assert by["base-2b@0"].checkpoint is None and by["base-2b@0"].output == evals / "base-2b/sametype/step-00000.jsonl"


def test_runs_filter(tree):
    runs, evals = tree
    jobs, _, _ = sb.plan_jobs(runs, evals, ["0.8b-*"])
    assert [j.label for j in jobs] == ["base-0.8b@0", "0.8b-leak-s2@50"]
    jobs, _, _ = sb.plan_jobs(runs, evals, ["dry-*"])
    assert [j.label for j in jobs] == ["base-2b@0", "dry-2b-rob@5"]


def test_metadata_sources(tmp_path):
    run = tmp_path / "4b-leak-s3"
    eval_step(run, 50)
    assert sb.run_metadata(run) == {"size": "4b", "reward": "leaky", "seed": 3}  # from the id
    atomic_write_json(run / "status.json", {"size": "4b", "reward": "robust", "seed": 9})
    assert sb.run_metadata(run) == {"size": "4b", "reward": "robust", "seed": 9}  # status.json before the id
    atomic_write_json(run / MANIFEST_FILE, {"size": "4b", "reward": "leaky", "seed": 3})
    assert sb.run_metadata(run) == {"size": "4b", "reward": "leaky", "seed": 3}  # the manifest first
    odd = tmp_path / "mystery"
    eval_step(odd, 50)
    assert sb.run_metadata(odd) is None


def test_job_argv_is_a_valid_heldout_sametype_command(tree):
    runs, evals = tree
    jobs, _, _ = sb.plan_jobs(runs, evals)
    for job in jobs:
        argv = job.argv(5, python="py", extra=["--evals-root", str(evals)])
        assert argv[:3] == ["py", "-u", str(sb.HELDOUT)]
        args = parse_args(argv[3:])
        assert (args.mode, args.gpus, args.output, args.thinking) == ("sametype", "5", job.output, None)
        if job.checkpoint is None:
            assert args.checkpoint == "base" and args.run_id is None
        else:
            assert (args.run_id, args.reward, args.seed, args.step) == (job.run_id, job.reward, job.seed, str(job.step))


class FakeProc:
    def __init__(self, pid: int, polls_until_exit: int | None, rc: int = 0, on_exit=None):
        self.pid = pid
        self.left = polls_until_exit
        self.rc = rc
        self.stopped = False
        self.on_exit = on_exit

    def poll(self):
        if self.stopped:
            return -15
        if self.left is None:
            return None
        self.left -= 1
        if self.left > 0:
            return None
        if self.on_exit:
            self.on_exit()
            self.on_exit = None
        return self.rc

    def wait(self, timeout=None):
        return -15 if self.stopped else self.rc


def _jobs(tmp_path: Path, n: int) -> list[sb.Job]:
    return [
        sb.Job(f"2b-rob-s{i}", "2b", "robust", i, 50, tmp_path / "ck", tmp_path / f"out{i}/step-00050.jsonl")
        for i in range(n)
    ]


def test_run_jobs_one_per_gpu_only_listed_gpus_and_progress(tmp_path):
    jobs = _jobs(tmp_path, 5)
    launched: list[tuple[str, int]] = []
    busy: set[int] = set()

    def launch(job, gpu):
        assert gpu in (3, 6) and gpu not in busy  # never a GPU outside --gpus, never two on one GPU
        busy.add(gpu)
        launched.append((job.label, gpu))
        rc = 1 if job.run_id == "2b-rob-s2" else 0
        if rc == 0:
            job.output.parent.mkdir(parents=True)
            job.output.write_text("{}\n")
        return FakeProc(len(launched), polls_until_exit=2, rc=rc, on_exit=lambda: busy.discard(gpu))

    lines: list[str] = []
    results = sb.run_jobs(jobs, [3, 6], launch=launch, sleep=lambda s: None, emit=lines.append, clock=lambda: 0.0)
    assert len(launched) == 5 and {g for _, g in launched} == {3, 6}
    assert [ln for ln in lines if ln.startswith("\nPROGRESS")] == [f"\nPROGRESS {k}/5" for k in range(1, 6)]
    assert [r.ok for r in sorted(results, key=lambda r: r.job.run_id)] == [True, True, False, True, True]
    s = sb.summary(results, skipped=jobs[:1])
    assert (s["total"], s["ok"], s["failed"], s["timeout"], s["skipped_existing"]) == (5, 4, ["2b-rob-s2@50"], [], 1)


def test_run_jobs_kills_a_job_past_its_timeout(tmp_path):
    jobs = _jobs(tmp_path, 2)
    now = [0.0]
    stopped: list[int] = []

    def stop(proc):
        proc.stopped = True
        stopped.append(proc.pid)
        return proc.wait()

    def launch(job, gpu):
        return FakeProc(int(job.run_id[-1]), polls_until_exit=None)  # hangs forever

    def sleep(s):
        now[0] += s

    results = sb.run_jobs(
        jobs, [0], timeout_s=100, launch=launch, stop=stop, clock=lambda: now[0], sleep=sleep, emit=lambda _: None
    )
    assert stopped == [0, 1]
    assert all(r.timed_out and not r.ok for r in results)
    assert sb.summary(results, [])["timeout"] == ["2b-rob-s0@50", "2b-rob-s1@50"]


def test_run_jobs_stops_running_evals_when_interrupted(tmp_path):
    stopped = []

    def sleep(_):
        raise KeyboardInterrupt

    def stop(proc):
        stopped.append(proc.pid)
        return 0

    with pytest.raises(KeyboardInterrupt):
        sb.run_jobs(_jobs(tmp_path, 3), [0, 1], launch=lambda j, g: FakeProc(g, None), stop=stop, sleep=sleep)
    assert sorted(stopped) == [0, 1]


def test_launch_pins_the_job_to_one_gpu(tmp_path, monkeypatch):
    seen = {}

    class Popen:
        def __init__(self, argv, **kw):
            seen.update(kw, argv=argv)
            self.pid = 1

    monkeypatch.setattr(subprocess, "Popen", Popen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    job = _jobs(tmp_path, 1)[0]
    sb.launch_subprocess(job, 6)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "6"
    assert seen["argv"][seen["argv"].index("--gpus") + 1] == "6"
    assert seen["start_new_session"] and seen["cwd"] == sb.REPO_ROOT
    assert sb.log_path(job).is_file() and sb.log_path(job).parent == job.output.parent / "logs"


def test_main_dry_run_and_missing_data(tree, tmp_path, capsys):
    runs, evals = tree
    base = ["--gpus", "4,5", "--runs-root", str(runs), "--evals-root", str(evals), "--data-dir", str(tmp_path / "nd")]
    assert sb.main([*base, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "4 to run, 1 already done" in out and "2b-rob-s1@100" in out
    assert sb.main(base) == 2
    assert "PROGRESS_EXIT 2" in capsys.readouterr().out


def test_main_runs_and_reports(tree, tmp_path, monkeypatch, capsys):
    runs, evals = tree
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sametype.jsonl").write_text("")

    def fake_run_jobs(jobs, gpus, **kw):
        assert gpus == [4, 5]
        return [sb.Result(j, 4, 0, False, 1.0) for j in jobs]

    monkeypatch.setattr(sb, "run_jobs", fake_run_jobs)
    argv = ["--gpus", "4,5", "--runs-root", str(runs), "--evals-root", str(evals), "--data-dir", str(tmp_path / "data")]
    assert sb.main(argv) == 1  # the fake wrote no outputs, so nothing counts as ok
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "PROGRESS_EXIT 1"
    report = json.loads(lines[-2])
    assert (report["total"], report["ok"], report["skipped_existing"]) == (4, 0, 1)


@pytest.mark.parametrize("bad", ["", "4,4", "-1", "a"])
def test_gpus_must_be_distinct_indices(bad):
    with pytest.raises(SystemExit):
        sb.build_parser().parse_args(["--gpus", bad])


STUB = """
import sys, time
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--output") + 1])
if "hang" in str(out):
    time.sleep(60)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("{}\\n")
"""


def test_real_subprocesses_finish_or_are_killed(tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setattr(sb, "HELDOUT", stub)
    ok = sb.Job("2b-rob-s1", "2b", "robust", 1, 50, tmp_path, tmp_path / "ok" / "sametype" / "step-00050.jsonl")
    hang = sb.Job("2b-rob-s2", "2b", "robust", 2, 50, tmp_path, tmp_path / "hang" / "sametype" / "step-00050.jsonl")
    results = sb.run_jobs([ok, hang], [0, 1], timeout_s=1.0, poll_s=0.05, emit=lambda _: None)
    by = {r.job.run_id: r for r in results}
    assert by["2b-rob-s1"].ok and ok.output.read_text() == "{}\n"
    assert by["2b-rob-s2"].timed_out and not hang.output.exists()
    assert sb.log_path(ok).is_file() and sb.log_path(hang).is_file()  # logs kept, nothing deleted
