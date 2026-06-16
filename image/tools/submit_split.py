#!/usr/bin/env python3
"""Drop-in ``submit`` for a split network/GPU topology.

Use this when the agent (and the only network access) live on one machine while
the H100 GPUs live in an air-gapped container, reachable over ssh, with NO
shared filesystem. It glues the two halves together:

  1. runs the comparability judge LOCALLY via ``submit_judge.py`` (needs network
     + codex credentials),
  2. if comparable, rsyncs the candidate directory into the GPU container and
     runs the timed training runs THERE via ``submit_runs.py`` (needs GPUs, no
     network),
  3. combines both into the same JSON verdict the original ``submit.py`` prints,
     and exits 0 only when the candidate is comparable AND the p-value is met.

Install it as the agent's ``submit`` on the networked machine, e.g.:
    ln -s "$PWD/image/tools/submit_split.py" /usr/local/bin/submit

Configure the GPU side via environment variables:
  BENCHMARK_REMOTE         ssh target for the GPU container (REQUIRED), e.g.
                           "qzssh3", "user@host", or an ssh config alias.
  BENCHMARK_REMOTE_WS      remote workspace root.  Default: /workspace
  BENCHMARK_REMOTE_TOOLS   remote tools dir.       Default: /opt/nanogpt/tools
  BENCHMARK_REMOTE_PYTHON  remote python.          Default: /opt/venv/bin/python3
  BENCHMARK_SSH            ssh command.            Default: "ssh"

Code is shipped to the container with tar-over-ssh (no rsync needed on either
end). The container must already have ``<BENCHMARK_REMOTE_WS>/data/fineweb10B``
(e.g. a symlink to the cluster's FineWeb10B shards) and a copy of
``submit_runs.py`` plus the other tool modules under ``BENCHMARK_REMOTE_TOOLS``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGE_RE = re.compile(r"^SUBMIT_JUDGE_JSON (.*)$", re.M)
RUNS_RE = re.compile(r"^SUBMIT_RUNS_JSON (.*)$", re.M)


def extract(pattern: re.Pattern[str], text: str, what: str) -> dict:
    """Pull the marked JSON line out of a child process's stdout."""

    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"could not parse {what} JSON from output:\n{text}")
    return json.loads(match.group(1))


def run_local_judge(directory: Path) -> dict:
    """Run the comparability judge on this (networked) machine."""

    cmd = [sys.executable, str(HERE / "submit_judge.py"), str(directory)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stderr.write(proc.stderr)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return extract(JUDGE_RE, proc.stdout, "judge")


def run_remote_runs(directory: Path, runs: int) -> dict:
    """Ship the candidate to the GPU container and run the timed training runs."""

    remote = os.environ.get("BENCHMARK_REMOTE")
    if not remote:
        raise SystemExit("BENCHMARK_REMOTE must be set to the GPU container ssh target")
    remote_ws = os.environ.get("BENCHMARK_REMOTE_WS", "/workspace")
    remote_tools = os.environ.get("BENCHMARK_REMOTE_TOOLS", "/opt/nanogpt/tools")
    remote_python = os.environ.get("BENCHMARK_REMOTE_PYTHON", "/opt/venv/bin/python3")
    ssh = shlex.split(os.environ.get("BENCHMARK_SSH", "ssh"))

    dest_dir = f"{remote_ws}/submissions/{directory.name}"

    # Ship the candidate to the container via tar-over-ssh (no rsync required).
    # Clear the destination first so the GPU side runs EXACTLY the code the judge
    # saw -- a stale leftover file from a previous push must not linger. dest_dir
    # is always a fresh leaf under <ws>/submissions, so wiping it is safe.
    subprocess.run(
        ssh + [remote, f"rm -rf {shlex.quote(dest_dir)} && mkdir -p {shlex.quote(dest_dir)}"],
        check=True,
    )
    tar = subprocess.Popen(
        [
            "tar", "cf", "-",
            "--exclude", "data", "--exclude", ".git",
            "--exclude", "__pycache__", "--exclude", "*.pyc",
            "-C", str(directory), ".",
        ],
        stdout=subprocess.PIPE,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
    )
    untar = subprocess.Popen(
        ssh + [remote, f"tar --warning=no-unknown-keyword -xf - -C {shlex.quote(dest_dir)}"],
        stdin=tar.stdout,
    )
    if tar.stdout is not None:
        tar.stdout.close()
    untar.communicate()
    if tar.wait() != 0 or untar.returncode != 0:
        raise SystemExit("failed to ship candidate to the GPU container")

    remote_cmd = (
        f"{shlex.quote(remote_python)} "
        f"{shlex.quote(remote_tools + '/submit_runs.py')} "
        f"{shlex.quote(dest_dir)} --runs {int(runs)}"
    )
    proc = subprocess.run(ssh + [remote, remote_cmd], text=True, capture_output=True)
    sys.stderr.write(proc.stderr)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return extract(RUNS_RE, proc.stdout, "runs")


def main(argv: list[str] | None = None) -> int:
    """Validate a candidate across the network/GPU split and print the verdict."""

    built = argparse.ArgumentParser(
        description="Split network/GPU submit (judge local, training runs remote).",
    )
    built.add_argument("directory", type=Path, help="Candidate submission directory.")
    built.add_argument("--runs", type=int, default=10, help="Number of training runs. Default: 10.")
    args = built.parse_args(argv)

    directory = args.directory.expanduser().resolve()
    assert directory.is_dir(), f"directory does not exist: {directory}"

    judge = run_local_judge(directory)
    if judge["cheating"]:
        verdict = {
            "valid": False,
            "validity_summary": judge["summary"],
            "p_value": None,
            "p_value_met": False,
            "avg_train_time_ms": None,
            "total_runs": 0,
            "successful_runs": 0,
            "val_losses": [],
            "train_times_ms": [],
        }
        print(json.dumps(verdict, indent=2))
        return 1

    runs = run_remote_runs(directory, args.runs)
    verdict = {
        "valid": True,
        "validity_summary": judge["summary"],
        "p_value": runs["p_value"],
        "p_value_met": runs["p_value_met"],
        "avg_train_time_ms": runs["avg_train_time_ms"],
        "total_runs": runs["total_runs"],
        "successful_runs": runs["successful_runs"],
        "val_losses": runs["val_losses"],
        "train_times_ms": runs["train_times_ms"],
    }
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["valid"] and verdict["p_value_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
