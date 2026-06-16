#!/usr/bin/env python3
"""GPU-side half of a split ``submit``: the timed training runs only.

Runs the N (default 10) training runs for a candidate directory and computes the
one-sided p-value. It performs NO comparability judging and needs NO network, so
it can run inside an air-gapped GPU container. Pair it with ``submit_judge.py``
(network side) through ``submit_split.py``.

The candidate directory must contain ``run.sh`` and ``train_gpt.py`` and live
under ``<workspace>/submissions/<name>`` so that the data symlink resolves the
same way ``submit.py`` expects (``<dir>/data -> <workspace>/data``).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import pvalue
import submit

# Marker prefix so submit_split.py can extract the JSON from interleaved stdout.
JSON_MARKER = "SUBMIT_RUNS_JSON "


def ensure_venv_on_path() -> None:
    """Put this interpreter's bin dir (the venv) on PATH so run.sh finds torchrun.

    ``run.sh`` calls a bare ``torchrun``; the image puts /opt/venv/bin on PATH, but
    non-login ssh shells (how this runs in a split topology) do not inherit it.
    """

    venv_bin = os.path.dirname(os.path.abspath(sys.executable))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if venv_bin not in parts:
        os.environ["PATH"] = os.pathsep.join([venv_bin, *parts]) if parts != [""] else venv_bin


def main(argv: list[str] | None = None) -> int:
    """Run the timed training-run half of submit and emit a JSON result."""

    built = argparse.ArgumentParser(
        description="Run the timed training-run half of submit (no judge, no network).",
    )
    built.add_argument("directory", type=Path, help="Candidate submission directory.")
    built.add_argument(
        "--runs",
        type=int,
        default=submit.NUM_RUNS,
        help=f"Number of training runs. Default: {submit.NUM_RUNS}.",
    )
    args = built.parse_args(argv)

    directory = args.directory.expanduser().resolve()
    assert directory.is_dir(), f"directory does not exist: {directory}"
    assert (directory / "run.sh").is_file(), f"missing run.sh in {directory}"
    assert (directory / "train_gpt.py").is_file(), f"missing train_gpt.py in {directory}"

    ensure_venv_on_path()
    print(f"running {args.runs} training runs for {directory.name}...", flush=True)
    runs = submit.evaluate(directory, args.runs)
    val_losses = [r.val_loss for r in runs if r.val_loss is not None]
    train_times = [r.train_time_ms for r in runs if r.train_time_ms is not None]
    p_val = pvalue.pvalue(val_losses, submit.VAL_LOSS_THRESHOLD)
    p_value_met = p_val is not None and p_val < submit.P_VALUE_THRESHOLD

    result = {
        "p_value": p_val,
        "p_value_met": p_value_met,
        "avg_train_time_ms": statistics.fmean(train_times) if train_times else None,
        "total_runs": args.runs,
        "successful_runs": len(val_losses),
        "val_losses": val_losses,
        "train_times_ms": train_times,
    }
    print(JSON_MARKER + json.dumps(result), flush=True)
    return 0 if p_value_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
