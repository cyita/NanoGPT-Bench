#!/usr/bin/env python3
"""Network-side half of a split ``submit``: comparability judging only.

Serializes a candidate directory and asks the LLM judge (codex / gpt-5.4 via
OpenAI or Azure, same as ``submit.py``) whether the candidate stays comparable
to the baseline record. It needs network and codex access but touches NO GPU and
reads NO training data, so it can run on the networked control machine while the
GPUs sit in an air-gapped container. Pair it with ``submit_runs.py`` (GPU side)
through ``submit_split.py``.

Auth, two modes:

* **API key (default)** -- set ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` (or the
  ``AZURE_*`` variables). Uses ``submit.check`` exactly like the stock submit.
* **Reuse local login** -- set ``BENCHMARK_CODEX_REUSE_LOGIN=1`` to reuse an
  existing interactive ``codex login`` (e.g. a ChatGPT-account login) instead of
  an API key. The judge copies ``auth.json`` from your real codex home
  (``$CODEX_HOME`` or ``~/.codex``) into an isolated run dir, so your real codex
  home is never written to. Override the judge model with ``BENCHMARK_CODEX_MODEL``
  and effort with ``BENCHMARK_CODEX_REASONING_EFFORT`` if your plan lacks gpt-5.4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import comparable
import overfit
import submit

# Marker prefix so submit_split.py can extract the JSON from interleaved stdout.
JSON_MARKER = "SUBMIT_JUDGE_JSON "

TRUTHY = {"1", "true", "yes", "on"}


def reuse_login_enabled() -> bool:
    """Whether to reuse an existing interactive codex login instead of an API key."""

    return os.environ.get("BENCHMARK_CODEX_REUSE_LOGIN", "").strip().lower() in TRUTHY


def codex_home() -> Path:
    """Resolve the user's real codex home (where ``codex login`` stored auth).

    Prefers ``BENCHMARK_JUDGE_CODEX_HOME`` because the agent harness repoints
    ``HOME`` (and may set ``CODEX_HOME``) at the run workspace, which would
    otherwise hide the interactive ``~/.codex`` login.
    """

    raw = (
        os.environ.get("BENCHMARK_JUDGE_CODEX_HOME")
        or os.environ.get("CODEX_HOME")
    )
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def run_codex_reuse(prompt_text: str, model: str, reasoning: str) -> str:
    """Run the codex judge reusing an existing interactive login.

    Mirrors ``comparable.run_codex`` but, instead of ``codex login --with-api-key``,
    copies the existing ``auth.json`` into an isolated, ephemeral codex home. The
    real codex home is only read, never modified.
    """

    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError("codex is not installed or not on PATH")

    auth_src = codex_home() / "auth.json"
    if not auth_src.is_file():
        raise RuntimeError(
            f"no codex login found at {auth_src}; run `codex login` first "
            "(or unset BENCHMARK_CODEX_REUSE_LOGIN and provide OPENAI_API_KEY)"
        )

    with tempfile.TemporaryDirectory(prefix="submit-judge-") as tmp:
        root = Path(tmp)
        home = root / "home"
        run_codex_home = home / ".codex"
        work = root / "work"
        reply_path = root / "reply.json"
        schema_path = root / "schema.json"
        run_codex_home.mkdir(parents=True)
        work.mkdir()
        # Bring just the credential over; model/effort come from our config.toml.
        shutil.copy2(auth_src, run_codex_home / "auth.json")
        (run_codex_home / "config.toml").write_text(
            f'model = "{model}"\nmodel_reasoning_effort = "{reasoning}"\n',
            encoding="utf-8",
        )
        schema_path.write_text(comparable.schema(), encoding="utf-8")

        env = os.environ.copy()
        env["HOME"] = str(home)
        env["CODEX_HOME"] = str(run_codex_home)
        # Force the reused login to be used rather than any stray API key.
        env.pop("OPENAI_API_KEY", None)
        env.pop("CODEX_API_KEY", None)

        completed = subprocess.run(
            (
                executable,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "-C",
                str(work),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(reply_path),
                "--color",
                "never",
                "-c",
                'web_search="disabled"',
                "-",
            ),
            input=prompt_text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            cwd=work,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "codex exec failed"
            raise RuntimeError(message)
        if not reply_path.is_file():
            raise RuntimeError("codex exec did not write the final output file")
        return reply_path.read_text(encoding="utf-8")


def check_reuse(directory: Path) -> comparable.Decision:
    """Comparability check using the reused interactive codex login."""

    model = os.environ.get("BENCHMARK_CODEX_MODEL") or overfit.DEFAULT_MODEL
    reasoning = os.environ.get("BENCHMARK_CODEX_REASONING_EFFORT") or overfit.DEFAULT_REASONING
    candidate_text = overfit.serialize(directory)
    prompt_text = submit.prompt(directory.name, candidate_text)
    raw = run_codex_reuse(prompt_text, model, reasoning)
    return comparable.reply(directory.name, raw)


def main(argv: list[str] | None = None) -> int:
    """Run the comparability-judge half of submit and emit a JSON result."""

    built = argparse.ArgumentParser(
        description="Run the comparability-judge half of submit (network only, no GPU).",
    )
    built.add_argument("directory", type=Path, help="Candidate submission directory.")
    args = built.parse_args(argv)

    directory = args.directory.expanduser().resolve()
    assert directory.is_dir(), f"directory does not exist: {directory}"

    print(f"checking validity of {directory.name}...", flush=True)
    if reuse_login_enabled():
        print("  (reusing local codex login)", flush=True)
        decision = check_reuse(directory)
    else:
        decision = submit.check(directory)
    print(
        f"  cheating={decision.cheating}"
        f" confidence={decision.confidence}"
        f" summary={decision.summary}",
        flush=True,
    )

    result = {
        "cheating": decision.cheating,
        "confidence": decision.confidence,
        "summary": decision.summary,
        "reasons": list(decision.reasons),
    }
    print(JSON_MARKER + json.dumps(result), flush=True)
    return 1 if decision.cheating else 0


if __name__ == "__main__":
    raise SystemExit(main())
