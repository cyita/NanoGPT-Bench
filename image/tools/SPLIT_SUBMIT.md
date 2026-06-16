# Split `submit` for an air-gapped GPU container

The stock `submit` (`submit.py`) does two things in one process:

1. **Comparability judge** — serializes the candidate code and asks an LLM judge
   (codex / `gpt-5.4` via OpenAI or Azure) whether it stays comparable to the
   baseline. **Needs network**, no GPU.
2. **Timed runs** — runs the candidate's `run.sh` 10× and computes the one-sided
   p-value for `val_loss < 3.28`. **Needs GPUs**, no network.

When the agent + network live on one machine and the H100s live in an
**air-gapped container** (no outbound net, no shared filesystem, reachable only
over ssh), those two steps must run in different places. These three scripts
split them:

| Script | Runs on | Needs |
| --- | --- | --- |
| `submit_judge.py <dir>` | networked machine | `codex` CLI + `OPENAI_API_KEY`/`CODEX_API_KEY` (or `AZURE_*`) |
| `submit_runs.py <dir> --runs 10` | GPU container | 8× H100, `/opt/venv` training stack, `<ws>/data/fineweb10B` |
| `submit_split.py <dir>` | networked machine | orchestrates the two; replaces `submit` |

`submit_split.py` runs the judge locally, and only if the candidate is
comparable, ships the candidate into the container (tar-over-ssh, no rsync
needed) and runs `submit_runs.py` there over ssh, then prints the **same JSON
verdict** as the original `submit.py` (exit 0 only when comparable AND p-value
met).

## Setup

On the **GPU container** — make `submit_runs.py` and its sibling modules
available, and link in the data once:

```bash
# tool modules (submit.py, pvalue.py, overfit.py, comparable.py, submit_runs.py)
cp image/tools/*.py /opt/nanogpt/tools/         # or rebuild the image
ln -sfn /inspire/.../data/fineweb10B /workspace/data/fineweb10B
```

On the **networked machine** — install the orchestrator as `submit` and point it
at the container:

```bash
ln -sf "$PWD/image/tools/submit_split.py" /usr/local/bin/submit

export BENCHMARK_REMOTE=qzssh3            # ssh target for the GPU container (required)
export BENCHMARK_REMOTE_WS=/workspace     # default
export BENCHMARK_REMOTE_TOOLS=/opt/nanogpt/tools   # default
export BENCHMARK_REMOTE_PYTHON=/opt/venv/bin/python3  # default
```

Then pick how the judge authenticates:

```bash
# Option A -- API key:
export OPENAI_API_KEY=...                 # or CODEX_API_KEY / AZURE_*

# Option B -- reuse an existing interactive `codex login` (e.g. ChatGPT account):
export BENCHMARK_CODEX_REUSE_LOGIN=1      # copies ~/.codex/auth.json into an isolated run dir
# export BENCHMARK_CODEX_MODEL=...        # optional: override gpt-5.4 if your plan lacks it
# export CODEX_HOME=/custom/.codex        # optional: if your login lives elsewhere
```

Then the agent calls `submit /path/to/submission_N` exactly as before.

## `torchrun` bridge (single training runs)

`submit` is the low-frequency GPU entry point; the high-frequency one is
`torchrun` (and `bash run.sh`, which calls it). The agent runs single training
runs constantly while iterating, but the networked machine has no GPUs. Install
`torchrun_remote.sh` as `torchrun`, earlier on PATH than the real one, so those
runs transparently execute in the container:

```bash
mkdir -p ~/bin && ln -sf "$PWD/image/tools/torchrun_remote.sh" ~/bin/torchrun
export PATH="$HOME/bin:$PATH"
# reuses the same vars as submit_split, plus the local workspace root:
export BENCHMARK_REMOTE=qzssh3
export BENCHMARK_WORKSPACE=/workspace      # local workspace root (default /workspace)
export BENCHMARK_REMOTE_WS=/workspace      # remote workspace root (default /workspace)
```

It maps the local cwd to the same relative path under `BENCHMARK_REMOTE_WS`,
ships the code up via tar-over-ssh (excluding `data`/`.git`), links `<dir>/data ->
<remote_ws>/data`, runs `/opt/venv/bin/torchrun "$@"` on the GPUs with live
output, then pulls produced files back. Override the remote binary with
`BENCHMARK_REMOTE_TORCHRUN` if needed.

With both bridges in place, the two GPU-touching commands the RULES allow
(`torchrun` and `submit`) work from the networked machine unchanged.

### ssh connection reuse (important)

Each `torchrun` makes several ssh calls and the agent runs hundreds of them, so
enable connection multiplexing in `~/.ssh/config` to avoid a handshake per call:

```
Host qzssh3
    HostName localhost
    User root
    Port 2224
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

```bash
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm   # one-time
```

### Transport requirements

Only `tar` + `ssh` are needed on both ends — the container does **not** need
`rsync`. macOS `bsdtar` is handled (`COPYFILE_DISABLE=1` to suppress `._*`
files); the container's GNU tar extracts with `--warning=no-unknown-keyword`.

## Run the halves directly (for testing)

```bash
# judge only (networked machine):
python3 image/tools/submit_judge.py /path/to/submission_N

# timed runs only (inside the GPU container):
/opt/venv/bin/python3 /opt/nanogpt/tools/submit_runs.py /workspace/submissions/submission_N --runs 10
```

## Notes

- Keep the judge on `gpt-5.4`/`xhigh` to match the published baselines; swapping
  the judge model changes the "cheating" verdicts and breaks comparability with
  official numbers.
- `submit_split.py` ships **only code** (it excludes `data/` and `.git/`); the
  container resolves data through `<ws>/data/fineweb10B`, the same way `submit.py`
  does via the auto-created `<dir>/data` symlink.
- This only covers `submit`. The other GPU-touching command, `torchrun`, has its
  own networked→container bridge (`torchrun_remote.sh`, see above) to complete a
  fully split run.
