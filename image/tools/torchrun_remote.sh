#!/usr/bin/env bash
# Networked-machine -> GPU-container bridge for `torchrun`.
#
# Use this in a split topology: the agent (and network) live on one machine that
# has NO GPUs, while the H100s live in an air-gapped container reachable only over
# ssh, with NO shared filesystem AND no rsync inside the container. Install this
# script as `torchrun`, earlier on PATH than the real one, so the agent's
# `torchrun ...` (and `bash run.sh`, which calls torchrun internally) transparently
# run on the GPUs:
#
#   mkdir -p ~/bin && ln -sf "$PWD/image/tools/torchrun_remote.sh" ~/bin/torchrun
#   export PATH="$HOME/bin:$PATH"
#
# Mechanism: ship code up via tar-over-ssh (overlay, no delete), run the real
# torchrun on the GPUs with stdout/stderr streamed back live, then pull produced
# files back via tar-over-ssh. Only tar + ssh are needed on both ends.
#
# Config (shared with submit_split.py):
#   BENCHMARK_REMOTE          ssh target for the GPU container (REQUIRED)
#   BENCHMARK_WORKSPACE       local workspace root.   Default: /workspace
#   BENCHMARK_REMOTE_WS       remote workspace root.  Default: /workspace
#   BENCHMARK_REMOTE_TORCHRUN remote torchrun path.   Default: /opt/venv/bin/torchrun
#   BENCHMARK_SSH             ssh command.            Default: "ssh"
#
# The container must already have <BENCHMARK_REMOTE_WS>/data/fineweb10B (e.g. a
# symlink to the cluster's FineWeb10B shards).
set -euo pipefail

remote="${BENCHMARK_REMOTE:-}"
if [[ -z "$remote" ]]; then
  echo "torchrun(remote): BENCHMARK_REMOTE must be set to the GPU container ssh target" >&2
  exit 2
fi
local_ws="${BENCHMARK_WORKSPACE:-/workspace}"
remote_ws="${BENCHMARK_REMOTE_WS:-/workspace}"
remote_torchrun="${BENCHMARK_REMOTE_TORCHRUN:-/opt/venv/bin/torchrun}"
read -r -a ssh_cmd <<<"${BENCHMARK_SSH:-ssh}"

# tar exclusions and macOS-quiet flags
TAR_EXCLUDES=(--exclude data --exclude .git --exclude __pycache__ --exclude '*.pyc')
export COPYFILE_DISABLE=1   # stop macOS bsdtar from emitting ._* AppleDouble files

# Map the local cwd to the same relative path under the remote workspace.
rel="$(python3 -c 'import os,sys; print(os.path.relpath(os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])))' "$PWD" "$local_ws")"
if [[ "$rel" == ".."* ]]; then
  echo "torchrun(remote): cwd $PWD is outside BENCHMARK_WORKSPACE ($local_ws); cannot map to the container" >&2
  exit 2
fi
if [[ "$rel" == "." ]]; then
  remote_dir="$remote_ws"
else
  remote_dir="$remote_ws/$rel"
fi

# --- push code up ------------------------------------------------------------
# Clear the destination first so removed/renamed files don't linger remotely and
# cause the GPU to run stale code. Only do this for a leaf dir under the
# workspace; at the workspace root ($rel == ".") fall back to overlay so we never
# wipe sibling dirs like data/ or submissions/.
if [[ "$rel" == "." ]]; then
  # Workspace root: remove stale top-level entries (files, dirs AND symlinks) so
  # deleted/renamed root code -- helper modules, package dirs -- can't linger
  # remotely and make the GPU import stale code. Preserve ONLY the data symlink,
  # which points at the shared shards and is intentionally NOT re-pushed below
  # (linking is skipped at $rel == "."). Everything else is recreated from the
  # tar push that follows, so wiping it first is safe.
  "${ssh_cmd[@]}" "$remote" \
    "mkdir -p $(printf '%q' "$remote_dir") && find $(printf '%q' "$remote_dir") -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf {} +"
else
  "${ssh_cmd[@]}" "$remote" "rm -rf $(printf '%q' "$remote_dir") && mkdir -p $(printf '%q' "$remote_dir")"
fi
tar cf - "${TAR_EXCLUDES[@]}" -C "$PWD" . \
  | "${ssh_cmd[@]}" "$remote" "tar --warning=no-unknown-keyword -xf - -C $(printf '%q' "$remote_dir")"
# Link data in for a leaf dir. At the workspace root remote_dir == remote_ws,
# which already has its own data symlink (created at setup) -- linking it to
# itself would break it, so skip.
if [[ "$rel" != "." ]]; then
  "${ssh_cmd[@]}" "$remote" \
    "ln -sfn $(printf '%q' "$remote_ws/data") $(printf '%q' "$remote_dir/data")"
fi

# --- run on the GPUs, forwarding args verbatim, streaming output live ---------
remote_args=""
for arg in "$@"; do
  remote_args+=" $(printf '%q' "$arg")"
done
remote_cmd="cd $(printf '%q' "$remote_dir") && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(printf '%q' "$remote_torchrun")$remote_args"

status=0
"${ssh_cmd[@]}" "$remote" "$remote_cmd" || status=$?

# --- pull produced files back (tar overlay; container GNU tar -> local tar) ----
"${ssh_cmd[@]}" "$remote" "tar cf - ${TAR_EXCLUDES[*]} -C $(printf '%q' "$remote_dir") ." \
  | tar xf - -C "$PWD"

exit "$status"
