#!/usr/bin/env bash
# In-container bootstrap for reproducing NanoGPT-Bench WITHOUT the host-side driver.
#
# Use this when you are ALREADY inside a container built from image/Dockerfile
# (so torch / triton / flash_attn_3 are present) and cannot run docker-in-docker.
# It replicates what nanogpt/driver.py does on the host:
#   1. stage()    -> materialize /workspace from the 2025-09-03_FA3 record,
#                    create the data/home/logs/submissions/experiments dirs,
#                    git init, and link in the FineWeb10B shards.
#   2. contract() -> export the fixed BENCHMARK_* environment contract.
# Then it invokes the chosen agent's run.sh directly.
#
# Normal flow (host):   bash nanogpt/run/claude_local.sh   -> driver.py -> docker run -> run.sh
# This script (in pod): bash nanogpt/run/in_container_bootstrap.sh -> run.sh
#
# Example:
#   export ANTHROPIC_API_KEY=...
#   AGENT=claude \
#   DATA_SRC=/inspire/hdd/project/aisystem-and-infra/26010/modded-nanogpt-agent/data/fineweb10B \
#   BENCHMARK_SESSION_HOURS=64 \
#   bash nanogpt/run/in_container_bootstrap.sh
set -euo pipefail

# --- Resolve repo layout from this script's location -------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd "$script_dir/.." && pwd)"          # nanogpt/
repo_dir="$(cd "$bench_dir/.." && pwd)"            # repo root

# --- Tunables (override via env) ---------------------------------------------
AGENT="${AGENT:-claude}"                            # claude | codex | autoresearch
RECORD="${RECORD:-2025-09-03_FA3}"                  # starting human record (comparability anchor)
WS="${BENCHMARK_WORKSPACE:-/workspace}"             # in-container workspace
DATA_SRC="${DATA_SRC:-/inspire/hdd/project/aisystem-and-infra/26010/modded-nanogpt-agent/data/fineweb10B}"
# 512 H100-hours / 8 GPUs = 64 wall-clock hours. 0 = single non-resuming attempt.
export BENCHMARK_SESSION_HOURS="${BENCHMARK_SESSION_HOURS:-64}"

record_dir="$repo_dir/human_baselines/$RECORD"
shared_prompts="$bench_dir/prompts"
agent_dir="$bench_dir/agents/$AGENT"

# --- Sanity checks -----------------------------------------------------------
[[ -d "$record_dir" ]]   || { echo "record not found: $record_dir" >&2; exit 1; }
[[ -d "$agent_dir" ]]    || { echo "unknown AGENT '$AGENT' ($agent_dir not found)" >&2; exit 1; }
[[ -d "$DATA_SRC" ]]     || { echo "FineWeb10B data not visible in container: $DATA_SRC" >&2
                              echo "  -> bind-mount it (restart container with -v) or fix DATA_SRC." >&2; exit 1; }
command -v npm >/dev/null || { echo "npm/node not found; agents/$AGENT/install.sh needs it." >&2
                               echo "  -> curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs" >&2; exit 1; }
if [[ ! -x /usr/local/bin/submit || ! -d /opt/nanogpt/tools ]]; then
  echo "WARN: submit validator missing (/usr/local/bin/submit or /opt/nanogpt/tools)." >&2
  echo "      Agent's 'submit' calls will fail. Install with:" >&2
  echo "        cp -r $repo_dir/image/tools /opt/nanogpt/tools" >&2
  echo "        install -m755 $repo_dir/image/submit.sh /usr/local/bin/submit" >&2
fi

# --- stage(): replicate driver.stage ----------------------------------------
echo ">> staging workspace at $WS from record $RECORD"
rm -rf "$WS"
mkdir -p "$WS"
cp -a "$record_dir"/. "$WS"/                        # train_gpt.py + run.sh as the starting point
mkdir -p "$WS"/{data,home,logs,submissions,experiments}
ln -sfn "$DATA_SRC" "$WS/data/fineweb10B"           # train_gpt.py reads ./data/fineweb10B (DATA_PATH=.)
(
  cd "$WS"
  git init -q -b main
  git add -A
  git -c user.email=bench@local -c user.name=bench commit -q -m "init: $RECORD" || true
)

# --- contract(): replicate driver.contract ----------------------------------
export HOME="$WS/home"
export BENCHMARK_WORKSPACE="$WS"
export BENCHMARK_RUN_ID="local-$(date +%Y%m%d-%H%M%S)"
export BENCHMARK_AGENT="$AGENT"
export BENCHMARK_TRACE_DIR="$WS/home"
export BENCHMARK_LOG_DIR="$WS/logs"
export BENCHMARK_EXPERIMENT_DIR="$WS/experiments"
export BENCHMARK_SUBMISSION_DIR="$WS/submissions"
export BENCHMARK_EVENTS_PATH="$WS/agent_events.jsonl"
export BENCHMARK_FINAL_PATH="$WS/agent_final.txt"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$WS"

# --- per-agent prompt wiring (mirrors nanogpt/run/*_local.sh) ----------------
case "$AGENT" in
  claude|codex)
    args=(
      --problem-file="$shared_prompts/problem.txt"
      --rules-file="$shared_prompts/RULES.md"
      --prompt-file="$shared_prompts/local_prompt.md"
      --resume-file="$shared_prompts/resume_prompt.md"
    )
    ;;
  autoresearch)
    ap="$agent_dir/prompts"
    args=(
      --rules-file="$shared_prompts/RULES.md"
      --program-file="$ap/program.md"
      --prompt-file="$ap/local.md"
      --resume-file="$ap/resume.md"
    )
    ;;
  *)
    echo "no prompt wiring for AGENT '$AGENT'" >&2; exit 1 ;;
esac

echo ">> launching agent '$AGENT' (session hours: $BENCHMARK_SESSION_HOURS)"
echo "   logs:  tail -f $WS/agent_trace.txt"
echo "   subs:  ls -lt $BENCHMARK_SUBMISSION_DIR"
exec bash "$agent_dir/run.sh" "${args[@]}"
