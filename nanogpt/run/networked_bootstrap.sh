#!/usr/bin/env bash
# Route B bootstrap: run the agent on a NETWORKED machine and dispatch the
# GPU-touching commands (torchrun, submit) into an air-gapped H100 container over
# ssh, with NO shared filesystem.
#
# Topology:
#   [this machine]  agent CLI + run.sh loop + edits/logs  (has network, NO GPUs)
#        | ssh + tar, only for torchrun / submit
#        v
#   [container]     torchrun training + submit's 10 timed runs  (8x H100, NO network)
#
# What it does (mirrors nanogpt/driver.py, split across the two machines):
#   1. stage()    -> local workspace from the 2025-09-03_FA3 record (fresh runs only).
#   2. bridges    -> install `torchrun` (torchrun_remote.sh) and `submit`
#                    (submit_split.py) on PATH so the agent uses the GPUs remotely.
#   3. remote     -> one-time container prep: check FineWeb10B + sync tool modules.
#   4. contract() -> export the BENCHMARK_* contract, then run agents/<agent>/run.sh.
#
# Resumable budget: BENCHMARK_TOTAL_HOURS (default 64) is a cumulative cap across
# legs, persisted in runs/<id>/budget.env. Each launch runs min(this leg, remaining)
# and adds the elapsed time to the total; downtime between legs is not counted.
# Stop a leg with Ctrl-C, then continue later with RESUME=<run_id> (or RESUME=latest),
# which reuses the same workspace, agent session, and remote tree.
#
# Example (start a 64h budget, run ~8h now):
#   export ANTHROPIC_API_KEY=...               # agent (claude)
#   export BENCHMARK_CODEX_REUSE_LOGIN=1        # judge reuses local `codex login`
#   export BENCHMARK_REMOTE=qzssh3              # ssh target for the GPU container
#   BENCHMARK_TOTAL_HOURS=64 BENCHMARK_SESSION_HOURS=8 bash nanogpt/run/networked_bootstrap.sh
# Later, after the gap, continue (runs the rest of the 64h, or another leg):
#   RESUME=latest BENCHMARK_SESSION_HOURS=8 bash nanogpt/run/networked_bootstrap.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd "$script_dir/.." && pwd)"          # nanogpt/
repo_dir="$(cd "$bench_dir/.." && pwd)"            # repo root
tools_dir="$repo_dir/image/tools"

# --- Tunables (override via env) ---------------------------------------------
AGENT="${AGENT:-claude}"                            # claude | codex | autoresearch
RECORD="${RECORD:-2025-09-03_FA3}"                  # starting human record (comparability anchor)
RESUME="${RESUME:-}"                                # "" = fresh | "latest" | a run_id = continue
# Cumulative budget across ALL legs (512 H100h / 8 GPUs = 64h). Persisted per run.
export BENCHMARK_TOTAL_HOURS="${BENCHMARK_TOTAL_HOURS:-64}"
# This leg's max duration; empty => run the full remaining budget in one sitting.
# Downtime between legs does NOT count -- only time while this script is running.
SESSION_HOURS_REQ="${BENCHMARK_SESSION_HOURS:-}"

# Reject non-numeric / non-positive hours up front (mirrors agents/*/run.sh).
# Otherwise awk silently turns "", a stray space, or garbage into 0 -- and a 0/
# negative budget means an instant "budget exhausted" or a bogus leg length.
require_pos_hours() {  # name value
  local name="$1" val="$2"
  if [[ ! "$val" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
     [[ "$(awk -v v="$val" 'BEGIN{print (v>0)?1:0}')" != "1" ]]; then
    echo "invalid $name: '$val' (must be a positive number of hours)" >&2
    exit 1
  fi
}
require_pos_hours BENCHMARK_TOTAL_HOURS "$BENCHMARK_TOTAL_HOURS"
[[ -n "$SESSION_HOURS_REQ" ]] && require_pos_hours BENCHMARK_SESSION_HOURS "$SESSION_HOURS_REQ"

# Remote (GPU container) wiring -- shared with torchrun_remote.sh / submit_split.py
export BENCHMARK_REMOTE="${BENCHMARK_REMOTE:-}"
# Each run gets its own remote tree at <base>/runs/<id> (set after run_id below),
# so concurrent / smoke vs formal runs never share files. Data is shared at
# <base>/data. BENCHMARK_REMOTE_WS may still be overridden explicitly.
export BENCHMARK_REMOTE_BASE="${BENCHMARK_REMOTE_BASE:-/workspace}"
export BENCHMARK_REMOTE_TOOLS="${BENCHMARK_REMOTE_TOOLS:-/opt/nanogpt/tools}"
export BENCHMARK_REMOTE_PYTHON="${BENCHMARK_REMOTE_PYTHON:-/opt/venv/bin/python3}"
export BENCHMARK_REMOTE_TORCHRUN="${BENCHMARK_REMOTE_TORCHRUN:-/opt/venv/bin/torchrun}"
export BENCHMARK_SSH="${BENCHMARK_SSH:-ssh}"

# Capture the real codex login location NOW, before HOME gets repointed at the
# run workspace below (and again inside run.sh). The split judge prefers this so
# BENCHMARK_CODEX_REUSE_LOGIN keeps finding your interactive ~/.codex auth.
export BENCHMARK_JUDGE_CODEX_HOME="${BENCHMARK_JUDGE_CODEX_HOME:-${CODEX_HOME:-$HOME/.codex}}"

record_dir="$repo_dir/human_baselines/$RECORD"
shared_prompts="$bench_dir/prompts"
agent_dir="$bench_dir/agents/$AGENT"
read -r -a ssh_cmd <<<"$BENCHMARK_SSH"

# --- Sanity checks -----------------------------------------------------------
[[ -n "$BENCHMARK_REMOTE" ]] || { echo "BENCHMARK_REMOTE must be set to the GPU container ssh target" >&2; exit 1; }
[[ -d "$record_dir" ]]       || { echo "record not found: $record_dir" >&2; exit 1; }
[[ -d "$agent_dir" ]]        || { echo "unknown AGENT '$AGENT' ($agent_dir not found)" >&2; exit 1; }
command -v npm >/dev/null    || { echo "npm/node not found; agents/$AGENT/install.sh needs it." >&2
                                  echo "  -> install Node 20 on this networked machine first." >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found (needed by the submit bridge)." >&2; exit 1; }

# --- resolve run dir (fresh stage vs resume) ---------------------------------
# Each run records its identity (agent/record) in run.meta so a resume can't
# silently attach to a smoke test or a run of a different agent/record. Parse it
# as DATA (never source) -- the agent can write under run_dir.
mkdir -p "$repo_dir/runs"
read_meta() { sed -n "s/^$2=\(.*\)\$/\1/p" "$1" 2>/dev/null | tail -1; }
if [[ -n "$RESUME" ]]; then
  if [[ "$RESUME" == "latest" ]]; then
    # Latest local-* whose recorded agent matches the requested AGENT, so a
    # different-agent run (or a bare dir with no meta) is never picked blindly.
    run_id=""
    while IFS= read -r cand; do
      [[ "$(read_meta "$repo_dir/runs/$cand/run.meta" agent)" == "$AGENT" ]] && run_id="$cand"
    done < <(ls -1 "$repo_dir/runs" 2>/dev/null | grep -E '^local-' | sort)
    [[ -n "$run_id" ]] || { echo "RESUME=latest: no runs/local-* with agent=$AGENT found (set RESUME=<run_id> to be explicit)" >&2; exit 1; }
  else
    run_id="$RESUME"
  fi
  run_dir="$repo_dir/runs/$run_id"
  WS="$run_dir/workspace"
  [[ -d "$WS" ]] || { echo "resume target not found: $WS" >&2; exit 1; }
  # Refuse to resume a run staged for a different agent/record (a common way to
  # corrupt a formal 512 H100-hour budget). Override with BENCHMARK_FORCE_RESUME=1.
  meta_agent="$(read_meta "$run_dir/run.meta" agent)"
  meta_record="$(read_meta "$run_dir/run.meta" record)"
  for pair in "agent:$meta_agent:$AGENT" "record:$meta_record:$RECORD"; do
    field="${pair%%:*}"; rest="${pair#*:}"; was="${rest%%:*}"; now="${rest#*:}"
    if [[ -n "$was" && "$was" != "$now" ]]; then
      echo "resume $run_id was staged with $field=$was but $field=$now now." >&2
      [[ "${BENCHMARK_FORCE_RESUME:-0}" == "1" ]] || { echo "  refusing; set BENCHMARK_FORCE_RESUME=1 to override." >&2; exit 1; }
      echo "  BENCHMARK_FORCE_RESUME=1 set; continuing anyway." >&2
    fi
  done
  echo ">> RESUME $run_id (reuse workspace + agent session + remote tree; no re-stage)"
else
  run_id="local-$(date +%Y%m%d-%H%M%S)"
  run_dir="$repo_dir/runs/$run_id"
  WS="$run_dir/workspace"
  echo ">> staging local workspace at $WS from record $RECORD"
  mkdir -p "$WS"
  cp -a "$record_dir"/. "$WS"/
  mkdir -p "$WS"/{home,logs,submissions,experiments}
  printf 'agent=%s\nrecord=%s\n' "$AGENT" "$RECORD" >"$run_dir/run.meta"
  (
    cd "$WS"
    git init -q -b main
    git add -A
    git -c user.email=bench@local -c user.name=bench commit -q -m "init: $RECORD" || true
  )
fi
# Per-run remote workspace (isolates this run's remote files; reused on resume).
export BENCHMARK_REMOTE_WS="${BENCHMARK_REMOTE_WS:-$BENCHMARK_REMOTE_BASE/runs/$run_id}"
# Record the remote ws once (first leg) for traceability; never overwrite on resume.
[[ -f "$run_dir/run.meta" ]] && ! grep -q '^remote_ws=' "$run_dir/run.meta" 2>/dev/null \
  && printf 'remote_ws=%s\n' "$BENCHMARK_REMOTE_WS" >>"$run_dir/run.meta"

# --- bridges: install torchrun + submit on PATH ------------------------------
bin_dir="$run_dir/bin"
mkdir -p "$bin_dir"
ln -sf "$tools_dir/torchrun_remote.sh" "$bin_dir/torchrun"
cat >"$bin_dir/submit" <<EOF
#!/usr/bin/env bash
exec python3 "$tools_dir/submit_split.py" "\$@"
EOF
chmod +x "$bin_dir/submit"
# Pin a specific agent CLI binary if requested (e.g. a particular claude version),
# shadowing whatever else is on PATH. Skip the npm (re)install by default so the
# pinned version is not upgraded/overwritten.
if [[ -n "${BENCHMARK_CLAUDE_BIN:-}" ]]; then
  [[ -x "$BENCHMARK_CLAUDE_BIN" ]] || { echo "BENCHMARK_CLAUDE_BIN not executable: $BENCHMARK_CLAUDE_BIN" >&2; exit 1; }
  ln -sf "$BENCHMARK_CLAUDE_BIN" "$bin_dir/claude"
  export BENCHMARK_SKIP_AGENT_INSTALL="${BENCHMARK_SKIP_AGENT_INSTALL:-1}"
fi
export PATH="$bin_dir:$PATH"
echo ">> bridges on PATH: $(command -v torchrun), $(command -v submit)"
if [[ -n "${BENCHMARK_CLAUDE_BIN:-}" ]]; then
  echo ">> claude pinned: $(command -v claude) -> $(readlink "$bin_dir/claude") ($("$bin_dir/claude" --version 2>/dev/null | head -1)); agent install skipped=$BENCHMARK_SKIP_AGENT_INSTALL"
fi

# --- remote: one-time container prep -----------------------------------------
echo ">> preparing container $BENCHMARK_REMOTE"
# 1) FineWeb10B must already live at the SHARED <base>/data/fineweb10B (an isolated
#    copy of just the 9 train + 1 val shards). This script never creates/links it.
remote_data="$BENCHMARK_REMOTE_BASE/data/fineweb10B"
if ! "${ssh_cmd[@]}" "$BENCHMARK_REMOTE" \
      "ls $(printf '%q' "$remote_data")/fineweb_train_000001.bin >/dev/null 2>&1"; then
  echo "data not found at $BENCHMARK_REMOTE:$remote_data" >&2
  echo "  -> place the 9 train + 1 val shards there before running." >&2
  exit 1
fi
echo "   data present at $remote_data"
# 2) Per-run remote workspace, with the shared data linked in.
"${ssh_cmd[@]}" "$BENCHMARK_REMOTE" "
  mkdir -p $(printf '%q' "$BENCHMARK_REMOTE_WS/submissions") &&
  ln -sfn $(printf '%q' "$BENCHMARK_REMOTE_BASE/data") $(printf '%q' "$BENCHMARK_REMOTE_WS/data")
"
echo "   remote workspace: $BENCHMARK_REMOTE_WS (data -> $BENCHMARK_REMOTE_BASE/data)"
# 3) Tool modules are a HARD dependency for submit -> create the dir and fail closed.
"${ssh_cmd[@]}" "$BENCHMARK_REMOTE" "mkdir -p $(printf '%q' "$BENCHMARK_REMOTE_TOOLS")"
if ! ( COPYFILE_DISABLE=1 tar cf - -C "$tools_dir" . ) \
      | "${ssh_cmd[@]}" "$BENCHMARK_REMOTE" \
        "tar --warning=no-unknown-keyword -xf - -C $(printf '%q' "$BENCHMARK_REMOTE_TOOLS")"; then
  echo "ERROR: failed to copy tool modules to $BENCHMARK_REMOTE:$BENCHMARK_REMOTE_TOOLS" >&2
  echo "  these are required for submit (submit_runs.py + deps); aborting." >&2
  exit 1
fi
echo "   tools synced to $BENCHMARK_REMOTE_TOOLS"

# --- contract(): BENCHMARK_* (local paths) -----------------------------------
export HOME="$WS/home"
export BENCHMARK_WORKSPACE="$WS"
export BENCHMARK_RUN_ID="$run_id"
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
# The shared prompts hard-code "/workspace/..."; in route B the agent actually
# works in $WS on this machine. Rewrite the prompts to the real local workspace
# so the agent creates the right dirs and `submit <path>` resolves. The bridges
# still map $WS-relative paths to <remote_ws> on the GPU side.
prompts_local="$run_dir/prompts"
mkdir -p "$prompts_local"
localize() {  # copy a prompt file with /workspace -> $WS, echo the new path
  local src="$1" dst="$prompts_local/$(basename "$1")"
  sed "s|/workspace|$WS|g" "$src" >"$dst"
  printf '%s' "$dst"
}
case "$AGENT" in
  claude|codex)
    args=(
      --problem-file="$(localize "$shared_prompts/problem.txt")"
      --rules-file="$(localize "$shared_prompts/RULES.md")"
      --prompt-file="$(localize "$shared_prompts/local_prompt.md")"
      --resume-file="$(localize "$shared_prompts/resume_prompt.md")"
    )
    ;;
  autoresearch)
    ap="$agent_dir/prompts"
    args=(
      --rules-file="$(localize "$shared_prompts/RULES.md")"
      --program-file="$(localize "$ap/program.md")"
      --prompt-file="$(localize "$ap/local.md")"
      --resume-file="$(localize "$ap/resume.md")"
    )
    ;;
  *)
    echo "no prompt wiring for AGENT '$AGENT'" >&2; exit 1 ;;
esac

# --- budget: accumulate leg time toward BENCHMARK_TOTAL_HOURS -----------------
budget_file="$run_dir/budget.env"
total_seconds="$(awk -v h="$BENCHMARK_TOTAL_HOURS" 'BEGIN{printf "%d", h*3600}')"
consumed_seconds=0
# Parse the budget file as DATA, never `source` it: the agent has a shell and could
# write run_dir/budget.env, so executing it would be an arbitrary-code-exec hole.
# Only accept integer total_seconds= / consumed_seconds= lines, then clamp.
read_budget() {
  local t c
  t="$(sed -n 's/^total_seconds=\([0-9][0-9]*\)$/\1/p' "$budget_file" 2>/dev/null | tail -1)"
  c="$(sed -n 's/^consumed_seconds=\([0-9][0-9]*\)$/\1/p' "$budget_file" 2>/dev/null | tail -1)"
  [[ -n "$t" ]] && total_seconds="$t"
  [[ -n "$c" ]] && consumed_seconds="$c"
  (( consumed_seconds < 0 )) && consumed_seconds=0
  (( consumed_seconds > total_seconds )) && consumed_seconds=$total_seconds
}
[[ -f "$budget_file" ]] && read_budget
printf 'total_seconds=%s\nconsumed_seconds=%s\n' "$total_seconds" "$consumed_seconds" >"$budget_file"

remaining=$(( total_seconds - consumed_seconds ))
if (( remaining <= 0 )); then
  echo ">> budget exhausted: consumed ${consumed_seconds}s of ${total_seconds}s. Nothing to do." >&2
  exit 0
fi
# this leg = min(requested, remaining); empty request => use the whole remaining
if [[ -n "$SESSION_HOURS_REQ" ]]; then
  leg_req="$(awk -v h="$SESSION_HOURS_REQ" 'BEGIN{printf "%d", h*3600}')"
else
  leg_req=$remaining
fi
leg_seconds=$(( leg_req < remaining ? leg_req : remaining ))
export BENCHMARK_SESSION_HOURS="$(awk -v s="$leg_seconds" 'BEGIN{printf "%.6f", s/3600}')"

echo ">> launching agent '$AGENT' on this machine"
echo "   budget:    consumed ${consumed_seconds}s / total ${total_seconds}s; this leg up to ${leg_seconds}s"
echo "   workspace: $WS"
echo "   logs:      tail -f $WS/agent_trace.txt"
echo "   GPUs via:  $BENCHMARK_REMOTE ($BENCHMARK_REMOTE_WS)"
echo "   resume:    RESUME=$run_id bash nanogpt/run/networked_bootstrap.sh"

# --- run with budget accounting -----------------------------------------------
# - heartbeat persists elapsed every 30s (survives ungraceful kills, <=30s stale)
# - watchdog HARD-stops the whole child tree at leg_seconds, so one long Claude
#   call / hung ssh cannot overrun this leg's wall-clock window
# - trap finalizes on INT/TERM/HUP/EXIT (NOT exec, so the trap can fire)
leg_start=$(date +%s)
persist_consumed() {
  local now c; now=$(date +%s); c=$(( consumed_seconds + now - leg_start ))
  (( c > total_seconds )) && c=$total_seconds
  printf 'total_seconds=%s\nconsumed_seconds=%s\n' "$total_seconds" "$c" >"$budget_file"
}
kill_tree() {  # TERM a pid and all of its descendants (portable; no setsid needed)
  local p="$1" child
  for child in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$child"; done
  kill -TERM "$p" 2>/dev/null || true
}

( while :; do sleep 30; persist_consumed; done ) &
hb_pid=$!

_finalized=0
finalize() {
  [[ "$_finalized" == 1 ]] && return; _finalized=1
  kill "$hb_pid" "${wd_pid:-}" 2>/dev/null || true
  [[ -n "${child_pid:-}" ]] && kill_tree "$child_pid"
  persist_consumed
  read_budget
  echo ">> leg ended; consumed ${consumed_seconds}/${total_seconds}s total. Resume: RESUME=$run_id" >&2
  # Killing the local child tree above does NOT reliably reap the remote
  # torchrun/submit_runs.py the bridges started over ssh (no PTY => no SIGHUP
  # propagation), so they can keep burning GPU after this leg ends. Reap them,
  # but RUN-SCOPED: only processes whose cwd is inside THIS run's remote
  # workspace, so a shared container's other experiments are untouched. (The old
  # global `pkill -f train_gpt.py` would have killed unrelated runs.) Safe to do
  # on every exit -- on a clean finish nothing under the ws is still running.
  if [[ "${BENCHMARK_KILL_REMOTE_ON_EXIT:-1}" == "1" ]]; then
    echo ">> reaping leftover remote training under $BENCHMARK_REMOTE_WS on $BENCHMARK_REMOTE" >&2
    local ws_q; ws_q="$(printf '%q' "$BENCHMARK_REMOTE_WS")"
    "${ssh_cmd[@]}" "$BENCHMARK_REMOTE" "
      ws=$ws_q
      for sig in TERM TERM KILL; do
        hit=0
        for p in /proc/[0-9]*; do
          cwd=\$(readlink \"\$p/cwd\" 2>/dev/null) || continue
          case \"\$cwd\" in
            \"\$ws\"|\"\$ws\"/*) kill -\$sig \"\${p#/proc/}\" 2>/dev/null && hit=1 ;;
          esac
        done
        [ \"\$hit\" = 0 ] && break
        sleep 2
      done
    " 2>/dev/null || true
  fi
}
trap finalize INT TERM HUP EXIT

bash "$agent_dir/run.sh" "${args[@]}" &
child_pid=$!
( sleep "$leg_seconds"; echo ">> watchdog: leg limit ${leg_seconds}s reached, stopping" >&2; kill_tree "$child_pid" ) &
wd_pid=$!

status=0
set +e; wait "$child_pid"; status=$?; set -e
exit "$status"
