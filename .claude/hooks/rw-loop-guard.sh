#!/usr/bin/env bash
# PreToolUse guard for the wavecast READ-WRITE research loop (ox-alpha passes).
#
# Posture: the looping session may read, edit, test, and commit INSIDE
# ~/wavecast (branch work only), and read/write scratch areas of ~/.wavecast —
# and nothing else. It must never: reach secrets (.env), push or switch
# branches, touch cron/systemd, or mutate the two PRODUCTION artifacts the A2i
# cron owns (the live model dir and the 6-month forward-prediction ledger).
#
# This is a deny-list guard (unlike autoapply's allow-list): the pass runs
# real dev commands (python, pytest, make, maturin), so enforcement focuses on
# the specific catastrophic actions. Recovery backstop: the orchestrator
# snapshots the forward ledger before every pass and diffs after.
#
# Contract: read hook JSON on stdin, emit a permissionDecision, exit 0.

set -uo pipefail

INPUT="$(cat)"
REPO="$HOME/wavecast"
WDATA="$HOME/.wavecast"

decide() { # $1=allow|deny  $2=reason
  jq -n --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:$d,permissionDecisionReason:$r}}'
  exit 0
}
allow() { decide allow "$1"; }
deny()  { decide deny  "wavecast rw-loop: $1"; }

TOOL="$(jq -r '.tool_name // ""' <<<"$INPUT")"

# Secret-bearing paths. The pass runs on a third-party stealth model, so the
# Massive API key file and shell rc are blocked outright. Boundary after .env
# so `os.environ` / `.env.example` in source stay readable.
SECRET_RE='\.env([^a-zA-Z0-9_.]|$)|\.credentials|\.zshrc|\.bashrc|\.ssh/|\.claude/projects'
check_secrets() {
  if [[ "$1" =~ $SECRET_RE ]]; then
    deny "that touches a secret-bearing path (.env / rc files / .ssh / memory). Scripts may LOAD the key at runtime; the key file itself is never read here."
  fi
}

# Production artifacts the A2i cron owns. Read = fine (evaluation reads the
# ledger). Mutation = never; catch-up/e2e runs write to a scratch forward dir.
PROTECT_RE='models/d1_augmented_v1|forward_tests/d1_forward_v1'

# ---------------------------------------------------------------- file writes
case "$TOOL" in
  Write|Edit|MultiEdit|NotebookEdit)
    FP="$(jq -r '.tool_input.file_path // .tool_input.notebook_path // ""' <<<"$INPUT")"
    check_secrets "$FP"
    if [[ "$FP" =~ $PROTECT_RE ]]; then
      deny "the live model dir and forward_tests/d1_forward_v1 are production artifacts owned by the A2i cron. Use a scratch dir (e.g. ~/.wavecast/forward_tests/loop_scratch/) and document promotion in the report."
    fi
    case "$FP" in
      "$REPO"/.git/*) deny "no direct .git internals writes — use git commands." ;;
      "$REPO"/*|"$WDATA"/*|/tmp/*) allow "write inside the sanctioned trees" ;;
      *) deny "writes are confined to ~/wavecast, ~/.wavecast (non-production), and /tmp. Blocked: $FP" ;;
    esac
    ;;
  Bash) : ;;   # falls through to the command analysis below
  Read|Grep|Glob)
    TARGET="$(jq -r '(.tool_input.file_path // "") + " " + (.tool_input.path // "")' <<<"$INPUT")"
    check_secrets "$TARGET"
    exit 0 ;;
  *) exit 0 ;;
esac

# -------------------------------------------------------------------- bash
CMD="$(jq -r '.tool_input.command // ""' <<<"$INPUT")"
[ -n "$CMD" ] || exit 0

check_secrets "$CMD"

# A mutating verb anywhere near a protected production path is refused.
if [[ "$CMD" =~ $PROTECT_RE ]]; then
  case "$CMD" in
    *rm\ *|*mv\ *|*cp\ *|*tee\ *|*truncate*|*dd\ *|*sed\ -i*|*'>'*)
      deny "that command names a production artifact (live model / d1_forward_v1 ledger) alongside a write-capable verb. Production artifacts are read-only to this loop; use a scratch dir." ;;
  esac
fi

check_git() {
  local sub="$1"; shift
  local args="$*"
  case "$sub" in
    status|log|diff|show|blame|shortlog|describe|grep|ls-files|ls-tree|\
rev-parse|rev-list|cat-file|for-each-ref|count-objects|whatchanged|reflog|check-ignore|\
add|commit|restore|rm|mv|apply)
      return 0 ;;
    branch)
      case " $args " in
        *" -d "*|*" -D "*|*" -m "*|*" -M "*|*--delete*|*--move*|*--set-upstream*)
          deny "git branch with a mutating flag — the loop stays on its own branch." ;;
      esac
      return 0 ;;
    stash)
      deny "git stash can strand or destroy work in a headless session. Commit instead — WIP commits on the loop branch are fine." ;;
    tag)
      case " $args " in *" -l"*|*"--list"*|" ") return 0 ;; esac
      deny "tag creation is Aaron's." ;;
    config)
      case " $args " in *" --get"*|*" --list"*|*" -l "*) return 0 ;; esac
      deny "git config writes." ;;
    push|pull|fetch|checkout|switch|reset|rebase|merge|clean|remote|cherry-pick|revert|worktree|submodule)
      deny "git $sub is off-limits: the loop commits on its own branch (loop/system-update-v1) and never pushes, switches, or rewrites history. Aaron reviews and pushes." ;;
    *)
      deny "git $sub is not on the sanctioned list (read subcommands + add/commit/restore/rm/mv/apply)." ;;
  esac
}

# Split on separators; check each segment's head command.
SEGMENTS="${CMD//&&/$'\n'}"; SEGMENTS="${SEGMENTS//||/$'\n'}"
SEGMENTS="${SEGMENTS//;/$'\n'}"

while IFS= read -r seg; do
  seg="${seg#"${seg%%[![:space:]]*}"}"; seg="${seg%"${seg##*[![:space:]]}"}"
  [ -n "$seg" ] || continue
  # shellcheck disable=SC2086
  set -- $seg
  while [ $# -gt 0 ]; do
    case "$1" in
      *=*) shift ;;
      timeout|nice|ionice) shift; [ $# -gt 0 ] && case "$1" in [0-9]*|-*) shift ;; esac ;;
      *) break ;;
    esac
  done
  [ $# -gt 0 ] || continue
  head_cmd="$(basename -- "$1")"; shift

  case "$head_cmd" in
    git) check_git "${1:-status}" "${@:2}" ;;
    crontab) deny "crontab is off-limits — the A2i cron entry is production. Recommend cron changes in the report." ;;
    systemctl|sudo|su|shutdown|reboot) deny "$head_cmd mutates system state." ;;
    ssh|scp|rsync|sftp) deny "$head_cmd reaches other machines." ;;
    gh|claude|br|bv|ntm|docker|kubectl) deny "$head_cmd reaches outside this loop's scope (no nested agents, no GitHub writes, no containers)." ;;
    curl|wget) deny "raw network fetches are blocked — data access goes through the repo's own fetchers (python), docs go in the report as open questions." ;;
    pkill|kill|killall)
      case "$seg" in
        *wavecast*|*pytest*) ;; # may clean up its own runaway test/train processes
        *) deny "process kills are limited to this loop's own wavecast/pytest processes." ;;
      esac ;;
    rm)
      case "$seg" in
        *" /home/"*|*" ~"*|*" \$HOME"*)
          case "$seg" in
            *"$REPO"*|*wavecast*) ;; # repo-internal cleanup is fine (PROTECT_RE already checked)
            *) deny "rm with an absolute path outside ~/wavecast." ;;
          esac ;;
      esac ;;
  esac
done <<< "$SEGMENTS"

allow "sanctioned rw-loop command"
