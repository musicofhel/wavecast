#!/usr/bin/env bash
# Launch ONE read-write research pass over ~/wavecast on OpenRouter's
# stealth/ox-alpha ("0x alpha") model, via the local retrying proxy on :8399,
# with the rw-loop guard hooks loaded via --settings so they bind to THIS
# process only.
#
# Called by the Fable orchestrator loop (/wavecast-orchestrate) or by hand.
set -euo pipefail

cd "$HOME/wavecast"

# The pass must be on the loop branch — it can't switch branches itself
# (guard denies checkout), so refuse to launch from anywhere else.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "loop/system-update-v1" ]; then
  echo "refusing: ~/wavecast is on '$BRANCH', not loop/system-update-v1" >&2
  exit 1
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  OPENROUTER_API_KEY="$(grep -oP '(?<=OPENROUTER_API_KEY=")[^"]+' "$HOME/.zshrc")"
fi
[ -n "$OPENROUTER_API_KEY" ] || { echo "no OPENROUTER_API_KEY found" >&2; exit 1; }

export ANTHROPIC_BASE_URL="http://127.0.0.1:8399"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""

# --allowedTools skips the (unanswerable) headless permission prompts; the
# PreToolUse guard still runs FIRST on every call and its deny always wins.
# Never widen this to --dangerously-skip-permissions.
exec claude \
  --model "stealth/ox-alpha" \
  --settings "$HOME/wavecast/.claude/rw-loop-settings.json" \
  --allowedTools "Read,Grep,Glob,Write,Edit,Bash" \
  --max-turns 150 \
  -p "/wavecast-pass${1:+ $1}"
