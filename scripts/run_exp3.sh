#!/usr/bin/env bash
# Phase 12 Experiment Runner - all 8 experiments sequential
set -eu

REPO="$HOME/wavecast"
RESULTS_DIR="$HOME/.wavecast/audit/exp3"
LOG="$REPO/scripts/exp3_progress.log"
STATUS="$REPO/scripts/exp3_status.txt"

cd "$REPO"
source .venv/bin/activate

mkdir -p "$RESULTS_DIR"

EXPERIMENTS=(
  "test_fracdiff:fracdiff"
  "test_finegrain:finegrain_7 finegrain_11"
  "test_context:context_32 context_48 context_64"
  "test_crossscale:crossscale"
  "test_range_dwt:range_dwt"
  "test_modwt:modwt"
  "test_regression:regression"
  "test_transition_head:transition_head"
)

TOTAL=${#EXPERIMENTS[@]}
DONE=0
FAILED=0
SKIPPED=0
CURRENT="starting"
START_TIME=$(date +%s)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

update_status() {
  local elapsed=$(( $(date +%s) - START_TIME ))
  local hrs=$(( elapsed / 3600 ))
  local mins=$(( (elapsed % 3600) / 60 ))
  cat > "$STATUS" << EOF
=== PHASE 12 EXPERIMENT RUNNER STATUS ===
Updated: $(date '+%Y-%m-%d %H:%M:%S')
Elapsed: ${hrs}h ${mins}m
Progress: ${DONE}/${TOTAL} done, ${FAILED} failed, ${SKIPPED} skipped
Remaining: $(( TOTAL - DONE - SKIPPED ))
Currently: ${CURRENT}
EOF
}

log "=========================================="
log "Starting Phase 12 experiments (${TOTAL} total)"
log "=========================================="

update_status

for entry in "${EXPERIMENTS[@]}"; do
  IFS=: read -r module result_names <<< "$entry"

  # Check if all results exist
  all_exist=true
  for rn in $result_names; do
    if [[ ! -f "${RESULTS_DIR}/${rn}_results.json" ]]; then
      all_exist=false
      break
    fi
  done

  if [[ "$all_exist" = true ]]; then
    log "SKIP ${module} - all results already exist"
    SKIPPED=$((SKIPPED + 1))
    DONE=$((DONE + 1))
    continue
  fi

  CURRENT="${module}"
  update_status

  log "START ${module} [${DONE}/${TOTAL}]"
  exp_start=$(date +%s)

  if timeout 2400 python -u -m "scripts.exp3.${module}" >> "$LOG" 2>&1; then
    exp_elapsed=$(( $(date +%s) - exp_start ))
    log "DONE ${module} in ${exp_elapsed}s"
    for rn in $result_names; do
      if [[ -f "${RESULTS_DIR}/${rn}_results.json" ]]; then
        verdict=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${rn}_results.json')); print(d.get('verdict','?'))" 2>/dev/null || echo "?")
        log "  VERDICT ${rn}: ${verdict}"
      fi
    done
  else
    exp_elapsed=$(( $(date +%s) - exp_start ))
    log "FAIL ${module} after ${exp_elapsed}s (exit=$?)"
    FAILED=$((FAILED + 1))
  fi

  DONE=$((DONE + 1))
  update_status

  sleep 5
done

CURRENT="FINISHED"
update_status

elapsed=$(( $(date +%s) - START_TIME ))
hrs=$(( elapsed / 3600 ))
mins=$(( (elapsed % 3600) / 60 ))

log "=========================================="
log "ALL DONE in ${hrs}h ${mins}m"
log "Completed: $((DONE - FAILED - SKIPPED)), Failed: ${FAILED}, Skipped: ${SKIPPED}"
log "=========================================="
