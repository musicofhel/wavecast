#!/usr/bin/env bash
# Round 2 Experiment Runner - sequential with progress logging
# Monitors itself, logs every experiment start/end, writes a status file
set -eu

REPO="$HOME/wavecast"
RESULTS_DIR="$HOME/.wavecast/audit/feature_tests"
LOG="$REPO/scripts/round2_progress.log"
STATUS="$REPO/scripts/round2_status.txt"

cd "$REPO"
source .venv/bin/activate

EXPERIMENTS=(
  "exp2/mdn-head:test_mdn_head:mdn_head"
  "exp2/empl-loss:test_empl_loss:empl_loss"
  "exp2/bqn-crps:test_bqn_crps:bqn_crps"
  "exp2/n3pom-ordinal:test_n3pom:n3pom"
  "exp2/iqn:test_iqn:iqn"
  "exp2/bocpd-feature:test_bocpd:bocpd"
  "exp2/permutation-entropy:test_perm_entropy:perm_entropy"
  "exp2/focal-psi-gamma:test_focal_psi:focal_psi"
  "exp2/arctan-pinball:test_arctan_pinball:arctan_pinball"
  "exp2/huber-quantile:test_huber_quantile:huber_quantile"
  "exp2/curriculum-transition:test_curriculum:curriculum"
  "exp2/ddat-difficulty:test_ddat:ddat"
  "exp2/tnc-contrastive:test_tnc:tnc"
  "exp2/cost-contrastive:test_cost:cost"
  "exp2/neural-wavelet-layer:test_neural_wavelet:neural_wavelet"
  "exp2/tft-variable-selection:test_tft_vsn:tft_vsn"
  "exp2/ordinal-conformal:test_ordinal_conformal:ordinal_conformal"
  "exp2/eraps-conformal:test_eraps:eraps"
  "exp2/encqr-bootstrap:test_encqr:encqr"
)

TOTAL=${#EXPERIMENTS[@]}
DONE=0
FAILED=0
SKIPPED=0
LAST_DONE="none"
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
=== ROUND 2 EXPERIMENT RUNNER STATUS ===
Updated: $(date '+%Y-%m-%d %H:%M:%S')
Elapsed: ${hrs}h ${mins}m
Progress: ${DONE}/${TOTAL} done, ${FAILED} failed, ${SKIPPED} skipped
Remaining: $(( TOTAL - DONE - SKIPPED ))
Currently: ${CURRENT}
Last completed: ${LAST_DONE}
EOF
}

log "=========================================="
log "Starting Round 2 remaining experiments (${TOTAL} total)"
log "=========================================="

update_status

for entry in "${EXPERIMENTS[@]}"; do
  IFS=: read -r branch module result_name <<< "$entry"

  if [[ -f "${RESULTS_DIR}/${result_name}_results.json" ]]; then
    log "SKIP ${module} - results already exist"
    SKIPPED=$((SKIPPED + 1))
    DONE=$((DONE + 1))
    continue
  fi

  CURRENT="${module}"
  update_status

  log "START ${module} (branch: ${branch}) [${DONE}/${TOTAL}]"
  exp_start=$(date +%s)

  git checkout "${branch}" 2>/dev/null || {
    log "ERROR ${module} - git checkout failed"
    FAILED=$((FAILED + 1))
    DONE=$((DONE + 1))
    git checkout master 2>/dev/null || true
    continue
  }

  if timeout 1800 python -u -m "scripts.feature_tests.${module}" >> "$LOG" 2>&1; then
    exp_elapsed=$(( $(date +%s) - exp_start ))
    log "DONE ${module} in ${exp_elapsed}s"
    if [[ -f "${RESULTS_DIR}/${result_name}_results.json" ]]; then
      verdict=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${result_name}_results.json')); print(d.get('verdict','?'))" 2>/dev/null || echo "?")
      log "  VERDICT: ${verdict}"
    fi
  else
    exp_elapsed=$(( $(date +%s) - exp_start ))
    log "FAIL ${module} after ${exp_elapsed}s (exit=$?)"
    FAILED=$((FAILED + 1))
  fi

  DONE=$((DONE + 1))
  LAST_DONE="${module}"
  update_status

  git checkout master 2>/dev/null || true

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
