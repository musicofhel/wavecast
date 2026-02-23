#!/usr/bin/env bash
# Run remaining exp3 experiments as separate processes (full GPU cleanup between)
set -u

cd ~/wavecast
source .venv/bin/activate

RESULTS_DIR="$HOME/.wavecast/audit/exp3"
LOG="/tmp/exp3_remaining3.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== REMAINING EXP3 v3 ==="

# 1. context_48
if [[ ! -f "${RESULTS_DIR}/context_48_results.json" ]]; then
    log "START context_48"
    timeout 2400 python -u -c "
from scripts.exp3.test_context import _make_context_builder
from scripts.exp3.evaluate_representation import evaluate_representation
evaluate_representation(name='context_48', build_dataset_fn=_make_context_builder(48), n_seeds=3, run_baseline=True)
" >> "$LOG" 2>&1
    if [[ -f "${RESULTS_DIR}/context_48_results.json" ]]; then
        verdict=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/context_48_results.json'))['verdict'])" 2>/dev/null)
        log "DONE context_48: $verdict"
    else
        log "FAIL context_48"
    fi
    sleep 5
else
    log "SKIP context_48"
fi

# 2. context_64
if [[ ! -f "${RESULTS_DIR}/context_64_results.json" ]]; then
    log "START context_64"
    timeout 2400 python -u -c "
from scripts.exp3.test_context import _make_context_builder
from scripts.exp3.evaluate_representation import evaluate_representation
evaluate_representation(name='context_64', build_dataset_fn=_make_context_builder(64), n_seeds=3, run_baseline=True)
" >> "$LOG" 2>&1
    if [[ -f "${RESULTS_DIR}/context_64_results.json" ]]; then
        verdict=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/context_64_results.json'))['verdict'])" 2>/dev/null)
        log "DONE context_64: $verdict"
    else
        log "FAIL context_64"
    fi
    sleep 5
else
    log "SKIP context_64"
fi

# 3. transition_head
if [[ ! -f "${RESULTS_DIR}/transition_head_results.json" ]]; then
    log "START transition_head"
    timeout 3600 python -u -m scripts.exp3.test_transition_head >> "$LOG" 2>&1
    if [[ -f "${RESULTS_DIR}/transition_head_results.json" ]]; then
        verdict=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/transition_head_results.json'))['verdict'])" 2>/dev/null)
        log "DONE transition_head: $verdict"
    else
        log "FAIL transition_head"
    fi
else
    log "SKIP transition_head"
fi

log "=== ALL DONE ==="

# Summary
python3 -c "
import json, os
rd = os.path.expanduser('~/.wavecast/audit/exp3')
for f in sorted(os.listdir(rd)):
    if f.endswith('_results.json'):
        d = json.load(open(os.path.join(rd, f)))
        n = f.replace('_results.json','')
        for k in ['challenger_avg','regression_avg','best_avg']:
            if k in d:
                m = d[k]
                print(f'  {n:20s} econ={m.get(\"econ_dir\",0):.1%} trans={m.get(\"transition_acc\",0):.1%} flat={m.get(\"pred_dist\",{}).get(\"flat\",0):.1%} → {d[\"verdict\"][:50]}')
                break
" | tee -a "$LOG"
