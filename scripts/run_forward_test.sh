#!/bin/bash
# WaveCast A2i Forward Test Runner
# Run after market close to resolve pending predictions and generate new ones.
#
# Cron (weekdays at 9 PM ET):
#   0 21 * * 1-5 /home/musicofhel/wavecast/scripts/run_forward_test.sh
#
# Manual:
#   bash scripts/run_forward_test.sh

set -euo pipefail

cd /home/musicofhel/wavecast
source .venv/bin/activate

LOG_DIR="$HOME/.wavecast/forward_tests/d1_forward_v1"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting forward test cycle" | tee -a "$LOG_FILE"

# Run one cycle: resolve pending predictions, generate new ones
python scripts/forward_test_d1.py --report 2>&1 | tee -a "$LOG_FILE"

echo "$(date '+%Y-%m-%d %H:%M:%S'): Forward test cycle complete" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
