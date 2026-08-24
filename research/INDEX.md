# WaveCast loop — research INDEX (updated 2026-08-23, pass 3)

## Phase A status
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [done 2026-08-23 94687d5] Massive fetch resilience: 429 retry, per-ticker isolation, pacing.
- **A3** [done 2026-08-23 c684215] Gap census (read-only) + CatchupRunner + `scripts/catchup_forward.py`; demo run into loop_scratch. ← done
- **A4** [open] Honest forward eval harness (economic direction, Wilson CIs, baselines). ← NEXT
- **A5** [open] End-to-end forecast proof cycle into loop_scratch.

Phase B (B1–B5) locked until A1–A5 done.

## Latest measurements
- `pytest tests/ -q`: **528 passed**, ~94s (+6 catch-up tests over A2's 522); ruff clean.
- Production ledger census (`python scripts/catchup_forward.py`): span 2026-02-17→08-22;
  **44 fully missing weekdays + 21 partial days = 1195 missing pairs.**
- Root cause of degradation: since 2026-07-10 every cron cycle dies at JPM 429
  after logging only AAPL/AMZN/GOOGL/MSFT/NVDA (structural: those 5 have 115
  rows each, the other 15 exactly 87). A2 fixes the mechanism — needs Aaron to
  deploy/restart the cron on this branch.
- Live catch-up demo: `--session 2026-06-17` → 17/20 tickers backfilled with real
  session bar timestamps into `~/.wavecast/forward_tests/loop_scratch/a3_catchup_20260617/`.
- Ground truth for A4 (production ledger): unfiltered 3-class acc 32.6% (606/1860);
  A2i-filtered 55.5% (167/301) vs archived claim of 66.7%.

## Reports (newest first)
- research/2026-08-23-2300.md — Pass 3 / A3: census table, JPM crash-point finding, catch-up demo, promotion policy question.
- research/2026-08-23-2249.md — Pass 2 / A2: retry design, isolation tests, suite counts.
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed. Note: `research/` is gitignored — reports need `git add -f`.
