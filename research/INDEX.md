# WaveCast loop — research INDEX (updated 2026-08-23, pass 4)

## Phase A status
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [done 2026-08-23 94687d5] Massive fetch resilience: 429 retry, per-ticker isolation, pacing.
- **A3** [done 2026-08-23 c684215] Gap census (read-only) + CatchupRunner + `scripts/catchup_forward.py`; demo run into loop_scratch.
- **A4** [done 2026-08-23 72ef72f] Honest forward eval harness: archived claim REFUTED on 6mo OOS. ← done
- **A5** [open] End-to-end forecast proof cycle into loop_scratch. ← NEXT

Phase B (B1–B5) locked until A1–A5 done.

## Latest measurements
- `pytest tests/ -q`: **544 passed**, ~98s (+16 forward-ledger tests); ruff clean.
- A4 verdict (`python scripts/forward_eval.py`, production ledger READ-ONLY, n=1860 resolved):
  unfiltered economic accuracy **32.6% [30.5%, 34.7%]** (random-level);
  A2i-filtered **55.5% [49.8%, 61.0%]** (n=301) — above random floor but loses money after
  7bps (total −6.15%, negative expectancy). Archived "66.7% acc / +10.40 Sharpe" refuted.
- Headline: **persistence baseline on the same opportunity set: total +119.52%, Sharpe +2.18**
  vs model −12.78% / −0.26. 1h bars mean-revert; the model's directions fight that structure.
- Per-ticker live spread: CVX 58.1%, XOM 44.2% vs SPY 8.1%, UNH 2.3%, PFE 1.2% — broad ETFs
  flipped from best-in-backtest to worst-in-live (feeds B2).
- Flat-prediction rate 42.1%; Aug 2026 accuracy 13.8% is thin-sample (n=65, cron 429 deaths).

## Reports (newest first)
- research/2026-08-23-2318.md — Pass 4 / A4: eval harness, verdict paragraph, persistence finding.
- research/2026-08-23-2300.md — Pass 3 / A3: census table, JPM crash-point finding, catch-up demo, promotion policy question.
- research/2026-08-23-2249.md — Pass 2 / A2: retry design, isolation tests, suite counts.
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed. Note: `research/` is gitignored — reports need `git add -f`.
