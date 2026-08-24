# WaveCast loop — research INDEX (updated 2026-08-24, pass 7)

## Phase A status — COMPLETE
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [done 2026-08-23 94687d5] Massive fetch resilience: 429 retry, per-ticker isolation, pacing.
- **A3** [done 2026-08-23 c684215] Gap census (read-only) + CatchupRunner + `scripts/catchup_forward.py`; demo run into loop_scratch.
- **A4** [done 2026-08-23 72ef72f] Honest forward eval harness: archived claim REFUTED on 6mo OOS.
- **A5** [done 2026-08-23 93003b5] E2E cycle proof via `scripts/a5_e2e_cycle.py`: catch-up rows need an explicit resolve pass; Massive 429s still cost 10-30% of tickers/cycle.

## Phase B status
- **B1** [done 2026-08-24 dcc4138] Backtest grid harness (`signals/grid.py`, `scripts/backtest_grid.py`): rules {persistence, mean_reversion, momentum:lb, model}, 7bps round-trip, flat-rate + persistence baseline per cell, JSONL ledger `research/results.jsonl`.
- **B2–B5** open. Next: **B2** per-ticker stability, or wire model predictions into `model` cells via `--predictions`.

## Latest measurements
- `pytest tests/ -q`: **561 passed**, ~93s; ruff clean.
- B1 first grid run (160 cells, 20 tickers × {1h,1d} × 4 rules, 7bps): median Sharpe by rule — persistence −0.85, mean_reversion −0.65, momentum(lb 2/8) −0.54. By interval: 1h −0.95 vs 1d −0.27. Best: BAC 1d persistence +1.14, GS 1d momentum +0.78.
- **Key nuance for Aaron's persistence question:** A4's forward-ledger persistence (+2.18) was daily-resolution; per-bar hourly persistence at 7bps is deeply negative — the edge question must be tested with holding periods (B4), not bar-flipping.
- A4 verdict (unchanged, n=1860): unfiltered econ acc 32.6%; A2i-filtered 55.5%, negative expectancy at 7bps.

## Reports (newest first)
- research/2026-08-23-2359.md — Pass 7 / B1: grid harness, first 160-cell run, hourly-cost finding.
- research/2026-08-23-2327.md — Pass 5 / A5: e2e cycle proof, resolve-phase gap, 429 coverage loss.
- research/2026-08-23-2318.md — Pass 4 / A4: eval harness, verdict paragraph, persistence finding.
- research/2026-08-23-2300.md — Pass 3 / A3: census table, JPM crash-point finding, catch-up demo, promotion policy question.
- research/2026-08-23-2249.md — Pass 2 / A2: retry design, isolation tests, suite counts.
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed. Note: `research/` is gitignored — reports need `git add -f`.
