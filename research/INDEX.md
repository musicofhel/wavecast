# WaveCast loop — research INDEX (updated 2026-08-24, pass 11)

## Phase A status — COMPLETE
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [done 2026-08-23 94687d5] Massive fetch resilience: 429 retry, per-ticker isolation, pacing.
- **A3** [done 2026-08-23 c684215] Gap census (read-only) + CatchupRunner + `scripts/catchup_forward.py`; demo run into loop_scratch.
- **A4** [done 2026-08-23 72ef72f] Honest forward eval harness: archived claim REFUTED on 6mo OOS.
- **A5** [done 2026-08-23 93003b5] E2E cycle proof via `scripts/a5_e2e_cycle.py`: catch-up rows need an explicit resolve pass; Massive 429s still cost 10-30% of tickers/cycle.

## Phase B status
- **B1** [done 2026-08-24 dcc4138] Backtest grid harness (`signals/grid.py`, `scripts/backtest_grid.py`): rules {persistence, mean_reversion, momentum:lb, model}, 7bps round-trip, flat-rate + persistence baseline per cell, JSONL ledger `research/results.jsonl`.
- **B2** [done 2026-08-24 db2576e] Ticker stability (`signals/stability.py`, `scripts/ticker_stability.py`): spearman train→test ≈ +0.5 on daily persistence/mean-rev; top-K selection = regime-concentration risk. Hourly cost-dead for ALL tickers.
- **B4** [done 2026-08-24 9b4fe0e] Trade management (`signals/trade_mgmt.py`, `scripts/trade_mgmt.py`): holding overlay + vol-scaled sizing. Headline hold=5 +0.30 — **REFUTED by B5** (same-window selection artifact).
- **B3** [done 2026-08-24 0d2ffb2] Timescale sweep at matched annual turnover: daily dominates hourly at every band; turnover-matching does NOT rescue hourly. Daily is the operating timescale. Daily retrain deferred to Aaron-gated lane.
- **B5** [done 2026-08-24 cb852c4] Walk-forward hold stability (`signals/walkforward.py`, `scripts/hold_walkforward.py`, 20 tickers × 4 folds): hold=5 re-selected only **38%** OOS (spread across all candidates); honest walk-forward Sharpe **−0.20** vs h5-fixed −0.14 on identical folds. B4's headline does not survive peek-free evaluation. h=10 fixed +0.83 is itself peeked (best-of-5). No unblocked tasks remain: longer-hold test vs Phase-B closeout needs Aaron's nod.

## Latest measurements
- `pytest tests/ -q`: **595 passed**, ~98s; ruff clean.
- B5 sweep (80 fold rows in `research/results.jsonl`, task="B5"; daily mean_rev, expanding train_bars=1000, 4 folds, 7bps): selection counts h1 5% / h2 14% / h3 26% / h5 38% / h10 18%; walk-forward mean Sharpe −0.197 (35/80 folds pos); fixed h10 +0.83 (peeked).
- B3 sweep: daily mean_rev h2 +0.15 / h5 +0.30 / h10 +0.29 (peeked); daily persistence ≤ −0.21; hourly ALL negative.
- B2 stability run: 1d persistence rho +0.51, topK_te +0.38 vs full_te −0.41.
- A4 verdict (n=1860): unfiltered econ acc 32.6%; A2i-filtered 55.5%, negative expectancy at 7bps.

## Reports (newest first)
- research/2026-08-24-0038.md — Pass 11 / B5: walk-forward hold stability; h5 unstable, honest Sharpe −0.20, B4 headline refuted.
- research/2026-08-24-0030.md — Pass 10 / B3: timescale sweep at matched annual turnover; hourly dead even then.
- research/2026-08-24-0020.md — Pass 9 / B4: holding-period overlay rescues daily mean-reversion from cost death (later refuted by B5).
- research/2026-08-24-0009.md — Pass 8 / B2: per-ticker stability, rank correlation, top-K OOS check.
- research/2026-08-23-2359.md — Pass 7 / B1: grid harness, first 160-cell run, hourly-cost finding.
- research/2026-08-23-2327.md — Pass 5 / A5: e2e cycle proof, resolve-phase gap, 429 coverage loss.
- research/2026-08-23-2318.md — Pass 4 / A4: eval harness, verdict paragraph, persistence finding.
- research/2026-08-23-2300.md — Pass 3 / A3: census table, JPM crash-point finding, catch-up demo, promotion policy question.
- research/2026-08-23-2249.md — Pass 2 / A2: retry design, isolation tests, suite counts.
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed. Note: `research/` is gitignored — reports need `git add -f`.
