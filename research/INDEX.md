# WaveCast loop — research INDEX (updated 2026-08-23, pass 5)

## Phase A status — COMPLETE
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [done 2026-08-23 94687d5] Massive fetch resilience: 429 retry, per-ticker isolation, pacing.
- **A3** [done 2026-08-23 c684215] Gap census (read-only) + CatchupRunner + `scripts/catchup_forward.py`; demo run into loop_scratch.
- **A4** [done 2026-08-23 72ef72f] Honest forward eval harness: archived claim REFUTED on 6mo OOS.
- **A5** [done 2026-08-23 93003b5] E2E cycle proof via `scripts/a5_e2e_cycle.py`: 18 pending + 16 catch-up rows in loop_scratch; explicit resolve pass resolved 15/16 (3 correct). Found: catch-up rows never auto-resolve; Massive 429s still cost 10-30% of tickers/cycle.

Phase B (B1–B5) UNLOCKED. Next: **B1** backtest grid harness.

## Latest measurements
- `pytest tests/ -q`: **544 passed**, ~94s; ruff clean (incl. new script).
- CI fix b1996b5: `pytest.importorskip("playwright")` added to `tests/test_dashboard_pages.py` — unblocks PR 1 red CI (playwright is local-only).
- A5 cycle (`scripts/a5_e2e_cycle.py --session 2026-08-21`, Sunday → Friday bars):
  phase 1 live run_once = 18 predictions PENDING; phase 2 catch-up = 16;
  phase 3 resolve vs historical bars = 15/16 resolved, 3 correct; SLV left
  pending by 429s. Session-end (23:00 bar) rows can resolve with actual_return=0.0
  via the tolerance fallback.
- A4 verdict (unchanged, n=1860): unfiltered econ acc 32.6%; A2i-filtered 55.5%
  but negative expectancy at 7bps; persistence baseline Sharpe +2.18 vs model −0.26.

## Reports (newest first)
- research/2026-08-23-2327.md — Pass 5 / A5: e2e cycle proof, resolve-phase gap, 429 coverage loss.
- research/2026-08-23-2318.md — Pass 4 / A4: eval harness, verdict paragraph, persistence finding.
- research/2026-08-23-2300.md — Pass 3 / A3: census table, JPM crash-point finding, catch-up demo, promotion policy question.
- research/2026-08-23-2249.md — Pass 2 / A2: retry design, isolation tests, suite counts.
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed. Note: `research/` is gitignored — reports need `git add -f`.
