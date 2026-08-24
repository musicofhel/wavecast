# WaveCast loop — research INDEX (updated 2026-08-23, pass 2)

## Phase A status
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [done 2026-08-23 94687d5] Massive fetch resilience: 429 retry (exp backoff + Retry-After), per-ticker isolation, pacing. ← done
- **A3** [open] Forward-run gap census + catch-up into loop_scratch. ← NEXT
- **A4** [open] Honest forward eval harness (economic direction, Wilson CIs, baselines).
- **A5** [open] End-to-end forecast proof cycle into loop_scratch.

Phase B (B1–B5) locked until A1–A5 done.

## Latest measurements
- `pytest tests/ -q`: **522 passed**, ~95s on RTX 2060 SUPER (510 at A1 + 12 from A2).
- `ruff check src/ tests/`: clean.
- A2 behaviors proven by tests: retry succeeds after two 429s; Retry-After honored; persistent 429 → DataError after 5 attempts; non-429 fails fast; BAD ticker skipped, run continues; pacing sleeps between tickers.
- Ground truth (production ledger, jq, 2026-08-23): unfiltered 3-class acc 32.6% (606/1860);
  A2i-filtered 55.5% (167/301) vs archived claim of 66.7% — A4's question.

## Reports (newest first)
- research/2026-08-23-2249.md — Pass 2 / A2: retry design, isolation tests, suite counts.
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed, config-ignore rationale. Note: `research/` is gitignored — reports need `git add -f`.
