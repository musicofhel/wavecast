# WaveCast loop — research INDEX (updated 2026-08-23, pass 1)

## Phase A status
- **A1** [done 2026-08-23 f57a913] Suite/env baseline: 510 tests pass (~95s), ruff 19→0 errors.
- **A2** [open] Massive fetch resilience (429 retry + pacing + per-ticker isolation). ← NEXT
- **A3** [open] Forward-run gap census + catch-up into loop_scratch.
- **A4** [open] Honest forward eval harness (economic direction, Wilson CIs, baselines).
- **A5** [open] End-to-end forecast proof cycle into loop_scratch.

Phase B (B1–B5) locked until A1–A5 done.

## Latest measurements
- `pytest tests/ -q`: **510 passed**, ~95s on RTX 2060 SUPER (CLAUDE.md's "424/~9s" is stale).
- `ruff check src/ tests/`: clean after pass-1 fixes; UP042 and cli/app.py E402 ignored via config (reasons in pyproject.toml).
- Ground truth (production ledger, jq, 2026-08-23): unfiltered 3-class acc 32.6% (606/1860);
  A2i-filtered 55.5% (167/301) vs archived claim of 66.7% — A4's question.

## Reports (newest first)
- research/2026-08-23-2233.md — Pass 1 / A1: baseline counts, lint rot fixed, config-ignore rationale. Note: `research/` is gitignored — reports need `git add -f`.
