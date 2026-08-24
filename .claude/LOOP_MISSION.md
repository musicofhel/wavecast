# WaveCast RW research loop — mission + task ledger

**Aaron's directive (2026-08-23):** read-write loop on wavecast, same ox-alpha
machinery as the autoapply cleanup loop, but implementing. "It needs a good
update before we move forward on auto research implementation — goal being it
is able to do forecasting with proper test." Research phase: "we backtest
different tickers and different ways to make it work — pyramiding, timescales
etc." Confirmed framing (Aaron, same day): **"0x is going to work from the
outside, inward until it's clean enough to start doing strategy testing in an
auto research manner"** — Phase A is that outside-in hardening; Phase B is the
strategy testing, and it stays locked until A is clean.

**Posture:** commits on branch `loop/system-update-v1` only, never push, never
touch the A2i cron, the live model dir (`~/.wavecast/models/d1_augmented_v1/`),
or the production forward ledger (`~/.wavecast/forward_tests/d1_forward_v1/` —
READ it freely, write NEVER; scratch runs go to
`~/.wavecast/forward_tests/loop_scratch/`).

**Ground truth measured 2026-08-23 (jq over the production ledger, 1880 rows):**
- 1860 resolved predictions since archival (2026-02 → 2026-08-21), 20 pending.
- Unfiltered 3-class accuracy: **32.6%** (606/1860) — random-level, consistent
  with the Phase 7 audit (unfiltered has no edge; that is expected, not news).
- **A2i-filtered (a2i_trade=true): 55.5% (167/301)** — above the ~33% random
  floor but far below the archived claim of 66.7%. Whether the edge DEGRADED or
  the 66.7% was thin-sample is task A4's question.
- The forward cron is FAILING on Massive API 429s (see cron.log: JPM
  MaxRetryError kills the whole run) — recent sessions have missing/partial
  prediction coverage.

## Phase A — system update: "forecasting with proper test"

- **A1 [done 2026-08-23 f57a913] Suite + environment baseline.** Baseline: pytest **510 passed** (~95s; CLAUDE.md's 424/~9s is stale), ruff **19 errors -> 0 fixed** (13 mechanical; UP042 + cli/app.py E402 handled via config ignores with reasons). Report: `research/2026-08-23-2233.md`.
  Original: Run `pytest tests/ -q` and
  `ruff check src/ tests/` from `.venv`; record exact counts/failures in the
  report. Fix any rot (dep drift, API breakage) with minimal diffs. torch
  2.10.0+cu128, CUDA available — verified 2026-08-23. Done = suite green (or
  every red documented + fixed) and the baseline recorded.
- **A2 [done 2026-08-23 94687d5] Massive fetch resilience.** `_fetch_aggs_with_retry` (5 attempts, exp backoff 2s->60s, Retry-After honored) shared by both fetchers; forward runner skips a failed ticker + paces between tickers. 12 new tests; suite 522 passed, ruff clean. Report: `research/2026-08-23-2249.md`.
  Original: `src/wavecast/data/sources.py`
  `fetch_massive_ohlcv` dies on 429 MaxRetryError and one ticker's death kills
  the whole forward run. Add 429-aware retry with exponential backoff (honor
  Retry-After if present), inter-ticker pacing, and per-ticker error isolation
  in the forward runner (one failed ticker → logged + skipped, run continues).
  Unit tests with mocked 429 responses. Done = tests green proving both
  behaviors.
- **A3 [done 2026-08-23 c684215] Gap census + catch-up.** Census (read-only):
  44 fully missing weekdays + 21 partial days = 1195 missing pairs; post
  2026-07-10 every run dies at JPM 429 -> only first 5 tech tickers logged.
  `forward/catchup.py` (`census_gaps`, `CatchupRunner.run_for_session`) +
  `scripts/catchup_forward.py` (refuses prod ledger); demonstrated on
  2026-06-17 -> 17/20 tickers into loop_scratch. Report: `research/2026-08-23-2300.md`.
  Original: From
  `~/.wavecast/forward_tests/d1_forward_v1/cron.log` + the ledger timestamps:
  which trading days since 2026-02 have missing/partial predictions, which
  tickers are systematically missing (JPM?). Implement a catch-up runner that
  backfills missed sessions into `loop_scratch/` (NEVER the production ledger)
  and write up how Aaron promotes/merges it. Done = census table in the report
  + catch-up run demonstrated on ≥1 missed day into scratch.
- **A4 [done 2026-08-23 72ef72f] Honest forward eval harness.** `evaluation/forward_ledger.py` + `scripts/forward_eval.py` + 16 tests (suite 544 passed). Verdict: archived claim REFUTED on 6mo OOS — unfiltered 32.6% (random), A2i-filtered 55.5% [49.8%, 61.0%] on n=301 loses money after 7bps; persistence baseline Sharpe +2.18 vs model -0.26 on the same set. Report: `research/2026-08-23-2318.md`.
  Original: Honest forward evaluation harness — the "proper test".**
  `scripts/forward_eval.py` + unit tests, reading the production ledger
  READ-ONLY: overall + A2i-filtered economic directional accuracy with Wilson
  CIs, per-ticker and per-month breakdown, Sharpe/expectancy at 7bps round-trip
  vs persistence + always-up baselines, flat-prediction rate. Uses ECONOMIC
  direction (actual_return sign), never the symbolic token metric (Phase 7
  audit trap). Verdict paragraph: is the archived "66.7% acc / +10.40 Sharpe"
  claim alive, degraded, or refuted on 6 months OOS? Done = script + tests
  green + verdict with numbers in the report.
- **A5 [open] End-to-end forecast proof.** One full forward-test cycle today
  into `loop_scratch/` (fetch → predict → later resolve): model loads,
  predictions land, resolution works with the A2 fixes. Done = scratch
  predictions.jsonl rows exist + the cycle's log is clean.

## Phase B — auto-research: backtests (unlocked when A1–A5 are done)

Aaron's frame: find configurations that make it WORK — universe, timescale,
trade management. Every experiment: proper OOS split, 5-metric eval with
flat-bias detection (Round-2 criteria — see PHASE12_RESULTS.md), costs at
7bps, and a persistence baseline. Results append to `research/results.jsonl`
(schema: task A4's metrics + config hash + data window).

- **B1 [open] Backtest grid harness.** One entry point that runs {universe
  subset × interval × trading rule} through the existing `signals/` framework
  (SignalBacktest, PositionSizer, TransactionCostModel) against the trained
  model's signals; results ledger + tests.
- **B2 [open] Ticker/universe selection.** Per-ticker forward + backtest
  performance; does a top-K sub-universe chosen on train hold up OOS, or is
  per-ticker performance unstable (rank correlation train→test)?
- **B3 [open] Timescale sweep.** Daily vs hourly bars (daily needs retrain —
  small models train in seconds locally; anything projected >15 min GPU is
  written up as a RunPod proposal instead of run — house rule). Also horizon
  variants on existing hourly.
- **B4 [open] Trade management.** Pyramiding (scale-in on consecutive
  same-direction signals), sizing variants (flat vs magnitude-scaled vs
  vol-scaled), tercile threshold sweep, holding-period/exit variants. Backtest
  on train, confirm on the untouched forward record where possible.
- **B5 [open] Open lane.** Proposals earned by B1–B4 findings; each needs
  Aaron's nod in the report before heavy implementation.

## Ledger protocol (every pass)

Pick the FIRST task not `[done]` (or continue an `[in_progress]` one). When a
pass advances a task, edit its status line here: `[open]` → `[in_progress:
<one-line state>]` → `[done <date> <commit>]`. Add discovered subtasks as
indented bullets under their parent. Never delete history — strike through
with `~~` if a task dies, and say why.
