# Handoff: WaveCast Streamlit Dashboard

**Date**: 2026-02-23
**Status**: Complete, all tests green

## What Was Built

4-page Streamlit dashboard for monitoring the WaveCast production trading system.

### Pages

1. **Forward Test** (`/forward_test`) — Live prediction monitoring: metric cards (total/resolved/pending/A2i trades/accuracy), equity curve (all trades vs A2i), rolling accuracy chart, recent predictions table, run log expander
2. **Production System** (`/production`) — Validated A2i reference: 2026 OOS vs 2025 test comparison, 8-config Sharpe bar chart, A2i vs B2i head-to-head table, reversal filter failure warning, cross-tab heatmap, system spec expander
3. **Ticker Drilldown** (`/drilldown`) — Per-ticker analysis: ticker selector, per-ticker metrics, prediction timeline scatter, magnitude distribution histogram, full predictions table
4. **Experiment Archive** (`/experiments`) — Research browser: 66-experiment summary, round/verdict filters, results table, econ_dir vs transition scatter with baseline crosshairs, experiment detail expander
5. **Landing Page** (`/`) — Overview with prediction counts and page descriptions

### File Structure

```
dashboard/
  app.py                    # Entry point + landing page
  config.py                 # Paths, colors, formatters, tickers, sectors
  __init__.py
  data/
    loader.py               # All data loading with @st.cache_data
    __init__.py
  components/
    charts.py               # Plotly builders (equity curve, rolling accuracy, bar chart)
    metrics.py              # Metric row, verdict badge helpers
    __init__.py
  pages/
    1_forward_test.py       # Page 1
    2_production.py         # Page 2
    3_drilldown.py          # Page 3
    4_experiments.py        # Page 4
    __init__.py
scripts/
  run_dashboard.sh          # Launch script
tests/
  test_dashboard_pages.py   # 32 Playwright E2E tests
```

### Data Sources

- `~/.wavecast/forward_tests/production_a2i/predictions.jsonl` — Forward test predictions (TTL=60s cache)
- `~/.wavecast/audit/production/2026_validation.json` — A2i OOS validation
- `~/.wavecast/audit/production/pnl_simulation.json` — 8-config PnL simulation
- `~/.wavecast/audit/exp3/*_results.json` — Phase 12 experiment results
- `~/.wavecast/audit/feature_tests/*_results.json` — Round 1+2 experiment results
- `~/.wavecast/audit/exp3/crosstab_results.json` — Cross-tab analysis
- `~/.wavecast/models/d1_augmented_v1/config.json` — Model configuration
- `~/.wavecast/audit/production/baseline_lock.json` — CE baseline reference

## Key Architecture Decisions

1. **Streamlit multipage**: Pages in `dashboard/pages/` with numeric prefixes for sidebar ordering. Each page is self-contained (sys.path setup + module-level render call).
2. **No page imports in app.py**: Landing page is standalone to avoid double-execution when Streamlit auto-discovers pages.
3. **URL routes**: Streamlit strips numeric prefixes — `/forward_test` not `/1_forward_test`.
4. **sys.path manipulation**: Required at top of each page file because Streamlit adds the script's directory (not project root) to sys.path.
5. **Plotly dark theme**: All charts use `template="plotly_dark"` with custom accent color `#3b82f6` (blue = production).

## Gotchas

- Streamlit's `set_page_config()` must only be called in `app.py`, never in page files
- `use_container_width` is deprecated in Streamlit 1.54+; use `width="stretch"` instead
- Ruff N999 rule rejects numeric-prefix filenames; suppressed via `per-file-ignores` in pyproject.toml
- Page files need `noqa: E402` suppression for imports after sys.path setup
- Playwright tests must wait for `stStatusWidget` to disappear (Streamlit's "running" indicator) rather than relying on `networkidle`

## Test Results

- 32 Playwright E2E tests: all pass (77s)
- 476 existing unit/integration tests: all pass (90s)
- Ruff lint: clean

## Running

```bash
cd ~/wavecast && source .venv/bin/activate
streamlit run dashboard/app.py
# or
bash scripts/run_dashboard.sh
```
