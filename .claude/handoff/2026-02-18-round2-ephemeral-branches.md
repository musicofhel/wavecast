# Round 2 Ephemeral Branch Experiments — Setup Complete

**Date**: 2026-02-18
**Status**: 26 branches created, pushed, ready to run

## Context

Round 1 (2026-02-17) ran 20 ephemeral branches. 6 PASS, 14 FAIL. But stacking diagnostics revealed ALL 6 winners were **flat-bias gamers** (heteroscedastic predicted flat ~82%, inflating econ_dir).

A 2,841-paper review by 13 agents produced 14 reports with new ideas. This round tests 26 of those ideas with updated evaluation criteria that catch flat-bias gaming.

## Updated Evaluation (5 Metrics)

Every experiment now reports:
1. **Econ dir accuracy** — directional prediction accuracy
2. **Prediction distribution** (up%/flat%/down%) — catches flat-bias gaming (>40% flat = FAIL)
3. **Transition accuracy** — performance at regime changes
4. **Large-move accuracy** — top 1/3 by |return| (the tradeable metric)
5. **Sharpe with costs** — net of 7 bps transaction costs

Helper: `scripts/feature_tests/exp2_helpers.py` (committed on master)

## Branch Summary

| # | Branch | Tier | Script | LOC | Description |
|---|--------|------|--------|-----|-------------|
| 3 | `exp2/focal-flatcheck` | 0: Diagnostic | `test_focal_flatcheck.py` | 233 | Focal loss prediction distribution check |
| 4 | `exp2/ordinal-flatcheck` | 0: Diagnostic | `test_ordinal_flatcheck.py` | 247 | Ordinal loss prediction distribution check |
| 1 | `exp2/mine-ceiling` | 0: Ceiling | `test_mine_ceiling.py` | 331 | MINE mutual information → Fano bound |
| 2 | `exp2/bolt-ceiling` | 0: Ceiling | `test_bolt_ceiling.py` | 384 | BOLT loss → Bayes-optimal accuracy |
| 5 | `exp2/mdn-head` | 1: Target Rep | `test_mdn_head.py` | 459 | Mixture Density Network head |
| 6 | `exp2/empl-loss` | 1: Target Rep | `test_empl_loss.py` | 411 | Earth Mover's Pinball Loss |
| 7 | `exp2/bqn-crps` | 1: Target Rep | `test_bqn_crps.py` | 316 | Bernstein Quantile Network + CRPS |
| 8 | `exp2/n3pom-ordinal` | 1: Target Rep | `test_n3pom.py` | 311 | Neural ordinal regression (N3POM) |
| 9 | `exp2/iqn` | 1: Target Rep | `test_iqn.py` | 413 | Implicit Quantile Networks |
| 10 | `exp2/bocpd-feature` | 2: Features | `test_bocpd.py` | ~200 | BOCPD run-length posterior |
| 11 | `exp2/permutation-entropy` | 2: Features | `test_perm_entropy.py` | ~150 | Ordinal pattern entropy |
| 12 | `exp2/cyclical-focal` | 3: Loss | `test_cyclical_focal.py` | ~200 | Time-varying gamma focal |
| 13 | `exp2/focal-psi-gamma` | 3: Loss | `test_focal_psi.py` | ~200 | Focal + psi-gamma calibration |
| 14 | `exp2/arctan-pinball` | 3: Loss | `test_arctan_pinball.py` | ~200 | Smooth arctan pinball |
| 15 | `exp2/huber-quantile` | 3: Loss | `test_huber_quantile.py` | ~200 | Huber quantile regression |
| 16 | `exp2/curriculum-transition` | 4: Training | `test_curriculum.py` | ~250 | Curriculum by transition density |
| 17 | `exp2/ddat-difficulty` | 4: Training | `test_ddat.py` | ~250 | Autoencoder difficulty weighting |
| 18 | `exp2/tnc-contrastive` | 5: Contrastive | `test_tnc.py` | 566 | Temporal Neighborhood Coding |
| 19 | `exp2/cost-contrastive` | 5: Contrastive | `test_cost.py` | 500 | CoST frequency-domain contrastive |
| 20 | `exp2/neural-wavelet-layer` | 6: Architecture | `test_neural_wavelet.py` | 457 | Trainable wavelet decomposition |
| 21 | `exp2/tft-variable-selection` | 6: Architecture | `test_tft_vsn.py` | 536 | TFT gated residual variable selection |
| 22 | `exp2/ordinal-conformal` | 7: Conformal | `test_ordinal_conformal.py` | 325 | Contiguous ordinal prediction sets |
| 23 | `exp2/eraps-conformal` | 7: Conformal | `test_eraps.py` | 431 | Non-exchangeable conformal (ERAPS) |
| 24 | `exp2/encqr-bootstrap` | 7: Conformal | `test_encqr.py` | 354 | EnCQR bootstrap conformal |
| 25 | `exp2/hocmim-selection` | 8: Feature Sel | `test_hocmim.py` | 432 | High-order CMI feature ranking |
| 26 | `exp2/smrmr-selection` | 8: Feature Sel | `test_smrmr.py` | 425 | Sparse mRMR with knockoff FDR |

## Execution Order

1. **Tier 0 first** (diagnostics + ceiling): #3, #4, #1, #2. If ceiling < 65%, Tiers 1-6 may be moot.
2. **Tier 8 next** (feature selection): #25, #26. Tells which features matter before adding more.
3. **Tier 3 loss variants**: #12, #13, #14, #15. Quick tests.
4. **Tier 1 target representation**: #5-#9. The big bets — new output heads.
5. **Tier 2 features**: #10, #11. BOCPD + permutation entropy.
6. **Tier 4 training strategy**: #16, #17.
7. **Tier 5 contrastive**: #18, #19. Two-phase pre-training.
8. **Tier 6 architecture**: #20, #21. Neural wavelet + TFT VSN.
9. **Tier 7 conformal**: #22, #23, #24. Wrap the best model from above.

## How to Run

```bash
# Switch to branch and run
git checkout exp2/<name>
source .venv/bin/activate
python -m scripts.feature_tests.test_<name>

# Results auto-saved to ~/.wavecast/audit/feature_tests/<name>_results.json
# Return to master
git checkout master
```

## Key Implementation Patterns

- **Custom training loops**: Use `WaveletGPTNet` directly (not `WaveletGPT` wrapper) for custom losses
- **`_parse_x()` pattern**: Split X into `(ctx, lvl, ac, aux)` matching `WaveletGPT._parse_x()`
- **`evaluate_5_metrics()`**: Shared function in `exp2_helpers.py` — all scripts use it
- **Import convention**: `from __future__` with `# noqa: I001`, then stdlib, torch, scripts.*, wavecast.*
- **`torch.nn.functional` as `f_nn`**: Avoids ruff ambiguity with single-letter `F`

## Pass/Fail Criteria

- **Diagnostics** (#1-4): Information only, no pass/fail
- **Absolute thresholds**: econ_dir >= 63%, transition >= 52%, large_move >= 68%
- **Flat-bias check**: flat% > 40% → automatic FAIL
- **Improvements**: Must beat baseline on at least one metric by meaningful margin
- **Overall verdict**: `verdict_from_metrics()` in `exp2_helpers.py`
