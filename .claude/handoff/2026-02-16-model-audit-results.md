# Model Audit Complete — Critical Findings

**Date**: 2026-02-16
**Script**: `scripts/model_audit.py` (13 analyses, 3.9s runtime)
**Results**: `~/.wavecast/audit/model_audit_results.json`

## TL;DR

**The model has NO tradeable edge.** The 95.8% directional accuracy is purely symbolic (compares SAX word ordinals, not price direction). Economic directional accuracy is ~46% — worse than a coin flip. The 62.4% token accuracy is real but economically meaningless.

## Key Findings

### Phase 0: Temporal Mapping Validated
- Bar mappings look correct: AAPL L1 maps to 2-bar spans (~1.3% returns), L5 maps to 32-bar spans (~0.08% returns)
- 11,524 of 11,603 test samples have valid mapped returns

### Q1: Economic Directional Accuracy is ~46% (Below Random)
| Threshold | N | Econ Dir | Symbolic |
|-----------|---|----------|----------|
| 0.00% | 11444 | 0.465 | 0.625 |
| 0.10% | 9586 | 0.466 | 0.634 |
| 0.50% | 4977 | 0.473 | 0.689 |
| 1.00% | 2715 | 0.478 | 0.763 |

Symbolic accuracy INCREASES with threshold (larger moves have more persistent tokens). Economic accuracy stays flat at ~47%.

### Q2: 57% of Returns are < 0.5%
Level 5 has the largest returns (only 19.6% < 0.5%). Levels 1-2 are dominated by noise (71%/58% < 0.5%).

### Q3: Level 5 Tokens are 82.6% Identical to Previous
Mean run length of 7 tokens. COP is 94.2% same. The model's high Level 5 accuracy (87.4%) is largely persistence.

### Q4: Persistence Decomposition (THE Key Finding)
| Bucket | Count | % | Token Acc | Econ Dir |
|--------|-------|---|-----------|----------|
| PERSISTENCE | 5242 | 45.2% | 79.2% | 46.0% |
| CHANGE | 6361 | 54.8% | 48.5% | 47.1% |

- Model predicts "same as last" 45% of the time with 79% accuracy (high, but just reflects token persistence)
- When it predicts a CHANGE, accuracy drops to 48.5% (near random)
- Economic directional accuracy is ~46% for BOTH buckets

### Q5: Overlapping Windows Don't Inflate Metrics
Token accuracy is stable at 62.1-63.1% across all subsampling strides (1, 2, 4, 8, 16). The metric is robust.

### Q6: Negative Sharpe (-0.56 Fixed, +0.18 Kelly)
| Sizing | Sharpe | Return | Max DD | Win Rate | Trades |
|--------|--------|--------|--------|----------|--------|
| Fixed | -0.5573 | -99.91% | 99.95% | 0.406 | 1100 |
| Kelly | +0.1808 | +5.18% | 1.25% | 0.007 | 1100 |

Kelly appears positive but only because it sizes most positions at ~0% (linear fallback with tiny positions). Fixed sizing reveals the true negative edge. 94% of trades exceed the 7bps cost hurdle in gross terms — the issue isn't costs, it's wrong direction.

### Q7: Can't Predict Direction Changes
| Type | N | Accuracy |
|------|---|----------|
| Transition | 4917 | 50.1% (random) |
| Continuation | 5194 | 43.0% (below random) |

### Q8: Model Uses Only 70 of 102 Tokens for Predictions
Mean entropy: 0.962 / 4.625 max. Very concentrated predictions. Token 2 is the most common wrong prediction (likely UNK or a dominant SAX word).

### Q9: 6 Tokens Account for 50% of Training Targets
| Coverage | Tokens Needed |
|----------|---------------|
| 50% | 6 |
| 75% | 21 |
| 90% | 46 |
| 95% | 65 |

Effective vocabulary is tiny despite 100 max.

### Q10: Confidence Calibration is Perfect for Tokens, Useless for Direction
| Decile | Conf | Token Acc | Econ Dir |
|--------|------|-----------|----------|
| 1 | 0.39 | 36.4% | 46.3% |
| 5 | 0.74 | 62.6% | 48.8% |
| 10 | 0.97 | 93.2% | 44.5% |

Token accuracy scales beautifully with confidence (36%→93%). Economic directional accuracy is FLAT (~46%) across all deciles. Confidence is meaningless for trading.

### Q11: All Levels Have Negative Sharpe
| Group | Sharpe | Trades |
|-------|--------|--------|
| Level 5 only | -0.88 | 39 |
| Levels 1+2 | -0.51 | 1060 |
| All levels | -0.56 | 1100 |

Level 5 is worst despite highest token accuracy.

### Q12: Model Doesn't Beat Random
- Real Sharpe: -0.56
- Random Sharpe: -0.61 ± 0.05 (P5=-0.70, P95=-0.54)
- Percentile of real: 84th (better than average random, but doesn't exceed P95)
- Model is slightly less bad than random, but still negative

### Q13: Fixed vs Kelly — No Edge Either Way
Both negative. Kelly's apparent positive is a position sizing artifact (near-zero positions).

## What This Means

1. **The model learns SAX token statistics, not price dynamics.** Token prediction is a statistical pattern-matching task within the SAX representation. These patterns have zero economic content.

2. **The 95.8% directional accuracy was always meaningless.** It compared SAX word ordinals — when `predicted_word == actual_word`, that counted as "correct direction." With 45% persistence predictions and high token repetition, this inflated to 95%+.

3. **Level 5's 87.4% token accuracy is persistence.** Tokens change only 17.4% of the time. Predicting "same" gives high accuracy but zero economic value.

4. **The signal generation framework works correctly** — the problem is upstream in the representation, not in the signal/backtest pipeline.

## Recommended Next Steps

Based on audit results, the critical path is:

1. **Rethink the target variable.** SAX tokens don't encode price direction. Options:
   - Directly predict signed returns (regression)
   - Predict return quantiles (classification)
   - Use SAX on returns series instead of price/coefficient levels
   - Predict direction of CHANGE (delta between consecutive coefficients), not the coefficient level

2. **Magnitude recovery.** The model can't distinguish a 0.01% move from a 5% move within the same SAX bucket. Consider:
   - Continuous targets alongside discrete tokens
   - Separate magnitude prediction head
   - Use reconstruction-aware training loss

3. **Don't optimize the current pipeline further.** The representation is the bottleneck, not the model architecture, position sizing, or signal generation.

## Files
- `scripts/model_audit.py` — standalone audit script (~550 LOC)
- `~/.wavecast/audit/model_audit_results.json` — full JSON results
- Plan: `.claude/plans/humble-marinating-cloud.md`
