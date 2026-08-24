"""Honest evaluation of a forward-test prediction ledger.

ECONOMIC metrics only: directional accuracy is the agreement between the
predicted direction and the sign of ``actual_return`` (Phase 7 audit trap —
the symbolic token-ordinal metric must never be reported as performance).

Read-only by convention: callers pass an existing ledger path; nothing here
writes to it.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

TRADING_HOURS_PER_YEAR = 252 * 7  # US equity RTH, used for hourly-bar annualization


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_ledger(path: str) -> list[dict]:
    """Load resolved predictions from a JSONL ledger file."""
    path = os.path.expanduser(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("actual_return") is not None:
                rows.append(rec)
    rows.sort(key=lambda r: r.get("target_timestamp") or "")
    return rows


@dataclass
class LedgerReport:
    n_total: int
    n_resolved: int
    flat_rate: float  # fraction of predictions with predicted_direction == 0
    accuracy: float
    ci_low: float
    ci_high: float
    filtered_accuracy: float | None  # a2i_trade=True subset
    filtered_n: int
    filtered_ci: tuple[float, float]
    per_ticker: dict = field(default_factory=dict)
    per_month: dict = field(default_factory=dict)
    trade_stats: dict = field(default_factory=dict)


def _bucket_accuracy(rows: list[dict]) -> tuple[int, int]:
    """3-class economic directional accuracy: predicted_direction == actual_direction.

    Flat predictions (direction 0) count as valid predictions — they are
    reported separately via the flat-prediction rate, never dropped.
    """
    hits = sum(1 for r in rows if (r["predicted_direction"] or 0) == r["actual_direction"])
    return hits, len(rows)


def evaluate(rows: list[dict], cost_bps: float = 7.0) -> LedgerReport:
    """Full honest evaluation of resolved ledger rows."""
    n = len(rows)
    flat = sum(1 for r in rows if r.get("predicted_direction") == 0)
    hits, _ = _bucket_accuracy(rows)
    lo, hi = wilson_ci(hits, n)

    filt = [r for r in rows if r.get("a2i_trade")]
    fhits, _ = _bucket_accuracy(filt)

    per_ticker = {}
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)
    for ticker, trows in sorted(by_ticker.items()):
        th, tn = _bucket_accuracy(trows)
        tlo, thi = wilson_ci(th, tn)
        per_ticker[ticker] = {"n": tn, "accuracy": th / tn if tn else 0.0, "ci": (tlo, thi)}

    per_month = {}
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ts = r.get("target_timestamp") or ""
        month = ts[:7] if ts else "unknown"
        by_month[month].append(r)
    for month, mrows in sorted(by_month.items()):
        mh, mn = _bucket_accuracy(mrows)
        per_month[month] = {"n": mn, "accuracy": mh / mn if mn else 0.0}

    stats = trade_stats(rows, cost_bps=cost_bps)
    return LedgerReport(
        n_total=n,
        n_resolved=n,
        flat_rate=flat / n if n else 0.0,
        accuracy=hits / n if n else 0.0,
        ci_low=lo,
        ci_high=hi,
        filtered_accuracy=fhits / len(filt) if filt else None,
        filtered_n=len(filt),
        filtered_ci=wilson_ci(fhits, len(filt)),
        per_ticker=per_ticker,
        per_month=per_month,
        trade_stats=stats,
    )


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(TRADING_HOURS_PER_YEAR)


def trade_stats(rows: list[dict], cost_bps: float = 7.0) -> dict:
    """Per-position returns for model + baselines over the same opportunity set.

    Each resolved row is one hourly opportunity. A strategy holds
    ``direction * actual_return`` each row; a round-trip cost (bps) is charged
    whenever its position changes.
    """
    cost = cost_bps / 10_000

    def persistence_positions():
        last_dir: dict[str, int] = {}
        out = []
        for r in rows:
            t = r["ticker"]
            out.append(last_dir.get(t, 0))
            d = r["actual_return"]
            last_dir[t] = 1 if d > 0 else -1 if d < 0 else 0
        return out

    positions = {
        "model": [(r["predicted_direction"] or 0) for r in rows],
        "always_up": [1] * len(rows),
        "always_down": [-1] * len(rows),
        "persistence": persistence_positions(),
        "model_a2i_filtered": [
            (r["predicted_direction"] or 0) if r.get("a2i_trade") else 0 for r in rows
        ],
    }
    return {name: run_positions(rows, ps, cost) for name, ps in positions.items()}


def run_positions(rows: list[dict], positions: list[int], cost: float) -> dict:
    prev_pos = 0
    rets = []
    wins = 0
    for r, pos in zip(rows, positions, strict=True):
        ret = pos * r["actual_return"]
        if pos != prev_pos:
            ret -= cost  # round trip on change (entry+exit approximated as one charge)
        elif pos != 0:
            ret -= cost / 2  # holding: exit-side only
        prev_pos = pos
        rets.append(ret)
        if ret > 0:
            wins += 1
    n = len(rets)
    return {
        "total_return": sum(rets),
        "expectancy": sum(rets) / n if n else 0.0,
        "sharpe_annualized": _sharpe(rets),
        "win_rate": wins / n if n else 0.0,
        "n": n,
    }


def format_report(rep: LedgerReport) -> str:
    lines = []
    lines.append(f"resolved predictions: {rep.n_resolved}")
    lines.append(
        f"economic directional accuracy: {rep.accuracy:.1%} "
        f"[{rep.ci_low:.1%}, {rep.ci_high:.1%}] (n={rep.n_resolved})"
    )
    lines.append(f"flat-prediction rate: {rep.flat_rate:.1%}")
    if rep.filtered_accuracy is not None:
        lines.append(
            f"A2i-filtered accuracy: {rep.filtered_accuracy:.1%} "
            f"[{rep.filtered_ci[0]:.1%}, {rep.filtered_ci[1]:.1%}] (n={rep.filtered_n})"
        )
    lines.append("\nper-ticker:")
    for t, s in rep.per_ticker.items():
        lines.append(f"  {t:>6}: {s['accuracy']:.1%} (n={s['n']})")
    lines.append("\nper-month:")
    for m, s in rep.per_month.items():
        lines.append(f"  {m}: {s['accuracy']:.1%} (n={s['n']})")
    lines.append(f"\ntrade stats (cost {7.0:.0f}bps):")
    for name, s in rep.trade_stats.items():
        lines.append(
            f"  {name:>18}: total={s['total_return']:+.2%} expectancy={s['expectancy']:+.5f} "
            f"sharpe={s['sharpe_annualized']:+.2f} win={s['win_rate']:.1%}"
        )
    return "\n".join(lines)
