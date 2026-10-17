"""Backtest Script for Sentiment-Driven Tactical Index Rebalancer (Module A).

Replays signals from the Ripple risk engine over the evaluation window:
1. Rebalances the 15-stock mock index daily using decayed news/social tilts.
2. Compares cumulative returns, volatility, Sharpe ratio, and max drawdown:
   - Tilted Sentiment Index (Net of 10 bps transaction costs)
   - Plain Equal-Weight Benchmark (1/15th per stock)
3. Outputs performance metrics and saves time-series to data/rebalance/backtest_results.json.
"""

from __future__ import annotations

import json
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ingest.helpers import load_config
from src.rebalance.rebalancer import IndexRebalancer, load_constituents
from src.scoring.event_study import fetch_price_history


def run_rebalance_backtest(
    window_days: int = 30,
    cost_bps: float = 10.0,
    save_results: bool = True,
) -> dict[str, Any]:
    """Execute backtest comparing tilted vs equal-weight index."""
    cfg = load_config()
    reb_cfg = cfg.get("rebalance", {})
    cost_bps = float(reb_cfg.get("transaction_cost_bps", cost_bps))

    constituents = load_constituents()
    tickers = [c["ticker"] for c in constituents]
    n_stocks = len(tickers)
    base_weight = 1.0 / n_stocks

    # 1. Load price histories for all constituents
    price_dict: dict[str, pd.Series] = {}
    for t in tickers:
        pdf = fetch_price_history(t, period="1y", use_cache=True)
        if not pdf.empty and "close" in pdf.columns:
            pdf.index = pd.to_datetime(pdf.index).tz_localize(None)
            price_dict[t] = pdf["close"]

    if not price_dict or len(price_dict) < n_stocks:
        raise RuntimeError("Could not load price series for all 15 index constituents.")

    prices_df = pd.DataFrame(price_dict).dropna().sort_index()
    if len(prices_df) < 10:
        raise RuntimeError(f"Insufficient overlapping price rows: {len(prices_df)}")

    # Restrict to evaluation window
    prices_window = prices_df.iloc[-window_days:]
    returns_df = prices_window.pct_change().dropna()
    dates = returns_df.index

    # 2. Load signals
    signals_file = Path("data/signals/signals.jsonl")
    all_signals: list[dict[str, Any]] = []
    if signals_file.exists():
        with open(signals_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_signals.append(json.loads(line))
                    except Exception:
                        continue

    # 3. Daily simulation loop
    rebalancer = IndexRebalancer(cfg)
    current_weights = {t: base_weight for t in tickers}

    tilted_daily_returns: list[float] = []
    bench_daily_returns: list[float] = []
    turnover_series: list[float] = []
    cost_series: list[float] = []
    weight_history: list[dict[str, Any]] = []

    # Clear previous history if fresh run
    if save_results and rebalancer.history_file.exists():
        rebalancer.history_file.unlink()

    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")
        dt_utc = dt.replace(tzinfo=timezone.utc)

        # Filter signals available up to the start of this trading day
        available_signals = [
            s for s in all_signals
            if s.get("timestamp", "")[:10] <= dt_str
        ]

        # Rebalance portfolio
        rec = rebalancer.rebalance(
            signals=available_signals,
            as_of_time=dt_utc,
            save_history=save_results,
        )
        new_weights = rec.weights
        turnover = rec.turnover

        # Calculate daily returns
        day_rets = returns_df.loc[dt]
        r_bench = float(sum(base_weight * day_rets[t] for t in tickers))
        r_tilted_gross = float(sum(current_weights[t] * day_rets[t] for t in tickers))

        # Apply transaction cost
        tx_cost = turnover * (cost_bps / 10000.0)
        r_tilted_net = r_tilted_gross - tx_cost

        tilted_daily_returns.append(r_tilted_net)
        bench_daily_returns.append(r_bench)
        turnover_series.append(turnover)
        cost_series.append(tx_cost)

        weight_snapshot = {"date": dt_str, "weights": dict(new_weights)}
        weight_history.append(weight_snapshot)

        current_weights = dict(new_weights)

    # 4. Compute performance metrics
    cum_tilted = np.cumprod(1.0 + np.array(tilted_daily_returns))
    cum_bench = np.cumprod(1.0 + np.array(bench_daily_returns))

    total_ret_tilted = float(cum_tilted[-1] - 1.0)
    total_ret_bench = float(cum_bench[-1] - 1.0)

    # Volatility and Sharpe
    vol_tilted = float(np.std(tilted_daily_returns) * np.sqrt(252))
    vol_bench = float(np.std(bench_daily_returns) * np.sqrt(252))

    sharpe_tilted = float((np.mean(tilted_daily_returns) * 252) / (vol_tilted + 1e-9))
    sharpe_bench = float((np.mean(bench_daily_returns) * 252) / (vol_bench + 1e-9))

    # Maximum Drawdowns
    def _calc_mdd(cum_arr: np.ndarray) -> float:
        peaks = np.maximum.accumulate(cum_arr)
        dds = (cum_arr - peaks) / peaks
        return float(np.min(dds))

    mdd_tilted = _calc_mdd(cum_tilted)
    mdd_bench = _calc_mdd(cum_bench)

    total_turnover = float(sum(turnover_series))
    total_tx_costs = float(sum(cost_series))

    results: dict[str, Any] = {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "cum_tilted": [round(float(x), 4) for x in cum_tilted],
        "cum_bench": [round(float(x), 4) for x in cum_bench],
        "tilted_daily_returns": [round(float(x), 5) for x in tilted_daily_returns],
        "bench_daily_returns": [round(float(x), 5) for x in bench_daily_returns],
        "turnover_series": [round(float(x), 4) for x in turnover_series],
        "metrics": {
            "window_days": len(dates),
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date": dates[-1].strftime("%Y-%m-%d"),
            "tilted_total_return": round(total_ret_tilted * 100.0, 2),
            "benchmark_total_return": round(total_ret_bench * 100.0, 2),
            "tilted_volatility": round(vol_tilted * 100.0, 2),
            "benchmark_volatility": round(vol_bench * 100.0, 2),
            "tilted_sharpe": round(sharpe_tilted, 2),
            "benchmark_sharpe": round(sharpe_bench, 2),
            "tilted_max_drawdown": round(mdd_tilted * 100.0, 2),
            "benchmark_max_drawdown": round(mdd_bench * 100.0, 2),
            "total_turnover": round(total_turnover, 3),
            "total_transaction_costs_bps": round(total_tx_costs * 10000.0, 1),
            "cost_assumption_bps": cost_bps,
        },
        "latest_weights": current_weights,
        "weight_history": weight_history,
    }

    if save_results:
        out_file = Path("data/rebalance/backtest_results.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def print_backtest_report(res: dict[str, Any]) -> None:
    """Print honest backtest comparison summary."""
    m = res["metrics"]
    print("=" * 72)
    print("  RIPPLE MODULE A: SENTIMENT-DRIVEN TACTICAL INDEX REBALANCER")
    print(f"  Backtest Window: {m['start_date']} to {m['end_date']} ({m['window_days']} trading days)")
    print("=" * 72)
    print(f"{'Metric':<32} {'Tilted Index (Net)':<20} {'Equal-Weight Benchmark':<20}")
    print("-" * 72)
    print(f"{'Cumulative Return':<32} {m['tilted_total_return']:>6.2f}%              {m['benchmark_total_return']:>6.2f}%")
    print(f"{'Annualized Volatility':<32} {m['tilted_volatility']:>6.2f}%              {m['benchmark_volatility']:>6.2f}%")
    print(f"{'Sharpe Ratio':<32} {m['tilted_sharpe']:>6.2f}               {m['benchmark_sharpe']:>6.2f}")
    print(f"{'Maximum Drawdown':<32} {m['tilted_max_drawdown']:>6.2f}%              {m['benchmark_max_drawdown']:>6.2f}%")
    print(f"{'Cumulative Turnover':<32} {m['total_turnover']:>6.2f}               0.00")
    print(f"{'Total Cost Incurred':<32} {m['total_transaction_costs_bps']:>6.1f} bps            0.0 bps")
    print("-" * 72)

    diff = m["tilted_total_return"] - m["benchmark_total_return"]
    print("\nHonest Assessment:")
    if diff > 0.05:
        print(f"  * The sentiment tilt OUTPERFORMED the equal-weight index by +{diff:.2f}% net of {m['cost_assumption_bps']:.0f} bps trading costs.")
    elif diff < -0.05:
        print(f"  * The sentiment tilt UNDERPERFORMED the equal-weight index by {diff:.2f}% net of {m['cost_assumption_bps']:.0f} bps trading costs.")
    else:
        print(f"  * The sentiment tilt tracked the equal-weight index closely (difference: {diff:+.2f}%).")

    print(f"  * Trading Cost Assumption: {m['cost_assumption_bps']:.0f} bps per unit of one-way turnover.")
    print("  * Saved full results to: data/rebalance/backtest_results.json")
    print("=" * 72)


if __name__ == "__main__":
    results = run_rebalance_backtest(window_days=30, cost_bps=10.0, save_results=True)
    print_backtest_report(results)
