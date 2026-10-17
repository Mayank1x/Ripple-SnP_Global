"""Event Study Analysis, Market Model Estimation, and Factor Series Changes.

Calculates:
1. Abnormal returns (AR) and cumulative abnormal returns (CAR) at day 0, +1, +3
   against a market model (S&P 500 ^GSPC or Nifty 50 ^NSEI).
2. Volatility changes (post-event vs pre-event window).
3. Factor series daily changes: Equity, Sectors, 10Y Yield, FX, Brent, VIX, Credit Spread (HYG vs LQD).
4. Full offline disk caching of yfinance price tables under data/raw/prices/.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scoring.schema import Event

# Sector to US ETF Proxy mapping
SECTOR_ETF_MAP: dict[str, str] = {
    "Financials": "XLF",
    "Banking": "XLF",
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Energy": "XLE",
    "Oil & Gas": "XLE",
    "Healthcare": "XLV",
    "Pharmaceuticals": "XLV",
    "Industrials": "XLI",
    "Aerospace & Defense": "XLI",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Automotive": "XLY",
    "Consumer Staples": "XLP",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Semiconductors": "SMH",
}

# Country to Index / ETF Proxy mapping
COUNTRY_INDEX_MAP: dict[str, str] = {
    "India": "^NSEI",
    "United States": "^GSPC",
    "United Kingdom": "^FTSE",
    "Germany": "^GDAXI",
    "Japan": "^N225",
    "China": "ASHR",
    "Hong Kong": "^HSI",
}

# Key Macro Factor Series
FACTOR_TICKERS: dict[str, str] = {
    "equity_us": "^GSPC",
    "equity_in": "^NSEI",
    "sector_fin": "XLF",
    "sector_tech": "XLK",
    "sector_energy": "XLE",
    "sector_realestate": "XLRE",
    "yield_10y": "^TNX",
    "fx_usdinr": "USDINR=X",
    "fx_dxy": "DX-Y.NYB",
    "brent_oil": "BZ=F",
    "vix": "^VIX",
    "credit_hyg": "HYG",
    "credit_lqd": "LQD",
}


def _get_price_cache_dir() -> Path:
    """Return the price cache directory."""
    cache_dir = Path("data/raw/prices")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _clean_symbol(symbol: str) -> str:
    """Convert ticker symbol to safe filename string."""
    return re.sub(r"[^\w\-]", "_", symbol)


def fetch_price_history(
    symbol: str,
    period: str = "1y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily price history from yfinance with local CSV caching.

    Args:
        symbol: Ticker symbol (e.g., 'JPM', '^GSPC', 'XLF').
        period: History period (default '1y').
        use_cache: If True, reads from data/raw/prices/{clean_symbol}.csv when available.

    Returns:
        DataFrame with lowercase columns ['open', 'high', 'low', 'close', 'volume'] indexed by datetime Date.
    """
    clean_sym = _clean_symbol(symbol)
    cache_file = _get_price_cache_dir() / f"{clean_sym}.csv"

    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            if not df.empty and len(df) >= 20:
                return df
        except Exception:
            pass

    # Download from yfinance
    try:
        import yfinance as yf

        raw_df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        # Handle modern yfinance MultiIndex columns (e.g. ('Close', 'JPM'))
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [str(c[0]).lower() for c in raw_df.columns]
        else:
            raw_df.columns = [str(c).lower() for c in raw_df.columns]

        # Ensure datetime index
        raw_df.index = pd.to_datetime(raw_df.index)
        raw_df.index.name = "Date"

        # Cache to disk
        try:
            raw_df.to_csv(cache_file)
        except Exception:
            pass

        return raw_df
    except Exception as exc:
        print(f"  [event_study] Failed to fetch prices for {symbol}: {exc}")
        return pd.DataFrame()


def resolve_event_asset(event: Event | dict[str, Any]) -> tuple[str, str]:
    """Map event entities to primary asset ticker and corresponding market benchmark.

    Returns:
        (asset_ticker, benchmark_ticker)
    """
    entities = event.entities if isinstance(event, Event) else event.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}

    # 1. Company Ticker
    companies = entities.get("companies", [])
    if companies and isinstance(companies, list):
        comp = companies[0]
        ticker = comp.get("ticker") if isinstance(comp, dict) else None
        country = comp.get("country", "") if isinstance(comp, dict) else ""
        if ticker:
            # Map Indian stocks to .NS if needed
            if "india" in str(country).lower() and not ticker.endswith(".NS") and not ticker.startswith("^"):
                return f"{ticker}.NS", "^NSEI"
            bench = "^NSEI" if "india" in str(country).lower() else "^GSPC"
            return ticker, bench

    # 2. Sector ETF
    sectors = entities.get("sectors", [])
    if sectors and isinstance(sectors, list):
        sec = sectors[0]
        if sec in SECTOR_ETF_MAP:
            return SECTOR_ETF_MAP[sec], "^GSPC"

    # 3. Country Index
    countries = entities.get("countries", [])
    if countries and isinstance(countries, list):
        cntry = countries[0]
        if cntry in COUNTRY_INDEX_MAP:
            bench = COUNTRY_INDEX_MAP[cntry]
            return bench, bench

    # Default fallback
    return "^GSPC", "^GSPC"


def compute_event_study(
    event: Event | dict[str, Any],
    estimation_window: int = 60,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Perform event study for a single real-world event.

    Estimates market model on pre-event window [T-75, T-15], computes abnormal returns
    at day 0, +1, +3, and measures volatility change.

    Returns:
        Dictionary containing abnormal returns, volatility change, and realized move.
    """
    ts_str = event.first_seen if isinstance(event, Event) else event.get("timestamp") or event.get("first_seen")
    event_dt = pd.to_datetime(ts_str or "2026-10-15").tz_localize(None)

    asset_sym, bench_sym = resolve_event_asset(event)

    df_asset = fetch_price_history(asset_sym, use_cache=use_cache)
    df_bench = fetch_price_history(bench_sym, use_cache=use_cache)

    # Fallback to benchmark if asset data unavailable
    if df_asset.empty or "close" not in df_asset.columns:
        df_asset = df_bench
        asset_sym = bench_sym

    if df_bench.empty or "close" not in df_bench.columns:
        # Synthetic fallback if totally offline with no cache
        return {
            "asset": asset_sym,
            "benchmark": bench_sym,
            "event_date": event_dt.strftime("%Y-%m-%d"),
            "alpha": 0.0,
            "beta": 1.0,
            "ar_0": 0.0,
            "ar_1": 0.0,
            "ar_3": 0.0,
            "car_0_1": 0.0,
            "car_0_3": 0.0,
            "vol_change": 0.0,
            "realized_move": 0.015,
        }

    # Normalize tz on index
    df_asset.index = pd.to_datetime(df_asset.index).tz_localize(None)
    df_bench.index = pd.to_datetime(df_bench.index).tz_localize(None)

    # Compute percentage returns
    r_asset = df_asset["close"].pct_change().dropna()
    r_bench = df_bench["close"].pct_change().dropna()

    aligned = pd.DataFrame({"r_asset": r_asset, "r_bench": r_bench}).dropna()
    if len(aligned) < 20:
        return {
            "asset": asset_sym,
            "benchmark": bench_sym,
            "event_date": event_dt.strftime("%Y-%m-%d"),
            "alpha": 0.0,
            "beta": 1.0,
            "ar_0": 0.0,
            "ar_1": 0.0,
            "ar_3": 0.0,
            "car_0_1": 0.0,
            "car_0_3": 0.0,
            "vol_change": 0.0,
            "realized_move": 0.015,
        }

    # 1. Locate day 0 (nearest trading day on or after event_dt)
    trade_dates = aligned.index
    future_dates = trade_dates[trade_dates >= event_dt]
    if len(future_dates) > 0:
        day_0_date = future_dates[0]
        day_0_idx = int(aligned.index.get_loc(day_0_date))
    else:
        # Event is on latest date or beyond; clamp to last available trading day
        day_0_idx = len(aligned) - 1
        day_0_date = aligned.index[day_0_idx]

    # Trading days +1 and +3
    day_1_idx = min(len(aligned) - 1, day_0_idx + 1)
    day_3_idx = min(len(aligned) - 1, day_0_idx + 3)

    # 2. Market Model Estimation window [T-75, T-15]
    est_start = max(0, day_0_idx - 75)
    est_end = max(0, day_0_idx - 15)

    if est_end - est_start >= 15:
        est_data = aligned.iloc[est_start:est_end]
        cov = float(np.cov(est_data["r_bench"], est_data["r_asset"])[0][1])
        var_bench = float(np.var(est_data["r_bench"]))
        beta = cov / var_bench if var_bench > 1e-8 else 1.0
        alpha = float(np.mean(est_data["r_asset"]) - beta * np.mean(est_data["r_bench"]))
    else:
        alpha, beta = 0.0, 1.0

    # 3. Abnormal returns: AR_t = R_i - (alpha + beta * R_m)
    def _ar_at(idx: int) -> float:
        r_i = float(aligned["r_asset"].iloc[idx])
        r_m = float(aligned["r_bench"].iloc[idx])
        return r_i - (alpha + beta * r_m)

    ar_0 = _ar_at(day_0_idx)
    ar_1 = _ar_at(day_1_idx) if day_1_idx != day_0_idx else ar_0
    ar_3 = _ar_at(day_3_idx) if day_3_idx != day_0_idx else ar_1

    # Cumulative abnormal return
    car_0_1 = (1.0 + ar_0) * (1.0 + ar_1) - 1.0 if day_1_idx != day_0_idx else ar_0
    car_0_3 = (1.0 + ar_0) * (1.0 + ar_1) * (1.0 + ar_3) - 1.0

    # 4. Volatility change: post-event [T0, T0+5] vs pre-event [T0-25, T0-1]
    pre_window = aligned["r_asset"].iloc[max(0, day_0_idx - 25) : max(1, day_0_idx)]
    post_window = aligned["r_asset"].iloc[day_0_idx : min(len(aligned), day_0_idx + 6)]

    pre_vol = float(np.std(pre_window)) if len(pre_window) >= 2 else 0.015
    post_vol = float(np.std(post_window)) if len(post_window) >= 2 else 0.015
    vol_change = post_vol - pre_vol

    # Composite realized absolute move: |CAR| + 0.35 * max(0, vol_change)
    realized_move = abs(car_0_1) + 0.35 * max(0.0, vol_change)

    return {
        "asset": asset_sym,
        "benchmark": bench_sym,
        "event_date": day_0_date.strftime("%Y-%m-%d"),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "ar_0": round(ar_0, 6),
        "ar_1": round(ar_1, 6),
        "ar_3": round(ar_3, 6),
        "car_0_1": round(car_0_1, 6),
        "car_0_3": round(car_0_3, 6),
        "vol_change": round(vol_change, 6),
        "realized_move": round(realized_move, 6),
    }


def compute_factor_daily_changes(period: str = "1y", use_cache: bool = True) -> pd.DataFrame:
    """Download and compute daily percentage changes across core macro risk factors.

    Factors: Equity (^GSPC, ^NSEI), Sector ETFs (XLF, XLK, XLE, XLRE), 10Y Yield (^TNX),
             FX (USDINR=X, DX-Y.NYB), Brent (BZ=F), VIX (^VIX), Credit Spread (HYG vs LQD).

    Returns:
        DataFrame of daily returns indexed by Date.
    """
    cache_file = _get_price_cache_dir() / "factor_daily_changes.csv"
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            if not df.empty and len(df) >= 30:
                return df
        except Exception:
            pass

    series_dict: dict[str, pd.Series] = {}

    for factor_name, ticker in FACTOR_TICKERS.items():
        pdf = fetch_price_history(ticker, period=period, use_cache=use_cache)
        if not pdf.empty and "close" in pdf.columns:
            pdf.index = pd.to_datetime(pdf.index).tz_localize(None)
            ret = pdf["close"].pct_change()
            series_dict[factor_name] = ret

    if not series_dict:
        return pd.DataFrame()

    factor_df = pd.DataFrame(series_dict).dropna(how="all")

    # Add Credit Spread return: HYG return - LQD return
    if "credit_hyg" in factor_df.columns and "credit_lqd" in factor_df.columns:
        factor_df["credit_spread_diff"] = factor_df["credit_hyg"] - factor_df["credit_lqd"]

    try:
        factor_df.to_csv(cache_file)
    except Exception:
        pass

    return factor_df
