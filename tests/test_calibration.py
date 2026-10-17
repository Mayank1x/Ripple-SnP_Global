"""Unit tests for Phase 4: Event study market model, factor series, and isotonic calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scoring.calibration import ImpactCalibrator, get_calibrator
from src.scoring.event_study import (
    compute_event_study,
    compute_factor_daily_changes,
    fetch_price_history,
    resolve_event_asset,
)
from src.scoring.pipeline import process_signals
from src.scoring.schema import Event, Signal


def test_resolve_event_asset_mapping():
    """Verify entity tickers, sectors, and countries resolve to appropriate market assets."""
    # Company ticker
    ev_comp = {"entities": {"companies": [{"ticker": "JPM", "name": "JPMorgan Chase"}]}}
    asset, bench = resolve_event_asset(ev_comp)
    assert asset == "JPM"
    assert bench == "^GSPC"

    # Sector mapping
    ev_sec = {"entities": {"sectors": ["Financials"]}}
    asset_sec, bench_sec = resolve_event_asset(ev_sec)
    assert asset_sec == "XLF"
    assert bench_sec == "^GSPC"

    # Indian entity mapping
    ev_in = {"entities": {"companies": [{"ticker": "RELIANCE", "country": "India"}]}}
    asset_in, bench_in = resolve_event_asset(ev_in)
    assert asset_in.endswith(".NS") or asset_in == "^NSEI"
    assert bench_in == "^NSEI"


def test_fetch_price_history_cached():
    """Verify price history is retrieved with required columns and datetime index."""
    df = fetch_price_history("^GSPC", use_cache=True)
    assert not df.empty
    assert "close" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) >= 20


def test_compute_event_study_market_model():
    """Verify market model estimation produces abnormal returns and volatility change."""
    dummy_event = {
        "event_id": "test_ev_study",
        "first_seen": "2026-10-15T12:00:00Z",
        "entities": {"sectors": ["Financials"]},
    }
    res = compute_event_study(dummy_event, use_cache=True)

    assert "asset" in res
    assert "benchmark" in res
    assert "beta" in res
    assert "ar_0" in res
    assert "car_0_1" in res
    assert "vol_change" in res
    assert "realized_move" in res
    assert res["realized_move"] >= 0.0


def test_compute_factor_daily_changes():
    """Verify factor series matrix contains daily returns for core risk factors."""
    factor_df = compute_factor_daily_changes(use_cache=True)
    assert not factor_df.empty
    assert "equity_us" in factor_df.columns
    assert "credit_spread_diff" in factor_df.columns
    assert len(factor_df) >= 30


def test_isotonic_calibrator_time_split_fit():
    """Verify ImpactCalibrator fits on chronological time split and outputs 1.0 to 10.0."""
    # Synthetic chronological pairs
    pairs = []
    dates = pd.date_range("2025-01-01", periods=20, freq="10D")
    for i, dt in enumerate(dates):
        raw_imp = 0.1 + (i / 20.0) * 0.8
        realized = 0.005 + raw_imp * 0.03 + np.random.normal(0, 0.002)
        pairs.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "raw_impact": raw_imp,
                "realized_move": max(0.002, realized),
            }
        )

    calib = ImpactCalibrator()
    metrics = calib.fit(pairs, time_split_ratio=0.70)

    assert metrics["n_total"] == 20
    assert metrics["n_train"] == 14
    assert metrics["n_test"] == 6
    assert "spearman_rho" in metrics
    assert "p_value" in metrics

    # Test monotonic prediction
    low_score, meta_low = calib.predict_score(0.10)
    high_score, meta_high = calib.predict_score(0.90)

    assert 1.0 <= low_score <= 10.0
    assert 1.0 <= high_score <= 10.0
    assert high_score >= low_score
    assert "predicted_abnormal_move" in meta_high


def test_signals_carry_calibrated_metadata():
    """Verify Signal objects outputted by pipeline contain calibrated fields."""
    sig = Signal(
        event_id="sig_calib_test",
        timestamp="2026-10-16T12:00:00Z",
        entities={"sectors": ["Energy"]},
        sentiment_score=-0.80,
        event_type="Credit Event",
        impact_score=8.5,
        confidence=0.92,
        divergence_flag=False,
        evidence=[],
        model_versions={"scoring": "1.0.0"},
        raw_impact=0.75,
        calibrated_impact=0.0125,
        calibration_metadata={"model": "IsotonicRegression"},
    )
    d = sig.to_dict()
    assert "raw_impact" in d
    assert "calibrated_impact" in d
    assert "calibration_metadata" in d

    restored = Signal.from_dict(d)
    assert restored.impact_score == 8.5
    assert restored.raw_impact == 0.75
    assert restored.calibration_metadata.get("model") == "IsotonicRegression"
