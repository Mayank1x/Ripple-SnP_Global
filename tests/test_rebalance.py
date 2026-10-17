"""Unit tests for Module A: Sentiment-Driven Tactical Index Rebalancer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.rebalance.rebalancer import IndexRebalancer, load_constituents


@pytest.fixture
def rebalancer() -> IndexRebalancer:
    """Fixture providing initialized IndexRebalancer."""
    return IndexRebalancer()


def test_constituents_loaded_correctly():
    """Verify that 15 constituents are loaded with equal base weights summing to 1.0."""
    constituents = load_constituents()
    assert len(constituents) == 15
    tickers = [c["ticker"] for c in constituents]
    assert "AAPL" in tickers
    assert "JPM" in tickers
    assert "XOM" in tickers

    total_weight = sum(c["base_weight"] for c in constituents)
    assert total_weight == pytest.approx(1.0, abs=1e-4)


def test_no_signals_means_no_change(rebalancer):
    """Invariance test: an empty signal list produces exactly zero turnover and unchanged weights."""
    initial_weights = dict(rebalancer.current_weights)
    rec = rebalancer.rebalance(signals=[], save_history=False)

    assert rec.turnover == 0.0
    for t, w in rec.weights.items():
        assert w == pytest.approx(initial_weights[t], abs=1e-6)
    assert all(abs(change) < 1e-6 for change in rec.weight_changes.values())


def test_weights_always_sum_to_one(rebalancer):
    """Verify that portfolio weights strictly sum to 1.0 across diverse signal conditions."""
    dummy_signals = [
        {
            "event_id": "sig_1",
            "title": "Apple reports record iPhone earnings",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": 0.85,
            "impact_score": 8.0,
            "confidence": 0.90,
            "entities": {"companies": [{"ticker": "AAPL", "name": "Apple Inc."}]},
        },
        {
            "event_id": "sig_2",
            "title": "JPMorgan faces credit downgrades on loan losses",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": -0.90,
            "impact_score": 9.0,
            "confidence": 0.95,
            "entities": {"companies": [{"ticker": "JPM", "name": "JPMorgan Chase"}]},
        },
    ]

    rec = rebalancer.rebalance(dummy_signals, save_history=False)
    total_weight = sum(rec.weights.values())
    assert total_weight == pytest.approx(1.0, abs=1e-5)


def test_respects_min_max_bounds(rebalancer):
    """Verify that all weights stay within [min_weight, max_weight] even under extreme shocks."""
    extreme_signals = [
        {
            "event_id": "extreme_pos",
            "title": "Massive AI surge lifts tech giants",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": 1.0,
            "impact_score": 10.0,
            "confidence": 1.0,
            "entities": {"companies": [{"ticker": "NVDA", "name": "Nvidia"}]},
        },
        {
            "event_id": "extreme_neg",
            "title": "Severe fraud investigation hits aerospace manufacturer",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": -1.0,
            "impact_score": 10.0,
            "confidence": 1.0,
            "entities": {"companies": [{"ticker": "BA", "name": "Boeing"}]},
        },
    ]

    # Rebalance multiple iterations to test boundary saturation
    for _ in range(5):
        rec = rebalancer.rebalance(extreme_signals, save_history=False)

    for ticker, w in rec.weights.items():
        assert w >= rebalancer.min_weight - 1e-5, f"{ticker} below min_weight: {w}"
        assert w <= rebalancer.max_weight + 1e-5, f"{ticker} above max_weight: {w}"

    assert sum(rec.weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_respects_turnover_cap(rebalancer):
    """Verify that no single constituent's weight changes by more than turnover_cap in one rebalance."""
    initial_weights = dict(rebalancer.current_weights)

    shock_signal = [
        {
            "event_id": "shock_1",
            "title": "Sudden regulatory fine on major bank",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_score": -1.0,
            "impact_score": 10.0,
            "confidence": 1.0,
            "entities": {"companies": [{"ticker": "JPM", "name": "JPMorgan"}]},
        }
    ]

    rec = rebalancer.rebalance(shock_signal, save_history=False)

    for ticker in rebalancer.tickers:
        delta = abs(rec.weights[ticker] - initial_weights[ticker])
        assert delta <= rebalancer.turnover_cap + 1e-5, (
            f"{ticker} exceeded turnover cap: delta={delta:.5f} > cap={rebalancer.turnover_cap}"
        )


def test_time_decay_fades_old_signals(rebalancer):
    """Verify that older signals produce exponentially smaller tilts due to half-life decay."""
    now = datetime.now(timezone.utc)
    from datetime import timedelta

    signal_fresh = {
        "event_id": "fresh",
        "title": "Fresh positive catalyst",
        "timestamp": now.isoformat(),
        "sentiment_score": 0.8,
        "impact_score": 8.0,
        "confidence": 0.9,
        "entities": {"companies": [{"ticker": "AAPL", "name": "Apple"}]},
    }

    signal_old = {
        "event_id": "old",
        "title": "Old positive catalyst",
        "timestamp": (now - timedelta(days=6)).isoformat(),  # 6 days old = 3 half-lives
        "sentiment_score": 0.8,
        "impact_score": 8.0,
        "confidence": 0.9,
        "entities": {"companies": [{"ticker": "AAPL", "name": "Apple"}]},
    }

    tilts_fresh, _ = rebalancer.compute_signals_tilt([signal_fresh], as_of_time=now)
    tilts_old, _ = rebalancer.compute_signals_tilt([signal_old], as_of_time=now)

    assert tilts_fresh["AAPL"] > tilts_old["AAPL"]
    # 3 half-lives (6 days with half_life=2) should decay by ~8x (0.5^3 = 0.125)
    assert tilts_old["AAPL"] == pytest.approx(tilts_fresh["AAPL"] * 0.125, rel=0.1)
