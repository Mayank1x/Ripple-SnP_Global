"""Tests for Phase 6 - Stress Engine (Module B).

Verifies:
1. Rates up lowers bond value.
2. Oil up hurts the right sectors (energy-consuming sectors suffer margin squeeze).
3. Zero shock gives zero change (exact invariant).
4. Analog event matching via FAISS and prior shrinkage when analogs < N.
5. End-to-end stress execution on a stored high-impact signal with before/after and confidence band.
"""

import json
from pathlib import Path

import pytest

from src.portfolio.schema import Portfolio, Position
from src.stress.revalue import revalue_portfolio, revalue_position
from src.stress.run import stress
from src.stress.shocks import compute_factor_shocks

# =====================================================================
# 1. Financial Invariant Tests
# =====================================================================

def test_rates_up_lowers_bond_value():
    """Verify that an upward shock in benchmark yields (dy > 0) strictly reduces bond value."""
    bond = Position(
        position_id="BOND_RATE_TEST",
        asset_class="Bond",
        instrument_type="Corporate Bond",
        counterparty="Apex Global Corp",
        sector="Industrials",
        country="United States",
        rating="BBB",
        notional=10_000_000.0,
        price=100.0,
        modified_duration=5.0,
        spread_duration=4.0,
        is_synthetic=True,
    )

    # Apply rates up shock: +50 bps (+0.0050)
    shocks_up = {"yield_10y": 0.0050, "credit_spread": 0.0}
    res_up = revalue_position(bond, shocks=shocks_up)

    # dP = -ModDur * dy = -5.0 * 0.0050 = -0.025 (-2.5%)
    # Expected loss = 10M * 0.025 = 250,000
    assert res_up.loss > 0.0
    assert res_up.post_value < res_up.pre_value
    assert res_up.loss == pytest.approx(250_000.0, rel=1e-3)
    assert res_up.post_value == pytest.approx(9_750_000.0, rel=1e-3)


def test_bond_convexity_reduces_loss_on_large_rate_hike():
    """Verify that 2nd-order bond convexity (+0.5 * C * dy^2) provides positive curvature benefit."""
    bond = Position(
        position_id="BOND_CONVEX_TEST",
        asset_class="Bond",
        instrument_type="Corporate Bond",
        counterparty="Apex Global Corp",
        sector="Industrials",
        country="United States",
        rating="BBB",
        notional=10_000_000.0,
        price=100.0,
        modified_duration=6.0,
        spread_duration=4.0,
        is_synthetic=True,
    )

    # Large rate shock: +200 bps (+0.0200)
    shocks_large = {"yield_10y": 0.0200, "credit_spread": 0.0}

    # Linear only
    res_linear = revalue_position(bond, shocks=shocks_large, include_convexity=False)
    # With 2nd order convexity
    res_convex = revalue_position(bond, shocks=shocks_large, include_convexity=True)

    # Convexity dampens bond price drop under large upward rate hikes
    assert res_convex.loss < res_linear.loss
    assert res_convex.post_value > res_linear.post_value
    assert "convexity" in res_convex.details
    assert res_convex.details["convexity_adjustment_bps"] > 0.0


def test_oil_up_hurts_the_right_sectors():
    """Verify that an oil price surge (+10% Brent) inflicts losses on energy-consuming sectors."""
    loan_industrials = Position(
        position_id="LOAN_IND",
        asset_class="Loan",
        instrument_type="Term Loan",
        counterparty="Midwest Heavy Machinery",
        sector="Industrials",  # Energy consumer
        country="United States",
        rating="BB",
        ead=20_000_000.0,
        pd=0.030,
        lgd=0.45,
        is_synthetic=True,
    )

    loan_unaffected = Position(
        position_id="LOAN_TECH",
        asset_class="Loan",
        instrument_type="Term Loan",
        counterparty="Cloud Hosting Inc",
        sector="Technology",  # Low direct energy consumption
        country="United States",
        rating="BB",
        ead=20_000_000.0,
        pd=0.030,
        lgd=0.45,
        is_synthetic=True,
    )

    # Oil spikes by +10%
    shocks_oil = {"brent_oil": 0.10, "yield_10y": 0.0, "credit_spread": 0.0}

    res_ind = revalue_position(loan_industrials, shocks=shocks_oil)
    res_tech = revalue_position(loan_unaffected, shocks=shocks_oil)

    # Industrials should suffer elevated stressed PD and positive loss
    assert res_ind.loss > 0.0
    assert res_ind.details["stressed_pd"] > res_ind.details["original_pd"]

    # Technology should suffer strictly less loss than Industrials
    assert res_ind.loss > res_tech.loss


def test_zero_shock_gives_zero_change():
    """Verify that applying zero factor shock results in exactly zero portfolio loss."""
    loan = Position(
        position_id="LOAN_ZERO",
        asset_class="Loan",
        instrument_type="Revolver",
        counterparty="Safe Borrower",
        sector="Financials",
        country="United States",
        rating="A",
        ead=15_000_000.0,
        pd=0.010,
        lgd=0.40,
        is_synthetic=True,
    )
    bond = Position(
        position_id="BOND_ZERO",
        asset_class="Bond",
        instrument_type="Senior Note",
        counterparty="Safe Issuer",
        sector="Technology",
        country="United States",
        rating="A",
        notional=10_000_000.0,
        price=100.0,
        modified_duration=4.5,
        spread_duration=3.5,
        is_synthetic=True,
    )
    deriv = Position(
        position_id="DERIV_ZERO",
        asset_class="Derivative",
        instrument_type="Interest Rate Swap",
        counterparty="Global Bank",
        sector="Financials",
        country="United Kingdom",
        rating="AA",
        notional=50_000_000.0,
        mtm=500_000.0,
        dv01=4500.0,
        derivative_type="Interest Rate Swap",
        is_synthetic=True,
    )

    port = Portfolio(positions=[loan, bond, deriv], name="Zero Shock Test Book")

    zero_shocks = {
        "yield_10y": 0.0,
        "credit_spread": 0.0,
        "brent_oil": 0.0,
        "fx_dxy": 0.0,
        "equity_us": 0.0,
    }

    # Revalue positions individually
    res_loan = revalue_position(loan, shocks=zero_shocks)
    res_bond = revalue_position(bond, shocks=zero_shocks)
    res_deriv = revalue_position(deriv, shocks=zero_shocks)

    assert res_loan.loss == 0.0
    assert res_loan.post_value == res_loan.pre_value

    assert res_bond.loss == 0.0
    assert res_bond.post_value == res_bond.pre_value

    assert res_deriv.loss == 0.0
    assert res_deriv.post_value == res_deriv.pre_value

    # Revalue entire portfolio
    port_res = revalue_portfolio(port, shocks=zero_shocks)
    assert port_res.total_loss == 0.0
    assert port_res.post_value == port_res.pre_value
    assert port_res.loss_pct == 0.0


# =====================================================================
# 2. Analog Matching & Prior Shrinkage Tests
# =====================================================================

def test_analog_matching_and_shrinkage():
    """Verify that FAISS retrieves analogs and applies Bayesian shrinkage when analogs < N."""
    event_query = {
        "title": "Severe liquidity crisis hits regional banking sector after commercial property losses",
        "event_type": "Credit Event",
    }

    # When min_analogs threshold is set high (e.g. 5), shrinkage should engage
    shocks = compute_factor_shocks(event_query, horizon_days=5, min_analogs=5)

    assert shocks.horizon_days == 5
    assert "yield_10y" in shocks.factors
    assert "credit_spread" in shocks.factors
    assert "p5" in shocks.factors["credit_spread"]
    assert "p50" in shocks.factors["credit_spread"]
    assert "p95" in shocks.factors["credit_spread"]

    # Check shrinkage behavior
    if len(shocks.analogs) < 5:
        assert shocks.shrunk_to_prior is True
        assert shocks.shrinkage_weight > 0.0
        assert "shrunk toward config prior table" in shocks.message
    else:
        assert shocks.shrunk_to_prior is False


# =====================================================================
# 3. End-to-End Stress Pipeline Tests
# =====================================================================

def test_stored_high_impact_signal_produces_stress_result(tmp_path):
    """Verify that a real stored high-impact signal from signals.jsonl executes full stress test."""
    signals_file = Path("data/signals/signals.jsonl")
    assert signals_file.exists(), "data/signals/signals.jsonl does not exist."

    # Find a signal with impact_score >= 7.0
    high_impact_signal = None
    with open(signals_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("impact_score", 0.0) >= 7.0:
                high_impact_signal = data
                break

    assert high_impact_signal is not None, "No signal with impact_score >= 7.0 found in signals.jsonl"

    # Execute stress engine
    result = stress(high_impact_signal, horizon_days=5, output_dir=tmp_path)

    # 1. Triggered check
    assert result.triggered is True
    assert "Stress test executed" in result.trigger_reason

    # 2. Before/after valuation and loss check
    assert result.portfolio_value_before > 1_000_000_000.0  # Multi-billion dollar book
    assert result.portfolio_value_after_p50 < result.portfolio_value_before
    assert result.loss_p50 > 0.0
    assert result.loss_pct_p50 > 0.0

    # 3. Confidence band check (p5 <= p95 loss)
    assert "p5_loss" in result.band
    assert "p95_loss" in result.band
    assert result.band["p5_loss"] <= result.band["p95_loss"]

    # 4. Dimensional breakdowns
    assert len(result.loss_by_asset_class) >= 2
    assert "Loan" in result.loss_by_asset_class
    assert len(result.loss_by_sector) > 0
    assert len(result.loss_by_country) > 0

    # 5. Top 10 positions
    assert len(result.top_10_positions) == 10
    assert result.top_10_positions[0]["loss"] >= result.top_10_positions[-1]["loss"]

    # 6. JSON artifact persistence
    assert result.saved_filepath is not None
    saved_file = Path(result.saved_filepath)
    assert saved_file.exists()

    with open(saved_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["run_id"] == result.run_id
    assert saved_data["portfolio_value_before"] == pytest.approx(result.portfolio_value_before, rel=1e-2)


def test_trigger_rule_skips_low_impact():
    """Verify that a signal below threshold 7.0 is skipped unless force=True."""
    low_signal = {
        "event_id": "low_test_001",
        "title": "Minor quarterly rebalancing announcement",
        "event_type": "Other",
        "impact_score": 4.2,
    }

    # Standard run: skipped
    res_skip = stress(low_signal, force=False, save_run=False)
    assert res_skip.triggered is False
    assert "below trigger threshold" in res_skip.trigger_reason
    assert res_skip.loss_p50 == 0.0

    # Forced run: executes
    res_force = stress(low_signal, force=True, save_run=False)
    assert res_force.triggered is True
    assert res_force.portfolio_value_before > 0.0
