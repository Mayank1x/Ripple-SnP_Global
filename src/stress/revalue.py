"""Position & Portfolio Revaluation Engine under Macroeconomic Shocks.

Implements exact financial valuation formulas for wholesale banking assets:
1. Corporate Loans:
   - stressed_pd = min(1.0, pd * exp(k * sector_intensity))
   - loss = ead * lgd * (stressed_pd - pd)
2. Bonds:
   - dP = - modified_duration * dy - spread_duration * dspread
   - loss = bond_mtm * (-dP)
3. Derivatives:
   - FX Forward: notional * dFX
   - Interest Rate Swap (IRS): dv01 * dRate_bp
   - Commodity Swap: notional * dPrice
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.portfolio.schema import Portfolio, Position

# Energy-consuming sectors that suffer margin compression when oil prices spike
OIL_CONSUMING_SECTORS: dict[str, float] = {
    "Industrials": 0.45,
    "Automotive": 0.50,
    "Consumer Discretionary": 0.35,
    "Materials": 0.40,
    "Utilities": 0.30,
}


@dataclass
class PositionStress:
    """Stress evaluation result for an individual position."""
    position_id: str
    counterparty: str
    asset_class: str
    instrument_type: str
    sector: str
    country: str
    rating: str
    pre_value: float
    post_value: float
    loss: float
    loss_pct: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioStressSummary:
    """Aggregated stress results across the entire wholesale banking book."""
    pre_value: float
    post_value: float
    total_loss: float
    loss_pct: float
    loss_by_asset_class: dict[str, float]
    loss_by_sector: dict[str, float]
    loss_by_country: dict[str, float]
    top_10_positions: list[dict[str, Any]]
    position_results: list[PositionStress]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_value": round(self.pre_value, 2),
            "post_value": round(self.post_value, 2),
            "total_loss": round(self.total_loss, 2),
            "loss_pct": round(self.loss_pct, 4),
            "loss_by_asset_class": {k: round(v, 2) for k, v in self.loss_by_asset_class.items()},
            "loss_by_sector": {k: round(v, 2) for k, v in self.loss_by_sector.items()},
            "loss_by_country": {k: round(v, 2) for k, v in self.loss_by_country.items()},
            "top_10_positions": self.top_10_positions,
        }


def revalue_position(
    pos: Position,
    shocks: dict[str, float],
    graph_intensities: dict[str, float] | None = None,
    k_loan: float = 2.0,
    include_convexity: bool = False,
) -> PositionStress:
    """Revalue a single banking position under factor shocks and network contagion.

    Args:
        pos: Position dataclass (Loan, Bond, or Derivative).
        shocks: Dictionary of macro factor moves (yield_10y, credit_spread, brent_oil, fx_dxy, equity_us).
        graph_intensities: Contagion intensity per node from ExposureGraph.propagate().
        k_loan: Exponential multiplier for loan PD sensitivity.
        include_convexity: If True, incorporates 2nd-order bond convexity (+0.5 * Convexity * dy^2).

    Returns:
        PositionStress result with pre_value, post_value, and loss.
    """
    intensities = graph_intensities or {}
    pre_val = pos.total_exposure
    loss = 0.0
    details: dict[str, Any] = {}

    # Extract macro factor changes
    dy = shocks.get("yield_10y", 0.0)          # Yield move in decimal (e.g. +0.0025 for +25 bps)
    dspread = shocks.get("credit_spread", 0.0) # Credit spread move in decimal (e.g. +0.0035 for +35 bps)
    dFX = shocks.get("fx_dxy", shocks.get("fx_usdinr", 0.0)) # FX move in decimal
    dPrice = shocks.get("brent_oil", shocks.get("commodity", 0.0)) # Commodity return in decimal

    # -------------------------------------------------------------
    # 1. Corporate Loans Book
    # -------------------------------------------------------------
    if pos.asset_class == "Loan":
        # Determine counterparty or sector contagion shock
        comp_shock = intensities.get(pos.counterparty, 0.0)
        sec_shock = intensities.get(pos.sector, 0.0)
        country_shock = intensities.get(pos.country, 0.0) * 0.50

        # Channel 1: Graph contagion
        sector_intensity = max(comp_shock, sec_shock, country_shock)

        # Channel 2: Macro credit spread widening directly lifts default intensity
        spread_shock_intensity = max(0.0, dspread * 100.0 * 0.15)

        # Channel 3: Commodity price surge stresses energy-consuming sectors
        oil_shock_intensity = 0.0
        if dPrice > 0 and pos.sector in OIL_CONSUMING_SECTORS:
            oil_shock_intensity = dPrice * OIL_CONSUMING_SECTORS[pos.sector]

        effective_intensity = max(sector_intensity, spread_shock_intensity, oil_shock_intensity)

        if effective_intensity > 0.0 and pos.pd > 0.0:
            # Formula: stressed PD = min(1.0, PD * exp(k * sector_intensity))
            stressed_pd = min(1.0, pos.pd * math.exp(k_loan * effective_intensity))
            # Formula: loss = EAD * LGD * (stressed PD - PD)
            loss = pos.ead * pos.lgd * (stressed_pd - pos.pd)
            loss = max(0.0, min(loss, pos.ead))
        else:
            stressed_pd = pos.pd
            loss = 0.0

        details = {
            "original_pd": pos.pd,
            "stressed_pd": round(stressed_pd, 6),
            "effective_intensity": round(effective_intensity, 4),
            "lgd": pos.lgd,
            "ead": pos.ead,
        }

    # -------------------------------------------------------------
    # 2. Bonds Book
    # -------------------------------------------------------------
    elif pos.asset_class == "Bond":
        # Formula: dP = -ModDur * dy - SpreadDur * dspread + (0.5 * Convexity * dy^2 if enabled)
        mod_dur = pos.modified_duration if pos.modified_duration > 0 else 4.0
        spread_dur = pos.spread_duration if pos.spread_duration > 0 else 3.5

        # Also incorporate issuer-specific contagion widening if present
        issuer_intensity = intensities.get(pos.counterparty, 0.0)
        effective_dspread = dspread + (issuer_intensity * 0.0050)

        # Convexity modeling: pos.metadata or empirical bond formula C ~= ModDur * (ModDur + 0.8) * 0.55
        meta = pos.metadata if (hasattr(pos, "metadata") and isinstance(pos.metadata, dict)) else {}
        convexity = float(meta.get("convexity", mod_dur * (mod_dur + 0.8) * 0.55))
        convexity_adj = 0.5 * convexity * (dy ** 2) if include_convexity else 0.0

        dP = - (mod_dur * dy) - (spread_dur * effective_dspread) + convexity_adj

        # Bond loss = MTM * (-dP)
        loss = pre_val * (-dP) if dP < 0 else 0.0
        loss = max(0.0, min(loss, pre_val))

        details = {
            "dy_bps": round(dy * 10_000, 2),
            "dspread_bps": round(effective_dspread * 10_000, 2),
            "mod_duration": mod_dur,
            "spread_duration": spread_dur,
            "convexity": round(convexity, 2),
            "convexity_adjustment_bps": round(convexity_adj * 10_000, 2),
            "dP_pct": round(dP * 100, 4),
        }

    # -------------------------------------------------------------
    # 3. Derivatives Book
    # -------------------------------------------------------------
    elif pos.asset_class == "Derivative":
        deriv_kind = (pos.derivative_type or pos.instrument_type or "").lower()

        if "fx" in deriv_kind or "forward" in deriv_kind:
            # Formula: FX forward loss = notional * dFX
            loss = pos.notional * abs(dFX) if dFX != 0.0 else 0.0
            details = {"type": "FX Forward", "dFX": dFX, "notional": pos.notional}

        elif "swap" in deriv_kind and ("rate" in deriv_kind or "irs" in deriv_kind):
            # Formula: IRS loss = DV01 * dRate(bp)
            dRate_bp = dy * 10_000.0  # Convert yield move to basis points
            dv01 = pos.dv01 if pos.dv01 > 0 else (pos.notional * 0.0001 * 0.5)
            # Upward rate shock generates DV01 sensitivity loss
            loss = dv01 * abs(dRate_bp) if dRate_bp != 0.0 else 0.0
            details = {"type": "IRS", "dv01": dv01, "dRate_bp": round(dRate_bp, 2)}

        elif "commodity" in deriv_kind or "oil" in deriv_kind:
            # Formula: Commodity swap loss = notional * dPrice
            loss = pos.notional * abs(dPrice) if dPrice != 0.0 else 0.0
            details = {"type": "Commodity Swap", "dPrice": dPrice, "notional": pos.notional}

        else:
            # General derivative sensitivity fallback
            loss = pre_val * max(abs(dy) * 5.0, abs(dspread) * 5.0, abs(dFX))
            details = {"type": "General Derivative", "loss_proxy": loss}

        loss = max(0.0, min(loss, pos.notional))

    post_val = max(0.0, pre_val - loss)
    loss_pct = (loss / pre_val) if pre_val > 0 else 0.0

    return PositionStress(
        position_id=pos.position_id,
        counterparty=pos.counterparty,
        asset_class=pos.asset_class,
        instrument_type=pos.instrument_type,
        sector=pos.sector,
        country=pos.country,
        rating=pos.rating,
        pre_value=round(pre_val, 2),
        post_value=round(post_val, 2),
        loss=round(loss, 2),
        loss_pct=round(loss_pct, 6),
        details=details,
    )


def revalue_portfolio(
    portfolio: Portfolio,
    shocks: dict[str, float],
    graph_intensities: dict[str, float] | None = None,
    k_loan: float = 2.0,
    include_convexity: bool = False,
) -> PortfolioStressSummary:
    """Revalue an entire wholesale portfolio under macroeconomic factor shocks and contagion."""
    results: list[PositionStress] = []
    loss_by_asset: dict[str, float] = {}
    loss_by_sector: dict[str, float] = {}
    loss_by_country: dict[str, float] = {}

    pre_total = 0.0
    post_total = 0.0
    total_loss = 0.0

    for pos in portfolio.positions:
        res = revalue_position(
            pos,
            shocks=shocks,
            graph_intensities=graph_intensities,
            k_loan=k_loan,
            include_convexity=include_convexity,
        )
        results.append(res)

        pre_total += res.pre_value
        post_total += res.post_value
        total_loss += res.loss

        # Breakdown aggregations
        ac = res.asset_class or "Other"
        sec = res.sector or "Other"
        ctry = res.country or "Other"

        loss_by_asset[ac] = loss_by_asset.get(ac, 0.0) + res.loss
        loss_by_sector[sec] = loss_by_sector.get(sec, 0.0) + res.loss
        loss_by_country[ctry] = loss_by_country.get(ctry, 0.0) + res.loss

    # Sort breakdowns descending
    sorted_sector = dict(sorted(loss_by_sector.items(), key=lambda x: -x[1]))
    sorted_country = dict(sorted(loss_by_country.items(), key=lambda x: -x[1]))
    sorted_asset = dict(sorted(loss_by_asset.items(), key=lambda x: -x[1]))

    # Top 10 positions by loss
    top_10 = sorted(results, key=lambda x: -x.loss)[:10]
    top_10_dicts = [p.to_dict() for p in top_10]

    loss_pct = (total_loss / pre_total) if pre_total > 0 else 0.0

    return PortfolioStressSummary(
        pre_value=pre_total,
        post_value=post_total,
        total_loss=total_loss,
        loss_pct=loss_pct,
        loss_by_asset_class=sorted_asset,
        loss_by_sector=sorted_sector,
        loss_by_country=sorted_country,
        top_10_positions=top_10_dicts,
        position_results=results,
    )


def compute_regulatory_capital(
    pre_val: float,
    total_loss: float,
    base_cet1_ratio: float = 0.1350,
) -> dict[str, Any]:
    """Calculate Basel III / IV Common Equity Tier 1 (CET1) capital impact under stress.

    Models:
    1. Baseline Risk-Weighted Assets (RWA) estimated at ~65% of wholesale banking book.
    2. Stressed Tier 1 Capital absorbed directly by mark-to-market and credit default losses.
    3. Rating migration expansion increasing stressed RWA by up to +3%.
    4. Compliance check against Basel III 10.50% minimum (4.5% min + 2.5% CCB + 3.5% D-SIB/G-SIB buffer).
    """
    base_rwa = pre_val * 0.65
    base_tier1 = base_rwa * base_cet1_ratio
    stressed_tier1 = max(0.0, base_tier1 - total_loss)

    # RWA density expansion from distressed borrower rating migrations
    migration_factor = 1.0 + min(0.08, (total_loss / pre_val) * 0.50 if pre_val > 0 else 0.0)
    stressed_rwa = base_rwa * migration_factor

    stressed_cet1 = (stressed_tier1 / stressed_rwa) if stressed_rwa > 0 else 0.0
    depletion_bps = (base_cet1_ratio - stressed_cet1) * 10_000.0
    min_requirement = 0.1050  # 10.50% Basel III minimum capital floor
    headroom_bps = (stressed_cet1 - min_requirement) * 10_000.0

    is_compliant = stressed_cet1 >= min_requirement
    status = "Adequate (Compliant)" if is_compliant else "Deficit (Recapitalization Alert)"

    return {
        "base_cet1": base_cet1_ratio,
        "stressed_cet1": round(stressed_cet1, 4),
        "depletion_bps": round(depletion_bps, 1),
        "min_requirement": min_requirement,
        "headroom_bps": round(headroom_bps, 1),
        "is_compliant": is_compliant,
        "status": status,
        "base_tier1": round(base_tier1, 2),
        "stressed_tier1": round(stressed_tier1, 2),
        "base_rwa": round(base_rwa, 2),
        "stressed_rwa": round(stressed_rwa, 2),
    }
