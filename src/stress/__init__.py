"""Ripple Stress Engine (Module B).

Provides:
- Macro factor shock derivation via historical FAISS analogs and prior shrinkage
- Exact position and portfolio revaluation across Loans, Bonds, and Derivatives
- End-to-end stress execution pipeline with confidence banding (p5/p50/p95)
"""

from src.stress.revalue import (
    PortfolioStressSummary,
    PositionStress,
    revalue_portfolio,
    revalue_position,
)
from src.stress.run import StressResult, stress
from src.stress.shocks import (
    AnalogMatch,
    FactorShocks,
    compute_factor_shocks,
    find_analogs,
)

__all__ = [
    "AnalogMatch",
    "FactorShocks",
    "PortfolioStressSummary",
    "PositionStress",
    "StressResult",
    "compute_factor_shocks",
    "find_analogs",
    "revalue_portfolio",
    "revalue_position",
    "stress",
]
