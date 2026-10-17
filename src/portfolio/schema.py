"""Portfolio and Position data structures for wholesale banking books."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Position:
    """Individual wholesale banking position (Loan, Bond, or Derivative)."""

    position_id: str
    asset_class: str  # "Loan", "Bond", "Derivative"
    instrument_type: str = "Corporate Exposure"  # "Term Loan", "Revolver", "Corporate Bond", "Interest Rate Swap", "FX Forward", etc.
    counterparty: str = ""
    sector: str = ""
    country: str = ""
    rating: str = "BBB"  # AAA, AA, A, BBB, BB, B, CCC
    maturity: str = ""  # Date string, e.g. "2028-06-30"

    # Exposure & Valuation fields
    ead: float = 0.0  # Exposure at Default ($) for Loans
    pd: float = 0.0  # Annual Probability of Default (0.0002 to 0.15)
    lgd: float = 0.45  # Loss Given Default (0.35 to 0.65)
    expected_loss: float = 0.0  # EAD * PD * LGD ($)

    notional: float = 0.0  # Par / Notional amount ($)
    price: float = 100.0  # Clean price for Bonds ($ per 100 par)
    yield_to_maturity: float = 0.0  # YTM for Bonds (%)
    modified_duration: float = 0.0  # Mod duration in years
    spread_duration: float = 0.0  # Spread duration in years

    mtm: float = 0.0  # Mark-to-market value ($)
    dv01: float = 0.0  # Dollar Value of a Basis Point ($/bp)
    delta: float = 0.0  # Sensitivity / Delta ratio
    derivative_type: str = ""  # FX Forward, Interest Rate Swap, Commodity Swap

    ticker: str | None = None
    is_synthetic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asset_class == "Loan" and self.expected_loss == 0.0 and self.ead > 0:
            self.expected_loss = round(self.ead * self.pd * self.lgd, 2)
        if self.asset_class == "Bond" and self.mtm == 0.0 and self.notional > 0:
            self.mtm = round(self.notional * (self.price / 100.0), 2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        return cls(**data)

    @property
    def bond_mtm(self) -> float:
        """Bond Mark-to-Market value ($)."""
        return self.mtm if self.mtm > 0 else (self.notional * (self.price / 100.0))

    @property
    def total_exposure(self) -> float:
        """Calculate primary risk exposure for aggregation."""
        if self.asset_class == "Loan":
            return self.ead
        elif self.asset_class == "Bond":
            return self.bond_mtm
        elif self.asset_class == "Derivative":
            return abs(self.mtm) if self.mtm != 0 else (self.notional * 0.05)
        return self.notional


class Portfolio:
    """Wholesale Bank Portfolio containing Loans, Bonds, and Derivatives."""

    def __init__(self, positions: list[Position] | None = None, name: str = "Wholesale Banking Book") -> None:
        self.name = name
        self.positions: list[Position] = positions or []

    def add_position(self, pos: Position) -> None:
        self.positions.append(pos)

    @property
    def is_synthetic(self) -> bool:
        """Return True if portfolio positions are synthetic."""
        return all(p.is_synthetic for p in self.positions) if self.positions else True

    def value(self) -> float:
        """Total aggregate economic exposure of the portfolio."""
        return sum(p.total_exposure for p in self.positions)

    def total_loans_ead(self) -> float:
        """Total Exposure at Default for Corporate Loans book."""
        return sum(p.ead for p in self.positions if p.asset_class == "Loan")

    def total_bonds_mtm(self) -> float:
        """Total Mark-to-Market value of the Bond book."""
        return sum(p.bond_mtm for p in self.positions if p.asset_class == "Bond")

    def total_derivatives_notional(self) -> float:
        """Total Gross Notional of Derivatives book."""
        return sum(p.notional for p in self.positions if p.asset_class == "Derivative")

    def total_expected_loss(self) -> float:
        """Total Expected Loss across the loan book."""
        return sum(p.expected_loss for p in self.positions if p.asset_class == "Loan")

    def exposure_by(self, dimension: str) -> dict[str, float]:
        """Aggregate total exposure by a given attribute dimension.

        Args:
            dimension: One of 'sector', 'country', 'asset_class', 'counterparty', 'rating'.

        Returns:
            Dictionary mapping category to total exposure amount ($), sorted descending.
        """
        valid_dims = {"sector", "country", "asset_class", "counterparty", "rating"}
        dim = dimension.lower()
        if dim not in valid_dims:
            raise ValueError(f"Unknown exposure dimension '{dimension}'. Valid options: {valid_dims}")

        totals: dict[str, float] = {}
        for p in self.positions:
            val = getattr(p, dim, "Other") or "Other"
            totals[val] = totals.get(val, 0.0) + p.total_exposure

        # Sort descending by exposure
        return dict(sorted(totals.items(), key=lambda item: -item[1]))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all positions to a pandas DataFrame."""
        return pd.DataFrame([p.to_dict() for p in self.positions])

    def summary(self) -> dict[str, Any]:
        """Generate high-level risk and breakdown summary."""
        n_pos = len(self.positions)
        n_loans = sum(1 for p in self.positions if p.asset_class == "Loan")
        n_bonds = sum(1 for p in self.positions if p.asset_class == "Bond")
        n_derivs = sum(1 for p in self.positions if p.asset_class == "Derivative")

        tot_val = self.value()
        loans_ead = self.total_loans_ead()
        bonds_mtm = self.total_bonds_mtm()
        deriv_notional = self.total_derivatives_notional()
        exp_loss = self.total_expected_loss()

        return {
            "portfolio_name": self.name,
            "total_positions": n_pos,
            "book_counts": {"loans": n_loans, "bonds": n_bonds, "derivatives": n_derivs},
            "total_economic_exposure": round(tot_val, 2),
            "total_loans_ead": round(loans_ead, 2),
            "total_bonds_mtm": round(bonds_mtm, 2),
            "total_derivatives_notional": round(deriv_notional, 2),
            "total_expected_loss": round(exp_loss, 2),
            "top_sectors": dict(list(self.exposure_by("sector").items())[:5]),
            "top_countries": dict(list(self.exposure_by("country").items())[:5]),
            "rating_breakdown": self.exposure_by("rating"),
        }

    def save(self, directory: Path | str = "data/portfolio") -> None:
        """Save portfolio positions to JSON and CSV files."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "source": "synthetic_generator",
            "count": len(self.positions),
            "positions": [p.to_dict() for p in self.positions],
        }

        # 1. Save full JSON
        with open(dir_path / "portfolio.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        # 2. Save full CSV
        df = self.to_dataframe()
        df.to_csv(dir_path / "portfolio.csv", index=False)

        # 3. Save book-specific CSVs
        df[df["asset_class"] == "Loan"].to_csv(dir_path / "loans.csv", index=False)
        df[df["asset_class"] == "Bond"].to_csv(dir_path / "bonds.csv", index=False)
        df[df["asset_class"] == "Derivative"].to_csv(dir_path / "derivatives.csv", index=False)

    @classmethod
    def load(cls, file_or_dir: Path | str = "data/portfolio") -> Portfolio:
        """Load portfolio from portfolio.json or directory."""
        path = Path(file_or_dir)
        target_file = path if path.is_file() else (path / "portfolio.json")

        if not target_file.exists():
            raise FileNotFoundError(f"Portfolio file not found at {target_file}")

        with open(target_file, encoding="utf-8") as fh:
            data = json.load(fh)

        name = data.get("name", "Wholesale Banking Book")
        raw_positions = data.get("positions", [])
        positions = [Position.from_dict(p) for p in raw_positions]
        return cls(positions=positions, name=name)
