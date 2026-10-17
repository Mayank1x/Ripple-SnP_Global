"""Generate seeded synthetic wholesale-bank portfolio (Loans, Bonds, Derivatives) and exposure graph.

Generates:
- ~70 corporate loans with counterparty, sector, country, rating, EAD, PD, LGD, maturity.
- ~60 bonds with issuer, notional, price, yield, modified duration, spread duration, rating.
- ~50 derivatives (IRS, FX forwards, Commodity swaps) with notional, MTM, DV01/delta.
- Saves to data/portfolio/ and builds docs/exposure_graph.png.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.portfolio.graph import build_exposure_graph
from src.portfolio.schema import Portfolio, Position

# Credit Rating Probability of Default (PD) lookup
RATING_PD_MAP: dict[str, float] = {
    "AAA": 0.0002,
    "AA": 0.0005,
    "A": 0.0010,
    "BBB": 0.0035,
    "BB": 0.0150,
    "B": 0.0450,
    "CCC": 0.1500,
}

# Rating distribution weights for wholesale banking book
RATING_WEIGHTS = {
    "AAA": 0.04,
    "AA": 0.10,
    "A": 0.26,
    "BBB": 0.36,
    "BB": 0.14,
    "B": 0.08,
    "CCC": 0.02,
}

SYNTHETIC_ADDITIONAL_COMPANIES = [
    ("Nordic Marine Logistics", "Industrials", "Norway", "BBB"),
    ("Rhine Industrial Chemicals", "Materials", "Germany", "A"),
    ("Pacific Renewable Power", "Utilities", "United States", "BBB"),
    ("Midwest Rail & Agri", "Industrials", "United States", "A"),
    ("Atlas Semiconductor Foundries", "Technology", "Taiwan", "AA"),
    ("Alpine Defense Technologies", "Industrials", "Switzerland", "A"),
    ("Bavarian Auto Components", "Automotive", "Germany", "BBB"),
    ("Tokyo Precision Robotics", "Technology", "Japan", "AA"),
    ("Caledonia Offshore Energy", "Energy", "United Kingdom", "BB"),
    ("Iberian Solar Infra", "Utilities", "Spain", "BBB"),
    ("Milan Luxury Apparel", "Consumer Discretionary", "Italy", "A"),
    ("Gothenburg Heavy Truck", "Automotive", "Sweden", "A"),
    ("Sydney Lithium Resources", "Materials", "Australia", "BBB"),
    ("Deccan Cloud Solutions", "Technology", "India", "BBB"),
    ("Singapore Port Terminals", "Industrials", "Singapore", "AA"),
]


def load_catalog_companies() -> list[tuple[str, str, str, str]]:
    """Load reference companies from data/reference/entities.csv."""
    ref_path = Path("data/reference/entities.csv")
    companies: list[tuple[str, str, str, str]] = []

    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("entity_type") == "company":
                    ticker = row.get("id", "")
                    name = row.get("name", "")
                    sec = row.get("sector", "Financials")
                    country = row.get("country", "United States")
                    companies.append((name, sec, country, ticker))

    # Add synthetic corporates
    for name, sec, country, rating in SYNTHETIC_ADDITIONAL_COMPANIES:
        companies.append((name, sec, country, None))

    return companies


def generate_synthetic_portfolio(seed: int = 42) -> Portfolio:
    """Generate wholesale bank portfolio with ~70 loans, ~60 bonds, and ~50 derivatives."""
    rng = random.Random(seed)
    companies = load_catalog_companies()

    portfolio = Portfolio(name="Apex Wholesale Bank Global Book")
    ratings = list(RATING_WEIGHTS.keys())
    weights = list(RATING_WEIGHTS.values())

    # ---------------------------------------------------------
    # 1. Corporate Loans (~70 positions)
    # ---------------------------------------------------------
    loan_types = ["Senior Secured Term Loan", "Revolving Credit Facility", "Syndicated Term Loan B", "Asset-Backed Facility"]
    for i in range(1, 71):
        comp = rng.choice(companies)
        name, sec, country, ticker = comp
        rating = rng.choices(ratings, weights=weights)[0]
        pd_val = RATING_PD_MAP[rating]

        # Loan EAD: $10M to $120M
        ead = round(rng.uniform(10_000_000, 120_000_000), 2)
        lgd = round(rng.uniform(0.35, 0.55), 2)
        expected_loss = round(ead * pd_val * lgd, 2)

        mat_year = rng.randint(2027, 2034)
        mat_month = rng.randint(1, 12)
        mat_day = rng.choice([15, 28, 30])
        maturity = f"{mat_year}-{mat_month:02d}-{mat_day:02d}"

        pos = Position(
            position_id=f"loan_{i:03d}",
            asset_class="Loan",
            instrument_type=rng.choice(loan_types),
            counterparty=name,
            sector=sec,
            country=country,
            rating=rating,
            maturity=maturity,
            ead=ead,
            pd=pd_val,
            lgd=lgd,
            expected_loss=expected_loss,
            ticker=ticker,
            is_synthetic=True,
            metadata={"facility_type": "Wholesale Corporate Lending"},
        )
        portfolio.add_position(pos)

    # ---------------------------------------------------------
    # 2. Corporate & Sovereign Bonds (~60 positions)
    # ---------------------------------------------------------
    bond_types = ["Senior Unsecured Bond", "Subordinated Tier 2 Bond", "Green Sustainability Bond", "High Yield Note"]
    for i in range(1, 61):
        comp = rng.choice(companies)
        name, sec, country, ticker = comp
        rating = rng.choices(ratings, weights=weights)[0]

        notional = round(rng.uniform(15_000_000, 100_000_000), 2)
        # Price: 94.00 to 104.50
        price = round(rng.uniform(94.0, 104.5), 2)
        ytm = round(rng.uniform(3.75, 8.25), 2)
        mod_dur = round(rng.uniform(2.5, 7.8), 2)
        spread_dur = round(max(1.5, mod_dur - rng.uniform(0.2, 0.8)), 2)
        mtm = round(notional * (price / 100.0), 2)

        mat_year = rng.randint(2027, 2035)
        maturity = f"{mat_year}-06-15"

        pos = Position(
            position_id=f"bond_{i:03d}",
            asset_class="Bond",
            instrument_type=rng.choice(bond_types),
            counterparty=name,
            sector=sec,
            country=country,
            rating=rating,
            maturity=maturity,
            notional=notional,
            price=price,
            yield_to_maturity=ytm,
            modified_duration=mod_dur,
            spread_duration=spread_dur,
            mtm=mtm,
            ticker=ticker,
            is_synthetic=True,
            metadata={"liquidity_tier": "Level 1/2 HQLA"},
        )
        portfolio.add_position(pos)

    # ---------------------------------------------------------
    # 3. Derivatives Book (~50 positions: IRS, FX, Commodity)
    # ---------------------------------------------------------
    deriv_configs = [
        ("Interest Rate Swap", ["Financials", "Industrials", "Technology"], "IRS"),
        ("FX Forward", ["Financials", "Consumer Discretionary", "Technology"], "FX"),
        ("Commodity Swap", ["Energy", "Materials", "Industrials"], "COMM"),
    ]

    for i in range(1, 51):
        inst_type, eligible_sectors, tag = rng.choice(deriv_configs)

        # Filter companies matching eligible sectors
        matching_comps = [c for c in companies if c[1] in eligible_sectors]
        comp = rng.choice(matching_comps) if matching_comps else rng.choice(companies)
        name, sec, country, ticker = comp

        rating = rng.choices(ratings, weights=weights)[0]
        notional = round(rng.uniform(25_000_000, 200_000_000), 2)

        # MTM can be positive or negative (-$6M to +$14M)
        mtm = round(rng.uniform(-6_000_000, 14_000_000), 2)

        if tag == "IRS":
            # DV01: $10,000 to $75,000 per bp
            dv01 = round(rng.uniform(10_000, 75_000), 2)
            delta = 0.0
        else:
            dv01 = 0.0
            # Delta hedge ratio (0.65 to 0.95)
            delta = round(rng.uniform(0.65, 0.95), 3)

        mat_year = rng.randint(2026, 2031)
        maturity = f"{mat_year}-12-31"

        pos = Position(
            position_id=f"deriv_{i:03d}",
            asset_class="Derivative",
            instrument_type=inst_type,
            counterparty=name,
            sector=sec,
            country=country,
            rating=rating,
            maturity=maturity,
            notional=notional,
            mtm=mtm,
            dv01=dv01,
            delta=delta,
            ticker=ticker,
            is_synthetic=True,
            metadata={"clearing": "Bilateral ISDA/CSA" if rng.random() > 0.4 else "Centrally Cleared"},
        )
        portfolio.add_position(pos)

    return portfolio


def main() -> None:
    print("=" * 70)
    print("RIPPLE PORTFOLIO ENGINE: Generating Synthetic Wholesale Banking Book")
    print("=" * 70)

    portfolio = generate_synthetic_portfolio(seed=42)
    output_dir = Path("data/portfolio")
    portfolio.save(output_dir)

    print(f"Saved synthetic portfolio files to: {output_dir.resolve()}")
    print("  - portfolio.json")
    print("  - portfolio.csv")
    print("  - loans.csv")
    print("  - bonds.csv")
    print("  - derivatives.csv")

    # Build exposure graph and save visualization
    print("\nBuilding interconnected exposure and contagion graph ...")
    eg = build_exposure_graph(portfolio, seed=42)
    graph_img = Path("docs/exposure_graph.png")
    eg.draw_graph(output_path=graph_img)

    # Print plausible summary table
    summary = portfolio.summary()
    print("\n" + "-" * 70)
    print(f"                   PORTFOLIO SUMMARY: {portfolio.name.upper()}")
    print("-" * 70)
    print(f"Total Positions               : {summary['total_positions']}")
    print(f"  * Corporate Loans           : {summary['book_counts']['loans']}")
    print(f"  * Bonds Book                : {summary['book_counts']['bonds']}")
    print(f"  * Derivatives Book          : {summary['book_counts']['derivatives']}")
    print("-" * 70)
    print(f"Total Economic Exposure       : ${summary['total_economic_exposure'] / 1e9:.2f} Billion")
    print(f"  * Total Loan EAD            : ${summary['total_loans_ead'] / 1e9:.2f} Billion")
    print(f"  * Total Bond MTM Value      : ${summary['total_bonds_mtm'] / 1e9:.2f} Billion")
    print(f"  * Derivatives Gross Notional: ${summary['total_derivatives_notional'] / 1e9:.2f} Billion")
    print(f"  * Total Expected Loss (EL)  : ${summary['total_expected_loss'] / 1e6:.2f} Million")
    print("-" * 70)

    print("Top Sector Exposures:")
    for sec, exp in summary["top_sectors"].items():
        pct = (exp / summary["total_economic_exposure"]) * 100
        print(f"  - {sec:<24}: ${exp / 1e6:>8.1f}M ({pct:>5.1f}%)")

    print("\nTop Country Exposures:")
    for ctry, exp in summary["top_countries"].items():
        pct = (exp / summary["total_economic_exposure"]) * 100
        print(f"  - {ctry:<24}: ${exp / 1e6:>8.1f}M ({pct:>5.1f}%)")

    print("\nRating Breakdown:")
    for r, exp in summary["rating_breakdown"].items():
        pct = (exp / summary["total_economic_exposure"]) * 100
        print(f"  - {r:<5}: ${exp / 1e6:>8.1f}M ({pct:>5.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
