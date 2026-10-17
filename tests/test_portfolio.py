"""Tests for Phase 5 - Synthetic Portfolio, Exposure Graph, and Contagion Propagation."""

from pathlib import Path

import pytest

from src.portfolio.graph import ExposureGraph, build_exposure_graph
from src.portfolio.schema import Portfolio, Position

# =====================================================================
# 1. Toy Graph Propagation Tests (Crucial User Requirement)
# =====================================================================

def test_toy_graph_propagation():
    """Verify exact mathematical decay across a simple 3-node chain: A -> B -> C."""
    eg = ExposureGraph()
    # Add nodes and edges manually with specific weights
    eg.g.add_node("A", node_type="company")
    eg.g.add_node("B", node_type="company")
    eg.g.add_node("C", node_type="company")
    eg.g.add_edge("A", "B", weight=0.8, relationship="supplies")
    eg.g.add_edge("B", "C", weight=0.5, relationship="supplies")

    # Initial shock = 1.0 at node A, decay = 0.5, hops = 2
    # Hop 0: A = 1.0
    # Hop 1: B = 1.0 * 0.8 * 0.5 = 0.40
    # Hop 2: C = 0.40 * 0.5 * 0.5 = 0.10
    result = eg.propagate(seed_nodes=["A"], shock=1.0, hops=2, decay=0.5)

    assert result["A"] == pytest.approx(1.0, rel=1e-3)
    assert result["B"] == pytest.approx(0.40, rel=1e-3)
    assert result["C"] == pytest.approx(0.10, rel=1e-3)


def test_toy_graph_single_hop():
    """Verify that hops=1 halts propagation before reaching node C."""
    eg = ExposureGraph()
    eg.g.add_edge("A", "B", weight=0.8)
    eg.g.add_edge("B", "C", weight=0.5)

    result = eg.propagate(seed_nodes="A", shock=1.0, hops=1, decay=0.5)

    assert "A" in result
    assert "B" in result
    assert result["B"] == pytest.approx(0.40, rel=1e-3)
    assert "C" not in result


def test_toy_graph_dict_seed():
    """Verify seed nodes passed as dict with custom initial shock."""
    eg = ExposureGraph()
    eg.g.add_edge("X", "Y", weight=0.6)

    # Initial shock at X = 0.5, decay = 0.5 -> Y = 0.5 * 0.6 * 0.5 = 0.15
    result = eg.propagate(seed_nodes={"X": 0.5}, hops=1, decay=0.5)

    assert result["X"] == pytest.approx(0.5, rel=1e-3)
    assert result["Y"] == pytest.approx(0.15, rel=1e-3)


def test_toy_graph_empty_seed():
    """Verify empty or nonexistent seeds return empty dict safely."""
    eg = ExposureGraph()
    eg.g.add_edge("A", "B", weight=1.0)

    assert eg.propagate(seed_nodes=[]) == {}
    assert eg.propagate(seed_nodes=["NonExistentNode"]) == {}


# =====================================================================
# 2. Portfolio Schema & Valuation Tests
# =====================================================================

@pytest.fixture
def sample_positions():
    """Construct a clean 3-asset test portfolio."""
    p1 = Position(
        position_id="LOAN_001",
        asset_class="Loan",
        counterparty="Titan Corp",
        sector="Technology",
        country="United States",
        rating="BBB",
        ead=10_000_000.0,
        pd=0.015,
        lgd=0.45,
        maturity="2028-06-30",
        is_synthetic=True,
    )
    p2 = Position(
        position_id="BOND_001",
        asset_class="Bond",
        counterparty="Titan Corp",
        sector="Technology",
        country="United States",
        rating="BBB",
        notional=5_000_000.0,
        price=98.5,
        yield_to_maturity=0.055,
        modified_duration=4.2,
        spread_duration=4.0,
        is_synthetic=True,
    )
    p3 = Position(
        position_id="DERIV_001",
        asset_class="Derivative",
        counterparty="Global Bank",
        sector="Financials",
        country="United Kingdom",
        rating="A",
        notional=20_000_000.0,
        mtm=450_000.0,
        dv01=3500.0,
        derivative_type="Interest Rate Swap",
        is_synthetic=True,
    )
    return [p1, p2, p3]


def test_position_properties(sample_positions):
    loan, bond, deriv = sample_positions

    # Loan expected loss = EAD * PD * LGD = 10M * 0.015 * 0.45 = 67,500
    assert loan.expected_loss == pytest.approx(67_500.0)
    assert loan.total_exposure == 10_000_000.0

    # Bond MTM = 5M * (98.5 / 100) = 4,925,000
    assert bond.bond_mtm == pytest.approx(4_925_000.0)
    assert bond.total_exposure == bond.bond_mtm

    # Derivative total exposure is abs(mtm) = 450K, gross notional = 20M
    assert deriv.total_exposure == 450_000.0
    assert deriv.notional == 20_000_000.0
    assert deriv.is_synthetic is True


def test_portfolio_aggregations(sample_positions):
    port = Portfolio(name="Test Book", positions=sample_positions)

    # Value = Loan EAD (10M) + Bond MTM (4.925M) + Deriv abs(MTM) (450K) = 15,375,000
    assert port.value() == pytest.approx(15_375_000.0)
    assert port.total_loans_ead() == pytest.approx(10_000_000.0)
    assert port.total_bonds_mtm() == pytest.approx(4_925_000.0)
    assert port.total_derivatives_notional() == pytest.approx(20_000_000.0)
    assert port.total_expected_loss() == pytest.approx(67_500.0)


def test_portfolio_exposure_by(sample_positions):
    port = Portfolio(name="Test Book", positions=sample_positions)

    by_sector = port.exposure_by("sector")
    assert "Technology" in by_sector
    assert "Financials" in by_sector
    # Tech exposure = 10M + 4.925M = 14.925M
    assert by_sector["Technology"] == pytest.approx(14_925_000.0)

    by_country = port.exposure_by("country")
    assert "United States" in by_country
    assert "United Kingdom" in by_country

    by_asset = port.exposure_by("asset_class")
    assert "Loan" in by_asset
    assert "Bond" in by_asset
    assert "Derivative" in by_asset

    by_rating = port.exposure_by("rating")
    assert "BBB" in by_rating
    assert "A" in by_rating

    by_counterparty = port.exposure_by("counterparty")
    assert "Titan Corp" in by_counterparty
    assert "Global Bank" in by_counterparty

    with pytest.raises(ValueError, match="Unknown exposure dimension"):
        port.exposure_by("invalid_dimension")


def test_portfolio_dataframe_and_summary(sample_positions):
    port = Portfolio(name="Test Book", positions=sample_positions)
    df = port.to_dataframe()
    assert len(df) == 3
    assert "position_id" in df.columns
    assert "is_synthetic" in df.columns

    summary = port.summary()
    assert summary["total_positions"] == 3
    assert summary["book_counts"]["loans"] == 1
    assert summary["book_counts"]["bonds"] == 1
    assert summary["book_counts"]["derivatives"] == 1


# =====================================================================
# 3. Real Synthetic Portfolio File Tests
# =====================================================================

def test_saved_portfolio_files():
    """Verify that data/portfolio/ contains the expected generated datasets."""
    port_json = Path("data/portfolio/portfolio.json")
    assert port_json.exists(), "data/portfolio/portfolio.json does not exist. Run scripts/make_portfolio.py first."

    port = Portfolio.load(port_json)
    assert len(port.positions) >= 150
    assert port.is_synthetic is True

    # Count by asset class
    counts = port.summary()["book_counts"]
    assert counts["loans"] >= 50
    assert counts["bonds"] >= 50
    assert counts["derivatives"] >= 40

    # Plausible economic totals
    assert port.value() > 1_000_000_000  # Multi-billion dollar book
    assert port.total_expected_loss() > 0


# =====================================================================
# 4. Exposure Graph & Real Contagion Network Tests
# =====================================================================

def test_built_exposure_graph_structure():
    port_json = Path("data/portfolio/portfolio.json")
    port = Portfolio.load(port_json)
    eg = build_exposure_graph(port, seed=42)

    # Check node types
    node_types = {data.get("node_type") for _, data in eg.g.nodes(data=True)}
    assert "company" in node_types
    assert "sector" in node_types
    assert "country" in node_types
    assert "commodity" in node_types

    # Check edge relationships
    edge_rels = {data.get("relationship") for _, _, data in eg.g.edges(data=True)}
    assert "member_of" in edge_rels
    assert "located_in" in edge_rels
    assert "sensitive_to" in edge_rels
    assert "supplies" in edge_rels


def test_exposure_graph_contagion_propagation():
    port_json = Path("data/portfolio/portfolio.json")
    port = Portfolio.load(port_json)
    eg = build_exposure_graph(port, seed=42)

    # Test shock from an energy commodity
    shock_results = eg.propagate(seed_nodes="Crude Oil", shock=1.0, hops=2, decay=0.5)

    assert "Crude Oil" in shock_results
    assert shock_results["Crude Oil"] == 1.0

    # Sector connected to Crude Oil should feel hop 1 shock
    assert "Energy" in shock_results
    assert 0.0 < shock_results["Energy"] <= 0.50

    # Energy companies should feel hop 2 shock
    companies_hit = [
        n for n in shock_results
        if eg.g.nodes[n].get("node_type") == "company"
    ]
    assert len(companies_hit) > 0
    for comp in companies_hit:
        assert shock_results[comp] < shock_results["Energy"]
