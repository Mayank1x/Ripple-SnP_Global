"""Financial Exposure & Contagion Graph using NetworkX.

Nodes: Company, Sector, Country, Commodity
Edges:
  - member_of: Company <-> Sector
  - located_in: Company <-> Country
  - sensitive_to: Sector <-> Commodity
  - supplies: Company A -> Company B (B2B supply chain)

Contagion:
  propagate(seed_nodes, initial_shock=1.0, hops=2, decay=0.5)
  Simulates multi-hop financial risk contagion across systemic channels.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.portfolio.schema import Portfolio


class ExposureGraph:
    """NetworkX-based graph representing wholesale corporate interconnectedness."""

    def __init__(self) -> None:
        self.g = nx.Graph()

    def add_company(self, name: str, sector: str, country: str, rating: str = "BBB") -> None:
        """Add a company node and its sector/country relationship edges."""
        self.g.add_node(name, node_type="company", label=name, rating=rating, sector=sector, country=country)

        # Sector node & edge
        self.g.add_node(sector, node_type="sector", label=sector)
        self.g.add_edge(name, sector, relationship="member_of", weight=1.0)

        # Country node & edge
        self.g.add_node(country, node_type="country", label=country)
        self.g.add_edge(name, country, relationship="located_in", weight=0.85)

    def add_commodity_sensitivity(self, sector: str, commodity: str, weight: float = 0.75) -> None:
        """Link a sector to an underlying commodity price exposure."""
        self.g.add_node(sector, node_type="sector", label=sector)
        self.g.add_node(commodity, node_type="commodity", label=commodity)
        self.g.add_edge(sector, commodity, relationship="sensitive_to", weight=weight)

    def add_supply_relationship(self, supplier: str, customer: str, weight: float = 0.65) -> None:
        """Add a B2B supply chain dependency between two companies."""
        if supplier in self.g and customer in self.g:
            self.g.add_edge(supplier, customer, relationship="supplies", weight=weight)

    def propagate(
        self,
        seed_nodes: list[str] | dict[str, float] | str,
        shock: float = 1.0,
        hops: int = 2,
        decay: float = 0.5,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Propagate shock intensity from seed nodes across graph edges with distance decay.

        Args:
            seed_nodes: List of seed node names, dict of {node_name: initial_intensity}, or single node string.
            shock: Default shock intensity (0.0 to 1.0).
            hops: Max number of edge traversals (default 2).
            decay: Multiplier applied at each hop (default 0.5).

        Returns:
            Dictionary of {node_name: shock_intensity} for all affected nodes.
        """
        # Support initial_shock alias if passed
        effective_shock = kwargs.get("initial_shock", shock)

        # Handle single string passed as seed_nodes
        if isinstance(seed_nodes, str):
            seed_nodes = [seed_nodes]

        # Initialize seed shocks
        intensities: dict[str, float] = {}
        if isinstance(seed_nodes, dict):
            for node, s in seed_nodes.items():
                if node in self.g:
                    intensities[node] = float(min(1.0, max(0.0, s)))
        elif isinstance(seed_nodes, (list, tuple, set)):
            for node in seed_nodes:
                if node in self.g:
                    intensities[node] = float(min(1.0, max(0.0, effective_shock)))

        if not intensities:
            return {}

        current_frontier = dict(intensities)

        for _ in range(hops):
            next_frontier: dict[str, float] = {}
            for u, u_shock in current_frontier.items():
                for v in self.g.neighbors(u):
                    edge_w = float(self.g[u][v].get("weight", 1.0))
                    transmitted = u_shock * edge_w * decay

                    # Highest transmitted shock along any path wins
                    if transmitted > next_frontier.get(v, 0.0):
                        next_frontier[v] = transmitted

            # Accumulate into global intensity map
            for v, v_shock in next_frontier.items():
                if v_shock > intensities.get(v, 0.0):
                    intensities[v] = v_shock

            current_frontier = next_frontier

        # Format and round
        return {node: round(val, 4) for node, val in intensities.items() if val > 1e-4}

    def draw_graph(self, output_path: Path | str = "docs/exposure_graph.png", max_nodes: int = 75) -> None:
        """Render high-resolution visualization of the exposure graph and save to docs/."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Select a subgraph if full graph is very large
        if len(self.g) > max_nodes:
            # Pick highest degree nodes
            top_nodes = [n for n, d in sorted(self.g.degree(), key=lambda x: -x[1])[:max_nodes]]
            sub_g = self.g.subgraph(top_nodes)
        else:
            sub_g = self.g

        plt.figure(figsize=(14, 10), dpi=300)
        pos = nx.spring_layout(sub_g, seed=42, k=0.45, iterations=60)

        # Node color mapping by node_type
        type_colors = {
            "company": "#3b82f6",    # Blue
            "sector": "#10b981",     # Green
            "country": "#f59e0b",    # Amber
            "commodity": "#ef4444",  # Red
        }

        node_colors = [
            type_colors.get(sub_g.nodes[n].get("node_type", "company"), "#94a3b8")
            for n in sub_g.nodes()
        ]

        node_sizes = [
            600 + sub_g.degree(n) * 120
            for n in sub_g.nodes()
        ]

        # Draw edges by relationship type
        nx.draw_networkx_edges(sub_g, pos, alpha=0.3, edge_color="#64748b", width=1.2)
        nx.draw_networkx_nodes(sub_g, pos, node_color=node_colors, node_size=node_sizes, alpha=0.92, edgecolors="#ffffff", linewidths=1.5)

        # Draw labels
        labels = {n: str(n) for n in sub_g.nodes()}
        nx.draw_networkx_labels(sub_g, pos, labels=labels, font_size=8, font_weight="bold", font_family="sans-serif")

        plt.title("Ripple Risk Engine — Wholesale Exposure & Contagion Graph", fontsize=15, fontweight="bold", pad=20)
        plt.axis("off")

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Company', markerfacecolor='#3b82f6', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Sector', markerfacecolor='#10b981', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Country', markerfacecolor='#f59e0b', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Commodity', markerfacecolor='#ef4444', markersize=12),
        ]
        plt.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"  [graph] Saved exposure graph picture to: {path}")


def build_exposure_graph(portfolio: Portfolio, seed: int = 42) -> ExposureGraph:
    """Build interconnected ExposureGraph from portfolio positions and commodity sensitivities."""
    rng = random.Random(seed)
    eg = ExposureGraph()

    # 1. Add all counterparties, sectors, and countries from portfolio
    added_companies: set[str] = set()
    for pos in portfolio.positions:
        comp = pos.counterparty
        if comp and comp not in added_companies:
            eg.add_company(comp, sector=pos.sector, country=pos.country, rating=pos.rating)
            added_companies.add(comp)

    # 2. Add structural commodity sensitivities
    sensitivities = [
        ("Energy", "Crude Oil", 0.90),
        ("Energy", "Brent", 0.92),
        ("Energy", "Natural Gas", 0.85),
        ("Technology", "Semiconductors", 0.88),
        ("Materials", "Copper", 0.80),
        ("Automotive", "Semiconductors", 0.70),
        ("Industrials", "Crude Oil", 0.60),
        ("Utilities", "Natural Gas", 0.75),
    ]
    for sec, comm, w in sensitivities:
        eg.add_commodity_sensitivity(sec, comm, weight=w)

    # 3. Add seeded B2B supply chain links between companies
    companies_by_sector: dict[str, list[str]] = {}
    for comp in added_companies:
        sec = eg.g.nodes[comp].get("sector", "")
        companies_by_sector.setdefault(sec, []).append(comp)

    # Cross-sector supply relationships (e.g. Tech -> Autos, Energy -> Industrials)
    tech_firms = companies_by_sector.get("Technology", [])
    auto_firms = companies_by_sector.get("Automotive", [])
    energy_firms = companies_by_sector.get("Energy", [])
    ind_firms = companies_by_sector.get("Industrials", [])

    for t in tech_firms:
        for a in auto_firms:
            if rng.random() > 0.45:
                eg.add_supply_relationship(t, a, weight=round(rng.uniform(0.5, 0.8), 2))

    for e in energy_firms:
        for i in ind_firms:
            if rng.random() > 0.50:
                eg.add_supply_relationship(e, i, weight=round(rng.uniform(0.5, 0.8), 2))

    return eg
