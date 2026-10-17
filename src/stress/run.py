"""Stress Engine Execution Pipeline (Module B).

Executes end-to-end stress tests on wholesale banking portfolios triggered by high-impact signals:
1. Evaluates trigger rule (impact_score >= threshold, default 7.0).
2. Retrieves analog crisis events and calculates factor shocks (p5, p50, p95).
3. Propagates network contagion through the Exposure Graph.
4. Computes portfolio revaluation before/after, confidence bands, dimensional losses, and top 10 positions.
5. Saves execution logs to data/stress_runs/.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingest.helpers import load_config
from src.portfolio.graph import ExposureGraph, build_exposure_graph
from src.portfolio.schema import Portfolio
from src.stress.revalue import revalue_portfolio
from src.stress.shocks import FactorShocks, compute_factor_shocks


@dataclass
class StressResult:
    """Complete output of a portfolio stress test execution."""
    run_id: str
    timestamp: str
    triggered: bool
    trigger_reason: str
    signal_summary: dict[str, Any]
    shocks: dict[str, Any]
    portfolio_value_before: float
    portfolio_value_after_p50: float
    loss_p50: float
    loss_pct_p50: float
    band: dict[str, float]  # p5 and p95 confidence band
    loss_by_asset_class: dict[str, float]
    loss_by_sector: dict[str, float]
    loss_by_country: dict[str, float]
    top_10_positions: list[dict[str, Any]]
    graph_shock_nodes: dict[str, float]
    saved_filepath: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "triggered": self.triggered,
            "trigger_reason": self.trigger_reason,
            "signal_summary": self.signal_summary,
            "shocks": self.shocks,
            "portfolio_value_before": round(self.portfolio_value_before, 2),
            "portfolio_value_after_p50": round(self.portfolio_value_after_p50, 2),
            "loss_p50": round(self.loss_p50, 2),
            "loss_pct_p50": round(self.loss_pct_p50, 4),
            "band": {k: round(v, 2) for k, v in self.band.items()},
            "loss_by_asset_class": {k: round(v, 2) for k, v in self.loss_by_asset_class.items()},
            "loss_by_sector": {k: round(v, 2) for k, v in self.loss_by_sector.items()},
            "loss_by_country": {k: round(v, 2) for k, v in self.loss_by_country.items()},
            "top_10_positions": self.top_10_positions,
            "graph_shock_nodes": {k: round(v, 4) for k, v in self.graph_shock_nodes.items()},
            "saved_filepath": self.saved_filepath,
        }


def _extract_seed_nodes(signal: dict[str, Any], graph: ExposureGraph) -> list[str]:
    """Extract valid graph seed nodes from signal entities."""
    entities = signal.get("entities", {})
    seeds: set[str] = set()

    # 1. Company entities
    for comp in entities.get("companies", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            if name in graph.g:
                seeds.add(name)
        elif isinstance(comp, str) and comp in graph.g:
            seeds.add(comp)

    # 2. Sector entities
    for sec in entities.get("sectors", []):
        if sec in graph.g:
            seeds.add(sec)

    # 3. Country entities
    for ctry in entities.get("countries", []):
        if ctry in graph.g:
            seeds.add(ctry)

    # 4. Commodities
    for comm in entities.get("commodities", []):
        if comm in graph.g:
            seeds.add(comm)

    # Fallback to key sectors or countries mentioned in title
    if not seeds:
        title = signal.get("title", "")
        if not title and "evidence" in signal and signal["evidence"]:
            title = signal["evidence"][0].get("title", "")

        for n in graph.g.nodes():
            if n.lower() in title.lower() and len(n) > 3:
                seeds.add(n)

    # Final fallback if no entities matched: seed default sector by event type
    if not seeds:
        etype = signal.get("event_type", "Other")
        if etype == "Credit Event" and "Financials" in graph.g:
            seeds.add("Financials")
        elif etype in ("Commodity/Supply Chain", "Geopolitical") and "Energy" in graph.g:
            seeds.add("Energy")
        elif "Technology" in graph.g:
            seeds.add("Technology")

    return list(seeds)


def stress(
    signal: dict[str, Any],
    portfolio: Portfolio | None = None,
    graph: ExposureGraph | None = None,
    horizon_days: int = 5,
    force: bool = False,
    save_run: bool = True,
    output_dir: Path | str | None = None,
) -> StressResult:
    """Execute complete stress test workflow on a wholesale portfolio for a given event signal.

    Args:
        signal: Event signal dictionary (from signals.jsonl or API).
        portfolio: Optional pre-loaded Portfolio instance (loads data/portfolio/portfolio.json if None).
        graph: Optional pre-built ExposureGraph instance (builds from portfolio if None).
        horizon_days: Forward shock horizon (1 or 5 days, default 5).
        force: If True, bypasses the impact_score threshold trigger rule.
        save_run: If True, saves JSON execution artifact to data/stress_runs/.
        output_dir: Custom destination directory for stress run JSON files.

    Returns:
        StressResult containing valuations, confidence bands, loss breakdowns, and top positions.
    """
    cfg = load_config()
    stress_cfg = cfg.get("stress", {})
    trigger_threshold = float(stress_cfg.get("trigger_impact_score", 7.0))
    k_loan = float(stress_cfg.get("loan_k_sensitivity", 2.0))

    if output_dir is None:
        output_dir = Path(stress_cfg.get("output_dir", "data/stress_runs"))
    else:
        output_dir = Path(output_dir)

    run_id = f"stress_{uuid.uuid4().hex[:10]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Extract signal metrics
    event_id = signal.get("event_id", "unknown_event")
    impact_score = float(signal.get("impact_score", 0.0))
    event_type = signal.get("event_type", "Other")
    title = signal.get("title", "")
    if not title and "evidence" in signal and signal["evidence"]:
        title = signal["evidence"][0].get("title", "")

    signal_summary = {
        "event_id": event_id,
        "title": title,
        "event_type": event_type,
        "impact_score": impact_score,
        "sentiment_score": signal.get("sentiment_score", 0.0),
        "confidence": signal.get("confidence", 0.0),
    }

    # 1. Evaluate Trigger Rule
    if impact_score < trigger_threshold and not force:
        return StressResult(
            run_id=run_id,
            timestamp=now_iso,
            triggered=False,
            trigger_reason=(
                f"Impact score {impact_score:.1f} is below trigger threshold {trigger_threshold:.1f}. "
                "Stress test skipped."
            ),
            signal_summary=signal_summary,
            shocks={},
            portfolio_value_before=0.0,
            portfolio_value_after_p50=0.0,
            loss_p50=0.0,
            loss_pct_p50=0.0,
            band={},
            loss_by_asset_class={},
            loss_by_sector={},
            loss_by_country={},
            top_10_positions=[],
            graph_shock_nodes={},
            saved_filepath=None,
        )

    # 2. Load Portfolio if needed
    if portfolio is None:
        port_path = Path("data/portfolio/portfolio.json")
        if not port_path.exists():
            raise FileNotFoundError(
                f"Portfolio file '{port_path}' not found. Run scripts/make_portfolio.py first."
            )
        portfolio = Portfolio.load(port_path)

    # 3. Build Exposure Graph if needed
    if graph is None:
        graph = build_exposure_graph(portfolio, seed=42)

    # 4. Propagate Contagion across Exposure Graph
    seeds = _extract_seed_nodes(signal, graph)
    shock_intensity = min(1.0, max(0.1, impact_score / 10.0))
    graph_intensities = graph.propagate(seed_nodes=seeds, shock=shock_intensity, hops=2, decay=0.5)

    # Keep active shock carriers
    active_shock_nodes = {n: v for n, v in graph_intensities.items() if v > 0.005}

    # 5. Compute Factor Shocks (p5, p50, p95) with prior shrinkage
    shocks_obj: FactorShocks = compute_factor_shocks(
        event_data=signal,
        event_type=event_type,
        horizon_days=horizon_days,
        top_k=5,
    )

    shocks_p50 = shocks_obj.get_shocks("p50")
    shocks_p5 = shocks_obj.get_shocks("p5")
    shocks_p95 = shocks_obj.get_shocks("p95")

    # 6. Revalue Portfolio across Median (p50), Mild (p5), and Severe (p95) bands
    res_p50 = revalue_portfolio(
        portfolio=portfolio,
        shocks=shocks_p50,
        graph_intensities=graph_intensities,
        k_loan=k_loan,
    )

    res_p5 = revalue_portfolio(
        portfolio=portfolio,
        shocks=shocks_p5,
        graph_intensities=graph_intensities,
        k_loan=k_loan,
    )

    res_p95 = revalue_portfolio(
        portfolio=portfolio,
        shocks=shocks_p95,
        graph_intensities=graph_intensities,
        k_loan=k_loan,
    )

    band = {
        "p5_loss": res_p5.total_loss,
        "p5_post_value": res_p5.post_value,
        "p5_loss_pct": res_p5.loss_pct,
        "p95_loss": res_p95.total_loss,
        "p95_post_value": res_p95.post_value,
        "p95_loss_pct": res_p95.loss_pct,
    }

    # 7. Package StressResult
    stress_result = StressResult(
        run_id=run_id,
        timestamp=now_iso,
        triggered=True,
        trigger_reason=f"Impact score {impact_score:.1f} >= threshold {trigger_threshold:.1f}. Stress test executed.",
        signal_summary=signal_summary,
        shocks=shocks_obj.to_dict(),
        portfolio_value_before=res_p50.pre_value,
        portfolio_value_after_p50=res_p50.post_value,
        loss_p50=res_p50.total_loss,
        loss_pct_p50=res_p50.loss_pct,
        band=band,
        loss_by_asset_class=res_p50.loss_by_asset_class,
        loss_by_sector=res_p50.loss_by_sector,
        loss_by_country=res_p50.loss_by_country,
        top_10_positions=res_p50.top_10_positions,
        graph_shock_nodes=active_shock_nodes,
        saved_filepath=None,
    )

    # 8. Save JSON record to data/stress_runs/
    if save_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"run_{event_id}_{run_id}.json"
        target_path = output_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(stress_result.to_dict(), f, indent=2)
        stress_result.saved_filepath = str(target_path)

    return stress_result
