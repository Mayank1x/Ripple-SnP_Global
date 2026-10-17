"""Analog Event Matching & Factor Shock Distribution Engine (Module B).

Finds historical analog events using FAISS cosine similarity over sentence embeddings
filtered by event_type. Computes empirical 1-day and 5-day forward factor moves.
If fewer than N analogs are found, shrinks toward the config prior table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.ingest.helpers import load_config
from src.scoring.cluster import embed_texts

# Standard historical factor shock database
HISTORICAL_FACTOR_MOVES: dict[str, dict[str, dict[str, float]]] = {
    # event_id -> {"1d": {...}, "5d": {...}}
    "hist_001": {  # SVB Bank Run (Credit Event)
        "1d": {"yield_10y": -0.0018, "credit_spread": 0.0035, "brent_oil": -0.012, "fx_dxy": 0.006, "equity_us": -0.014},
        "5d": {"yield_10y": -0.0032, "credit_spread": 0.0065, "brent_oil": -0.028, "fx_dxy": 0.011, "equity_us": -0.032},
    },
    "hist_002": {  # Signature Bank Regulators (Regulatory / Credit)
        "1d": {"yield_10y": -0.0012, "credit_spread": 0.0028, "brent_oil": -0.008, "fx_dxy": 0.005, "equity_us": -0.011},
        "5d": {"yield_10y": -0.0024, "credit_spread": 0.0052, "brent_oil": -0.019, "fx_dxy": 0.009, "equity_us": -0.026},
    },
    "hist_003": {  # UBS Takeover Credit Suisse (Merger / Credit)
        "1d": {"yield_10y": -0.0008, "credit_spread": 0.0022, "brent_oil": -0.005, "fx_dxy": 0.003, "equity_us": -0.007},
        "5d": {"yield_10y": -0.0015, "credit_spread": 0.0040, "brent_oil": -0.012, "fx_dxy": 0.006, "equity_us": -0.018},
    },
    "hist_004": {  # Fed Holds Rates High (Macroeconomic)
        "1d": {"yield_10y": 0.0010, "credit_spread": 0.0012, "brent_oil": 0.004, "fx_dxy": 0.006, "equity_us": -0.008},
        "5d": {"yield_10y": 0.0022, "credit_spread": 0.0024, "brent_oil": 0.011, "fx_dxy": 0.014, "equity_us": -0.016},
    },
    "hist_005": {  # Nvidia Blockbuster Earnings (Corporate / Product)
        "1d": {"yield_10y": 0.0002, "credit_spread": -0.0008, "brent_oil": 0.002, "fx_dxy": -0.002, "equity_us": 0.012},
        "5d": {"yield_10y": 0.0005, "credit_spread": -0.0015, "brent_oil": 0.005, "fx_dxy": -0.004, "equity_us": 0.024},
    },
    "hist_006": {  # CrowdStrike Outage (Operational / Cyber)
        "1d": {"yield_10y": -0.0002, "credit_spread": 0.0010, "brent_oil": -0.002, "fx_dxy": 0.001, "equity_us": -0.009},
        "5d": {"yield_10y": -0.0005, "credit_spread": 0.0020, "brent_oil": -0.004, "fx_dxy": 0.002, "equity_us": -0.018},
    },
    "hist_007": {  # Japan Carry Trade Unwind (Macro / Liquidity)
        "1d": {"yield_10y": -0.0022, "credit_spread": 0.0045, "brent_oil": -0.025, "fx_dxy": -0.012, "equity_us": -0.030},
        "5d": {"yield_10y": -0.0015, "credit_spread": 0.0035, "brent_oil": -0.015, "fx_dxy": -0.008, "equity_us": -0.012},
    },
    "hist_008": {  # Fed 50 bps Rate Cut (Macroeconomic)
        "1d": {"yield_10y": -0.0012, "credit_spread": -0.0015, "brent_oil": 0.008, "fx_dxy": -0.006, "equity_us": 0.015},
        "5d": {"yield_10y": -0.0020, "credit_spread": -0.0025, "brent_oil": 0.014, "fx_dxy": -0.010, "equity_us": 0.028},
    },
    "hist_009": {  # Iran Missile Strike (Geopolitical)
        "1d": {"yield_10y": -0.0008, "credit_spread": 0.0025, "brent_oil": 0.048, "fx_dxy": 0.009, "equity_us": -0.018},
        "5d": {"yield_10y": -0.0012, "credit_spread": 0.0042, "brent_oil": 0.076, "fx_dxy": 0.016, "equity_us": -0.034},
    },
    "hist_010": {  # US Election Rally (Geopolitical)
        "1d": {"yield_10y": 0.0016, "credit_spread": -0.0012, "brent_oil": -0.004, "fx_dxy": 0.014, "equity_us": 0.025},
        "5d": {"yield_10y": 0.0028, "credit_spread": -0.0020, "brent_oil": -0.009, "fx_dxy": 0.022, "equity_us": 0.042},
    },
    "hist_011": {  # JPMorgan Record Earnings (Corporate / Product)
        "1d": {"yield_10y": 0.0003, "credit_spread": -0.0006, "brent_oil": 0.001, "fx_dxy": 0.001, "equity_us": 0.008},
        "5d": {"yield_10y": 0.0006, "credit_spread": -0.0012, "brent_oil": 0.003, "fx_dxy": 0.002, "equity_us": 0.016},
    },
    "hist_012": {  # US CPI Spike (Macroeconomic)
        "1d": {"yield_10y": 0.0018, "credit_spread": 0.0016, "brent_oil": 0.012, "fx_dxy": 0.008, "equity_us": -0.014},
        "5d": {"yield_10y": 0.0031, "credit_spread": 0.0028, "brent_oil": 0.022, "fx_dxy": 0.015, "equity_us": -0.025},
    },
    "hist_013": {  # Commercial Real Estate Default Rate (Credit Event)
        "1d": {"yield_10y": -0.0006, "credit_spread": 0.0024, "brent_oil": -0.006, "fx_dxy": 0.004, "equity_us": -0.012},
        "5d": {"yield_10y": -0.0014, "credit_spread": 0.0048, "brent_oil": -0.014, "fx_dxy": 0.008, "equity_us": -0.022},
    },
    "hist_014": {  # OPEC Production Cuts (Commodity / Supply Chain)
        "1d": {"yield_10y": 0.0006, "credit_spread": 0.0014, "brent_oil": 0.052, "fx_dxy": 0.003, "equity_us": -0.008},
        "5d": {"yield_10y": 0.0012, "credit_spread": 0.0026, "brent_oil": 0.088, "fx_dxy": 0.007, "equity_us": -0.019},
    },
    "hist_015": {  # DoJ Antitrust Lawsuit (Regulatory / Legal)
        "1d": {"yield_10y": 0.0000, "credit_spread": 0.0012, "brent_oil": 0.000, "fx_dxy": 0.000, "equity_us": -0.008},
        "5d": {"yield_10y": 0.0000, "credit_spread": 0.0022, "brent_oil": 0.000, "fx_dxy": 0.000, "equity_us": -0.016},
    },
}


@dataclass
class AnalogMatch:
    """Historical event analog record with similarity score and factor moves."""
    event_id: str
    title: str
    event_type: str
    similarity: float
    timestamp: str
    moves_1d: dict[str, float]
    moves_5d: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorShocks:
    """Distribution of forward factor shocks (p5, p50, p95)."""
    horizon_days: int
    factors: dict[str, dict[str, float]]  # {factor_name: {"p5": ..., "p50": ..., "p95": ...}}
    analogs: list[AnalogMatch]
    shrunk_to_prior: bool
    shrinkage_weight: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_days": self.horizon_days,
            "factors": self.factors,
            "analogs": [a.to_dict() for a in self.analogs],
            "shrunk_to_prior": self.shrunk_to_prior,
            "shrinkage_weight": self.shrinkage_weight,
            "message": self.message,
        }

    def get_shocks(self, percentile: str = "p50") -> dict[str, float]:
        """Convenience method to retrieve flat factor shock map for a percentile ('p5', 'p50', 'p95')."""
        return {k: v.get(percentile, 0.0) for k, v in self.factors.items()}


def load_historical_events(filepath: Path | str = "data/calibration/historical_shocks.json") -> list[dict[str, Any]]:
    """Load historical events from disk."""
    path = Path(filepath)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_analogs(
    query_title: str,
    event_type: str,
    top_k: int = 5,
    history_path: Path | str = "data/calibration/historical_shocks.json",
) -> list[AnalogMatch]:
    """Retrieve top-k historical analog events using FAISS cosine similarity over same event_type."""
    history = load_historical_events(history_path)
    if not history:
        return []

    # Filter to events of the exact same event_type
    candidates = [e for e in history if e.get("event_type") == event_type]
    # Fallback to all candidates if none of exact event_type exist
    if not candidates:
        candidates = history

    candidate_titles = [c.get("title", "") for c in candidates]
    if not candidate_titles:
        return []

    # 1. Embed query and candidates
    query_emb = embed_texts([query_title])
    cand_embs = embed_texts(candidate_titles)

    if cand_embs.shape[0] == 0 or query_emb.shape[0] == 0:
        return []

    # 2. FAISS Cosine Index (normalized vectors + inner product)
    dim = cand_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cand_embs)

    k = min(top_k, len(candidate_titles))
    distances, indices = index.search(query_emb, k)

    matches: list[AnalogMatch] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(candidates):
            continue
        c = candidates[idx]
        eid = c.get("event_id", "")
        # Lookup forward moves
        moves_info = HISTORICAL_FACTOR_MOVES.get(eid, {})
        m1 = moves_info.get("1d", {
            "yield_10y": 0.0, "credit_spread": 0.0015, "brent_oil": 0.0, "fx_dxy": 0.0, "equity_us": -0.010
        })
        m5 = moves_info.get("5d", {
            "yield_10y": 0.0, "credit_spread": 0.0030, "brent_oil": 0.0, "fx_dxy": 0.0, "equity_us": -0.020
        })

        matches.append(
            AnalogMatch(
                event_id=eid,
                title=c.get("title", ""),
                event_type=c.get("event_type", event_type),
                similarity=round(float(dist), 4),
                timestamp=c.get("timestamp", ""),
                moves_1d=m1,
                moves_5d=m5,
            )
        )

    # Sort descending by similarity
    matches.sort(key=lambda m: -m.similarity)
    return matches


def compute_factor_shocks(
    event_data: dict[str, Any] | str,
    event_type: str | None = None,
    horizon_days: int = 5,
    top_k: int = 5,
    min_analogs: int | None = None,
) -> FactorShocks:
    """Calculate p5, p50, and p95 factor moves for an event with prior shrinkage.

    Args:
        event_data: Event dictionary (e.g. from signals.jsonl) or event title string.
        event_type: Event type string (inferred from event_data if dict).
        horizon_days: Forward shock horizon (1 or 5 days, default 5).
        top_k: Max analogs to retrieve via FAISS (default 5).
        min_analogs: Minimum analogs threshold N before shrinking to prior (from config).

    Returns:
        FactorShocks object containing p5/p50/p95 distributions per factor and shrinkage metadata.
    """
    cfg = load_config()
    stress_cfg = cfg.get("stress", {})
    if min_analogs is None:
        min_analogs = int(stress_cfg.get("min_analogs", 3))

    prior_table = stress_cfg.get("prior_shocks", {})

    # Extract title and event_type
    if isinstance(event_data, dict):
        title = event_data.get("title", "")
        if not title and "evidence" in event_data and event_data["evidence"]:
            title = event_data["evidence"][0].get("title", "")
        etype = event_type or event_data.get("event_type", "Other")
    else:
        title = str(event_data)
        etype = event_type or "Other"

    # Default prior fallback for this event type
    priors: dict[str, float] = prior_table.get(
        etype,
        prior_table.get("Other", {
            "yield_10y": 0.0005,
            "credit_spread": 0.0010,
            "brent_oil": 0.0000,
            "fx_dxy": 0.0020,
            "equity_us": -0.0050,
        })
    )

    # Retrieve analogs via FAISS
    analogs = find_analogs(query_title=title, event_type=etype, top_k=top_k)
    k_found = len(analogs)

    factors_list = ["yield_10y", "credit_spread", "brent_oil", "fx_dxy", "equity_us"]
    horizon_key = "5d" if horizon_days >= 5 else "1d"

    # Gather empirical moves across analogs
    empirical_moves: dict[str, list[float]] = {f: [] for f in factors_list}
    for a in analogs:
        moves_map = a.moves_5d if horizon_key == "5d" else a.moves_1d
        for f in factors_list:
            if f in moves_map:
                empirical_moves[f].append(moves_map[f])

    factor_distributions: dict[str, dict[str, float]] = {}

    if k_found >= min_analogs:
        # Sufficient empirical analogs (k >= N)
        shrunk = False
        shrinkage_weight = 0.0
        msg = f"Empirical distribution calculated from {k_found} historical analogs (k >= N={min_analogs})."

        for f in factors_list:
            vals = empirical_moves[f]
            if vals:
                p5 = float(np.percentile(vals, 5))
                p50 = float(np.percentile(vals, 50))
                p95 = float(np.percentile(vals, 95))
            else:
                prior_val = priors.get(f, 0.0)
                p5, p50, p95 = prior_val * 0.5, prior_val, prior_val * 1.5

            factor_distributions[f] = {
                "p5": round(p5, 6),
                "p50": round(p50, 6),
                "p95": round(p95, 6),
            }
    else:
        # Fewer than N analogs (k < N): Shrink toward documented prior table
        shrunk = True
        # Bayesian shrinkage weight: lambda = k / N
        lam = k_found / float(min_analogs) if min_analogs > 0 else 0.0
        shrinkage_weight = round(1.0 - lam, 2)
        msg = (
            f"Fewer than N={min_analogs} analogs found (k={k_found}); "
            f"shrunk toward config prior table with weight {shrinkage_weight:.2f}."
        )

        for f in factors_list:
            prior_val = priors.get(f, 0.0)
            # Dispersion band for prior (e.g. 50% spread around prior)
            prior_disp = abs(prior_val) * 0.5 if abs(prior_val) > 1e-4 else 0.0010
            prior_p5 = prior_val - prior_disp
            prior_p50 = prior_val
            prior_p95 = prior_val + prior_disp

            vals = empirical_moves[f]
            if vals and k_found > 0:
                emp_p50 = float(np.median(vals))
                emp_p5 = float(np.min(vals))
                emp_p95 = float(np.max(vals))
            else:
                emp_p5, emp_p50, emp_p95 = prior_p5, prior_p50, prior_p95

            # Linear shrinkage combination
            p5 = lam * emp_p5 + (1.0 - lam) * prior_p5
            p50 = lam * emp_p50 + (1.0 - lam) * prior_p50
            p95 = lam * emp_p95 + (1.0 - lam) * prior_p95

            # Ensure order: p5 <= p50 <= p95
            lo = min(p5, p95)
            hi = max(p5, p95)
            mid = min(max(p50, lo), hi)

            factor_distributions[f] = {
                "p5": round(lo, 6),
                "p50": round(mid, 6),
                "p95": round(hi, 6),
            }

    return FactorShocks(
        horizon_days=horizon_days,
        factors=factor_distributions,
        analogs=analogs,
        shrunk_to_prior=shrunk,
        shrinkage_weight=shrinkage_weight,
        message=msg,
    )
