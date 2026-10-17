"""Impact score calculation for real-world events.

Transparent formula:
  raw_impact = w_prior * event_prior
             + w_sentiment * (|sentiment| * confidence)
             + w_breadth * log(distinct_sources)
             + w_novelty * (1 - max_sim_to_events_last_30d)

Normalized to [0.0, 1.0] and binned to [1.0, 10.0].
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from src.ingest.helpers import load_config
from src.scoring.cluster import embed_texts, parse_timestamp
from src.scoring.schema import Event


def compute_novelty(
    event_embedding: np.ndarray,
    prior_embeddings: list[np.ndarray],
) -> float:
    """Calculate event novelty: 1 - max cosine similarity to events in last 30 days.

    Args:
        event_embedding: Normalized embedding vector of the current event title.
        prior_embeddings: List of normalized embedding vectors of events in the preceding 30 days.

    Returns:
        Novelty score in [0.0, 1.0]. Returns 1.0 if no prior events exist.
    """
    if not prior_embeddings:
        return 1.0

    # Stack prior embeddings
    prior_matrix = np.vstack(prior_embeddings)  # Shape (M, dim)
    # Cosine similarities (dot products since normalized)
    sims = np.dot(prior_matrix, event_embedding.squeeze())
    max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
    # Clamp max similarity to [0.0, 1.0]
    max_sim_clamped = max(0.0, min(1.0, max_sim))
    return round(1.0 - max_sim_clamped, 4)


def calculate_impact(
    event: Event,
    novelty: float = 1.0,
) -> tuple[float, float, dict[str, float]]:
    """Compute raw_impact (0-1) and binned impact_score (1-10) using transparent formula.

    Returns:
        (raw_impact, impact_score, component_breakdown)
    """
    cfg = load_config()
    impact_cfg = cfg.get("scoring", {}).get("impact", {})

    weights = impact_cfg.get(
        "weights",
        {
            "event_prior": 0.35,
            "sentiment_confidence": 0.30,
            "source_breadth": 0.15,
            "novelty": 0.20,
        },
    )
    event_priors = impact_cfg.get(
        "event_priors",
        {
            "Credit Event": 0.95,
            "Geopolitical": 0.85,
            "Macroeconomic": 0.80,
            "Regulatory/Legal": 0.75,
            "Operational/Cyber": 0.70,
            "Commodity/Supply Chain": 0.65,
            "Merger/Acquisition": 0.60,
            "Corporate/Product": 0.50,
            "Other": 0.30,
        },
    )

    # 1. Component 1: Event-type prior (0.0 to 1.0)
    c1_prior = float(event_priors.get(event.event_type, 0.50))

    # 2. Component 2: Sentiment intensity x confidence (0.0 to 1.0)
    sentiment_magnitude = abs(event.sentiment_score)
    c2_sentiment = float(sentiment_magnitude * event.confidence)

    # 3. Component 3: Source breadth (log-scaled 0.0 to 1.0)
    # Standard 4 distinct sources: GDELT, NewsAPI, RSS, Tweets
    # Scale log(1 + n) / log(1 + 4)
    n_sources = max(1, event.n_distinct_sources)
    c3_breadth = float(math.log(1.0 + n_sources) / math.log(1.0 + 4.0))
    c3_breadth = min(1.0, max(0.0, c3_breadth))

    # 4. Component 4: Novelty (0.0 to 1.0)
    c4_novelty = float(max(0.0, min(1.0, novelty)))

    # Weighted sum
    w1 = float(weights.get("event_prior", 0.35))
    w2 = float(weights.get("sentiment_confidence", 0.30))
    w3 = float(weights.get("source_breadth", 0.15))
    w4 = float(weights.get("novelty", 0.20))
    total_w = w1 + w2 + w3 + w4

    if total_w <= 0:
        total_w = 1.0

    raw_impact = (w1 * c1_prior + w2 * c2_sentiment + w3 * c3_breadth + w4 * c4_novelty) / total_w
    raw_impact = round(max(0.0, min(1.0, raw_impact)), 4)

    # Bin to 1.0 - 10.0
    impact_score = round(1.0 + 9.0 * raw_impact, 1)

    components = {
        "event_prior": round(c1_prior, 4),
        "sentiment_intensity": round(c2_sentiment, 4),
        "source_breadth": round(c3_breadth, 4),
        "novelty": round(c4_novelty, 4),
        "raw_impact": raw_impact,
    }

    return raw_impact, impact_score, components


def score_events(events: list[Event]) -> list[tuple[Event, float]]:
    """Compute novelty and impact scores across an ordered list of events.

    Maintains a rolling 30-day history of event embeddings to assess novelty.

    Returns:
        List of (Event, impact_score) tuples.
    """
    if not events:
        return []

    # Sort events by first_seen
    sorted_events = sorted(events, key=lambda e: parse_timestamp(e.first_seen))

    # Pre-embed all event titles
    titles = [e.title for e in sorted_events]
    embeddings = embed_texts(titles)

    results: list[tuple[Event, float]] = []

    # History: list of (datetime, embedding)
    history: list[tuple[datetime, np.ndarray]] = []
    window_30d_seconds = 30 * 86400.0

    for i, event in enumerate(sorted_events):
        event_dt = parse_timestamp(event.first_seen)
        event_emb = embeddings[i : i + 1]

        # Prior embeddings within last 30 days
        recent_priors = [
            prior_emb
            for prior_dt, prior_emb in history
            if 0 < (event_dt - prior_dt).total_seconds() <= window_30d_seconds
        ]

        novelty = compute_novelty(event_emb, recent_priors)
        raw_impact, _uncalibrated_score, _ = calculate_impact(event, novelty=novelty)

        # Apply empirical isotonic calibration if available
        from src.scoring.calibration import get_calibrator

        calibrated_score, cal_meta = get_calibrator().predict_score(raw_impact)

        results.append((event, calibrated_score, raw_impact, cal_meta))
        history.append((event_dt, event_emb))

    return results
