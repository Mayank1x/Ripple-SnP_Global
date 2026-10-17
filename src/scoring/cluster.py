"""Title embedding, FAISS rolling-window clustering, and event aggregation.

Steps:
1. Embed document titles using sentence-transformers (all-MiniLM-L6-v2).
2. Group documents by cosine similarity inside a rolling 48h window using FAISS.
3. Aggregate constituent documents into a single real-world Event:
   - Sentiment: average weighted by source reliability.
   - Event type: majority vote across cluster docs.
   - Entities: union of extracted entities.
   - Source divergence: flag if news sentiment and social sentiment diverge.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.ingest.helpers import load_config
from src.nlp.schema import AnalysedDoc
from src.scoring.schema import Event

# Global model cache
_EMBED_MODEL = None


def _get_embed_model():
    """Lazily load sentence-transformers model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        cfg = load_config()
        model_name = cfg.get("scoring", {}).get("cluster", {}).get(
            "model_name", "all-MiniLM-L6-v2"
        )
        _EMBED_MODEL = SentenceTransformer(model_name, device="cpu")
    return _EMBED_MODEL


def parse_timestamp(ts: str | None) -> datetime:
    """Robust ISO timestamp parser returning UTC datetime."""
    if not ts:
        return datetime.now(timezone.utc)
    ts_clean = str(ts).strip()
    # Normalize trailing Z to +00:00
    if ts_clean.endswith("Z"):
        ts_clean = ts_clean[:-1] + "+00:00"
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(ts_clean, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Fallback to current time
    return datetime.now(timezone.utc)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Compute L2-normalized embeddings for a list of text strings."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_embed_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.ascontiguousarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Rolling Window Event Clustering with FAISS
# ---------------------------------------------------------------------------


def cluster_documents(
    docs: list[AnalysedDoc],
    similarity_threshold: float | None = None,
    window_hours: float | None = None,
) -> list[Event]:
    """Group AnalysedDoc items into real-world Events within a rolling time window.

    Args:
        docs: List of analysed documents.
        similarity_threshold: Minimum cosine similarity to merge into existing cluster.
        window_hours: Max hours between docs in the same active cluster.

    Returns:
        List of aggregated Event objects.
    """
    if not docs:
        return []

    cfg = load_config()
    cluster_cfg = cfg.get("scoring", {}).get("cluster", {})
    if similarity_threshold is None:
        similarity_threshold = float(cluster_cfg.get("similarity_threshold", 0.72))
    if window_hours is None:
        window_hours = float(cluster_cfg.get("rolling_window_hours", 48.0))

    window_seconds = window_hours * 3600.0

    # 1. Sort documents chronologically
    sorted_docs = sorted(docs, key=lambda d: parse_timestamp(d.published_at))

    # 2. Compute embeddings for all document titles
    titles = [d.title.strip() if d.title else "Untitled Article" for d in sorted_docs]
    embeddings = embed_texts(titles)  # Shape: (N, 384), already normalized

    # 3. Cluster with rolling FAISS index
    import faiss

    dim = embeddings.shape[1]

    # Active cluster state
    # Each entry: {"docs": [AnalysedDoc], "embeddings": [np.ndarray], "latest_dt": datetime, "centroid": np.ndarray}
    clusters: list[dict[str, Any]] = []

    for i, doc in enumerate(sorted_docs):
        doc_emb = embeddings[i : i + 1]  # shape (1, dim)
        doc_dt = parse_timestamp(doc.published_at)

        best_cluster_idx = -1

        if clusters:
            # Build active candidates whose latest_dt is within rolling window
            active_indices = [
                c_idx
                for c_idx, c in enumerate(clusters)
                if abs((doc_dt - c["latest_dt"]).total_seconds()) <= window_seconds
            ]

            if active_indices:
                # Query centroids of active clusters using FAISS IndexFlatIP
                active_centroids = np.vstack([clusters[idx]["centroid"] for idx in active_indices])
                # Ensure normalized
                faiss.normalize_L2(active_centroids)

                index = faiss.IndexFlatIP(dim)
                index.add(active_centroids)

                k = min(len(active_indices), 5)
                sims, rank_indices = index.search(doc_emb, k)

                top_sim = float(sims[0][0])
                top_active_idx = active_indices[rank_indices[0][0]]

                if top_sim >= similarity_threshold:
                    best_cluster_idx = top_active_idx

        if best_cluster_idx >= 0:
            # Add to matching cluster and update centroid
            c = clusters[best_cluster_idx]
            c["docs"].append(doc)
            c["embeddings"].append(doc_emb[0])
            c["latest_dt"] = max(c["latest_dt"], doc_dt)

            # Update centroid
            new_centroid = np.mean(c["embeddings"], axis=0, keepdims=True).astype(np.float32)
            faiss.normalize_L2(new_centroid)
            c["centroid"] = new_centroid[0]
        else:
            # Start new cluster
            clusters.append(
                {
                    "docs": [doc],
                    "embeddings": [doc_emb[0]],
                    "latest_dt": doc_dt,
                    "centroid": doc_emb[0],
                }
            )

    # 4. Aggregate each cluster into an Event
    events: list[Event] = []
    for c in clusters:
        event = aggregate_cluster(c["docs"], c["embeddings"])
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Cluster Aggregation & Source Divergence
# ---------------------------------------------------------------------------


def aggregate_cluster(cluster_docs: list[AnalysedDoc], cluster_embeddings: list[np.ndarray]) -> Event:
    """Aggregate a group of related AnalysedDoc items into a unified Event."""
    cfg = load_config()
    agg_cfg = cfg.get("scoring", {}).get("aggregation", {})
    source_weights = agg_cfg.get(
        "source_weights",
        {"newsapi": 1.0, "gdelt": 1.0, "rss": 1.0, "tweets": 0.6, "benchmark": 1.0, "synthetic": 1.0},
    )

    # 1. Timestamps
    doc_dates = [parse_timestamp(d.published_at) for d in cluster_docs]
    first_seen_dt = min(doc_dates)
    last_seen_dt = max(doc_dates)
    first_seen_str = first_seen_dt.isoformat()
    last_seen_str = last_seen_dt.isoformat()

    # 2. Representative title (medoid closest to cluster centroid)
    centroid = np.mean(cluster_embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-6:
        centroid /= norm
    sims = [float(np.dot(emb, centroid)) for emb in cluster_embeddings]
    rep_idx = int(np.argmax(sims))
    rep_title = cluster_docs[rep_idx].title

    # 3. Source-weighted sentiment score
    weighted_sentiment_sum = 0.0
    total_sentiment_weight = 0.0
    for d in cluster_docs:
        w = float(source_weights.get(d.source.lower(), 1.0))
        weighted_sentiment_sum += d.sentiment_score * w
        total_sentiment_weight += w

    if total_sentiment_weight > 0.0:
        avg_sentiment = round(weighted_sentiment_sum / total_sentiment_weight, 4)
    else:
        avg_sentiment = 0.0

    # 4. Majority vote for event_type weighted by confidence
    vote_weights: dict[str, float] = {}
    for d in cluster_docs:
        vote_weights[d.event_type] = vote_weights.get(d.event_type, 0.0) + d.event_confidence

    winning_event_type = max(vote_weights, key=vote_weights.get) if vote_weights else "Other"

    # Winning confidence = mean confidence of documents voting for the winner
    winning_confs = [d.event_confidence for d in cluster_docs if d.event_type == winning_event_type]
    event_confidence = round(float(np.mean(winning_confs)) if winning_confs else 0.80, 4)

    # 5. Union of entities
    merged_companies: dict[str, dict[str, Any]] = {}
    merged_sectors: set[str] = set()
    merged_countries: set[str] = set()
    merged_commodities: set[str] = set()
    merged_unmapped_orgs: set[str] = set()
    merged_unmapped_gpes: set[str] = set()

    for d in cluster_docs:
        ents = d.entities or {}
        for comp in ents.get("companies", []):
            ticker = comp.get("ticker") or comp.get("name")
            if ticker and ticker not in merged_companies:
                merged_companies[ticker] = comp
        for s in ents.get("sectors", []):
            merged_sectors.add(s)
        for c in ents.get("countries", []):
            merged_countries.add(c)
        for comm in ents.get("commodities", []):
            merged_commodities.add(comm)
        for org in ents.get("unmapped_orgs", []):
            merged_unmapped_orgs.add(org)
        for gpe in ents.get("unmapped_gpes", []):
            merged_unmapped_gpes.add(gpe)

    aggregated_entities = {
        "companies": list(merged_companies.values()),
        "sectors": sorted(merged_sectors),
        "countries": sorted(merged_countries),
        "commodities": sorted(merged_commodities),
        "unmapped_orgs": sorted(merged_unmapped_orgs),
        "unmapped_gpes": sorted(merged_unmapped_gpes),
    }

    # 6. Source counts & divergence flag
    distinct_sources = sorted({d.source.lower() for d in cluster_docs})
    n_distinct_sources = len(distinct_sources)

    div_cfg = agg_cfg.get("divergence", {})
    div_threshold = float(div_cfg.get("threshold", 0.40))
    min_news_docs = int(div_cfg.get("min_news_docs", 1))
    min_social_docs = int(div_cfg.get("min_social_docs", 1))

    news_docs = [d for d in cluster_docs if d.source.lower() in ("gdelt", "newsapi", "rss")]
    social_docs = [d for d in cluster_docs if d.source.lower() in ("tweets", "social")]

    divergence_flag = False
    if len(news_docs) >= min_news_docs and len(social_docs) >= min_social_docs:
        news_sentiment = float(np.mean([d.sentiment_score for d in news_docs]))
        social_sentiment = float(np.mean([d.sentiment_score for d in social_docs]))
        if abs(news_sentiment - social_sentiment) >= div_threshold:
            divergence_flag = True

    # 7. Evidence list (up to 5 items)
    evidence: list[dict[str, str]] = []
    for d in cluster_docs[:5]:
        evidence.append(
            {
                "url": d.url,
                "title": d.title,
                "source": d.source,
                "snippet": d.evidence or d.text[:160],
            }
        )

    # 8. Deterministic Event ID
    # Hash of first doc ID + first seen timestamp
    id_seed = f"{cluster_docs[0].id}_{first_seen_str}"
    event_id = "evt_" + hashlib.sha256(id_seed.encode("utf-8")).hexdigest()[:12]

    return Event(
        event_id=event_id,
        title=rep_title,
        first_seen=first_seen_str,
        last_seen=last_seen_str,
        event_type=winning_event_type,
        confidence=event_confidence,
        sentiment_score=avg_sentiment,
        entities=aggregated_entities,
        n_docs=len(cluster_docs),
        n_distinct_sources=n_distinct_sources,
        docs=cluster_docs,
        divergence_flag=divergence_flag,
        evidence=evidence,
    )
