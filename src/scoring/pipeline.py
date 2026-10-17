"""End-to-end risk scoring pipeline: clusters AnalysedDocs into Events, computes impact, and produces Signals."""

from __future__ import annotations

from src.nlp.schema import AnalysedDoc
from src.scoring.cluster import cluster_documents
from src.scoring.impact import score_events
from src.scoring.schema import Event, Signal
from src.scoring.store import save_signals

DEFAULT_MODEL_VERSIONS = {
    "sentiment": "ProsusAI/finbert",
    "clustering": "all-MiniLM-L6-v2",
    "events": "zero-shot-mnli-or-gemini",
    "scoring_version": "1.0.0",
}


def process_signals(
    analysed_docs: list[AnalysedDoc],
    save_to_disk: bool = True,
    similarity_threshold: float | None = None,
    window_hours: float | None = None,
) -> list[Signal]:
    """Execute end-to-end clustering, aggregation, and impact scoring pipeline.

    Args:
        analysed_docs: List of AnalysedDoc objects from Phase 2 NLP analysis.
        save_to_disk: If True, writes signals to SQLite and signals.jsonl.
        similarity_threshold: Optional override for cosine clustering threshold.
        window_hours: Optional override for rolling window hours.

    Returns:
        List of generated Signal objects.
    """
    if not analysed_docs:
        return []

    # 1. Cluster documents into real-world events
    events: list[Event] = cluster_documents(
        analysed_docs,
        similarity_threshold=similarity_threshold,
        window_hours=window_hours,
    )

    # 2. Score events with novelty and impact
    scored_pairs = score_events(events)

    # 3. Create Signals with calibrated impact scores
    signals: list[Signal] = []
    for item in scored_pairs:
        event = item[0]
        calibrated_score = float(item[1])
        raw_impact = float(item[2]) if len(item) > 2 else 0.0
        cal_meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}

        sig = Signal(
            event_id=event.event_id,
            timestamp=event.first_seen,
            entities=event.entities,
            sentiment_score=event.sentiment_score,
            event_type=event.event_type,
            impact_score=calibrated_score,
            confidence=event.confidence,
            divergence_flag=event.divergence_flag,
            evidence=event.evidence,
            model_versions=DEFAULT_MODEL_VERSIONS,
            raw_impact=raw_impact,
            calibrated_impact=float(cal_meta.get("predicted_abnormal_move", 0.0)),
            calibration_metadata=cal_meta,
        )
        signals.append(sig)

    # 4. Save to SQLite and JSONL
    if save_to_disk:
        save_signals(signals)

    return signals
