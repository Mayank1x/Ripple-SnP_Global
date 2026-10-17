"""Ripple Risk Scoring & Signal Generation Module (Phase 3)."""

from src.scoring.cluster import cluster_documents, embed_texts
from src.scoring.impact import calculate_impact, compute_novelty, score_events
from src.scoring.pipeline import process_signals
from src.scoring.schema import Event, EvidenceItem, Signal
from src.scoring.store import get_signal_by_id, init_db, query_signals, save_signals

__all__ = [
    "Event",
    "EvidenceItem",
    "Signal",
    "cluster_documents",
    "embed_texts",
    "calculate_impact",
    "compute_novelty",
    "score_events",
    "process_signals",
    "init_db",
    "save_signals",
    "query_signals",
    "get_signal_by_id",
]
