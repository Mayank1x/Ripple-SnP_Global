"""Ripple Risk Scoring, Event Study, and Calibration Module (Phases 3 & 4)."""

from src.scoring.calibration import (
    CalibrationMetadata,
    ImpactCalibrator,
    get_calibrator,
)
from src.scoring.cluster import cluster_documents, embed_texts
from src.scoring.event_study import (
    compute_event_study,
    compute_factor_daily_changes,
    fetch_price_history,
    resolve_event_asset,
)
from src.scoring.impact import calculate_impact, compute_novelty, score_events
from src.scoring.pipeline import process_signals
from src.scoring.schema import Event, EvidenceItem, Signal
from src.scoring.store import get_signal_by_id, init_db, query_signals, save_signals

__all__ = [
    "CalibrationMetadata",
    "Event",
    "EvidenceItem",
    "ImpactCalibrator",
    "Signal",
    "calculate_impact",
    "cluster_documents",
    "compute_event_study",
    "compute_factor_daily_changes",
    "compute_novelty",
    "embed_texts",
    "fetch_price_history",
    "get_calibrator",
    "get_signal_by_id",
    "init_db",
    "process_signals",
    "query_signals",
    "resolve_event_asset",
    "save_signals",
    "score_events",
]
