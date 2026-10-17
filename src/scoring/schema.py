"""Data schemas for Phase 3: Events, Signals, Evidence, and Clustering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.nlp.schema import AnalysedDoc


@dataclass
class EvidenceItem:
    """An evidence snippet supporting a clustered event."""

    url: str
    title: str
    source: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> EvidenceItem:
        return cls(**data)


@dataclass
class Event:
    """Aggregated real-world event formed by clustering related documents."""

    event_id: str
    title: str
    first_seen: str
    last_seen: str
    event_type: str
    confidence: float
    sentiment_score: float
    entities: dict[str, Any]
    n_docs: int
    n_distinct_sources: int
    docs: list[AnalysedDoc] = field(default_factory=list)
    divergence_flag: bool = False
    evidence: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Convert AnalysedDoc objects to dicts
        data["docs"] = [d.to_dict() if isinstance(d, AnalysedDoc) else d for d in self.docs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        copied = dict(data)
        if "docs" in copied and copied["docs"]:
            copied["docs"] = [
                AnalysedDoc.from_dict(d) if isinstance(d, dict) else d for d in copied["docs"]
            ]
        return cls(**copied)


@dataclass
class Signal:
    """Actionable risk signal published to data/signals/ and served by FastAPI."""

    event_id: str
    timestamp: str
    entities: dict[str, Any]
    sentiment_score: float
    event_type: str
    impact_score: float  # Calibrated impact score (1.0 to 10.0)
    confidence: float
    divergence_flag: bool
    evidence: list[dict[str, str]]
    model_versions: dict[str, str]
    raw_impact: float = 0.0  # Raw impact before calibration (0.0 to 1.0)
    calibrated_impact: float = 0.0  # Predicted abnormal market move (%)
    calibration_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signal:
        copied = dict(data)
        # Handle older signals without calibration fields
        if "raw_impact" not in copied:
            copied["raw_impact"] = 0.0
        if "calibrated_impact" not in copied:
            copied["calibrated_impact"] = 0.0
        if "calibration_metadata" not in copied:
            copied["calibration_metadata"] = {}
        return cls(**copied)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
