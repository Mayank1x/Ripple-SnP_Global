"""Schema for NLP-analysed documents.

Contains sentiment scoring, event classification, extracted entities,
impact calibration, and evidence snippets for regulatory audit trails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.ingest.schema import Document
from src.nlp.entities import ExtractedEntities

ALLOWED_EVENT_TYPES = (
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Regulatory/Legal",
    "Merger/Acquisition",
    "Commodity/Supply Chain",
    "Operational/Cyber",
    "Corporate/Product",
    "Other",
)


@dataclass
class AnalysedDoc:
    """Document enriched with NLP risk signals and audit evidence."""

    id: str
    source: str
    published_at: str
    title: str
    text: str
    url: str
    language: str
    raw_meta: dict[str, Any]

    # Sentiment (-1.0 severe negative to +1.0 strong positive)
    sentiment_score: float
    sentiment_label: str  # positive, negative, neutral
    sentiment_probs: dict[str, float] = field(default_factory=dict)

    # Event classification
    event_type: str = "Other"
    event_confidence: float = 0.5
    event_reason: str = ""

    # Entities
    entities: dict[str, Any] = field(default_factory=dict)

    # Risk impact score (1 to 10)
    impact_score: float = 1.0

    # Evidence quote for audit trail
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysedDoc:
        """Create AnalysedDoc from dictionary."""
        return cls(**data)

    @classmethod
    def from_document(
        cls,
        doc: Document,
        sentiment_score: float,
        sentiment_label: str,
        sentiment_probs: dict[str, float],
        event_type: str,
        event_confidence: float,
        event_reason: str,
        entities: ExtractedEntities | dict[str, Any],
        impact_score: float,
        evidence: str,
    ) -> AnalysedDoc:
        """Construct AnalysedDoc from an ingested Document and NLP outputs."""
        ent_dict = entities.to_dict() if isinstance(entities, ExtractedEntities) else entities
        return cls(
            id=doc.id,
            source=doc.source,
            published_at=doc.published_at,
            title=doc.title,
            text=doc.text,
            url=doc.url,
            language=doc.language,
            raw_meta=doc.raw_meta,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            sentiment_probs=sentiment_probs,
            event_type=event_type,
            event_confidence=event_confidence,
            event_reason=event_reason,
            entities=ent_dict,
            impact_score=impact_score,
            evidence=evidence,
        )
