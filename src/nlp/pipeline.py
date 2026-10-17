"""End-to-end NLP analysis pipeline.

Transforms ingested Documents into AnalysedDocs containing:
  - Sentiment score (-1.0 to 1.0) and individual probabilities (FinBERT)
  - Event type classification with confidence and rationale (Zero-Shot / Gemini)
  - Entity extraction (companies, sectors, countries, commodities)
  - Impact score calibration (1 to 10)
  - Evidence text snippet for audit trails
"""

from __future__ import annotations

import re

from src.ingest.schema import Document
from src.nlp.entities import extract_entities
from src.nlp.events import classify_event_batch
from src.nlp.schema import AnalysedDoc
from src.nlp.sentiment import analyse_sentiment_batch

# Baseline impact score mapping by event type
_EVENT_BASE_IMPACT = {
    "Credit Event": 7.5,
    "Operational/Cyber": 7.0,
    "Geopolitical": 6.5,
    "Macroeconomic": 6.0,
    "Commodity/Supply Chain": 5.5,
    "Regulatory/Legal": 5.0,
    "Merger/Acquisition": 4.5,
    "Corporate/Product": 4.0,
    "Other": 2.0,
}


def _calculate_impact_score(
    event_type: str,
    sentiment_score: float,
    company_count: int,
    sector_count: int,
) -> float:
    """Calculate calibrated risk impact score from 1.0 to 10.0."""
    base = _EVENT_BASE_IMPACT.get(event_type, 3.0)

    # Negative sentiment or severe shock amplifies risk impact
    sentiment_multiplier = abs(sentiment_score) * 2.0

    # Systemic breadth (multiple companies/sectors impacted)
    breadth_boost = min(1.5, (company_count * 0.3) + (sector_count * 0.2))

    raw_score = base + sentiment_multiplier + breadth_boost
    return round(max(1.0, min(10.0, raw_score)), 1)


def _extract_evidence(doc: Document, event_reason: str) -> str:
    """Extract a representative evidence snippet for the audit trail."""
    full_text = f"{doc.title}. {doc.text}".strip()
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    # If event trigger keyword is known, find the sentence that contains it
    trigger_match = re.search(r"Trigger: '([^']+)'", event_reason)
    if trigger_match:
        trigger = trigger_match.group(1).lower()
        for sent in sentences:
            if trigger in sent.lower():
                return sent.strip()[:200]

    # Otherwise return the title or first meaningful sentence
    if doc.title:
        return doc.title.strip()[:200]
    return sentences[0][:200] if sentences else ""


def analyse(doc: Document, use_cache: bool = True) -> AnalysedDoc:
    """Analyse a single Document into an AnalysedDoc."""
    results = analyse_batch([doc], batch_size=1, use_cache=use_cache)
    return results[0]


def analyse_batch(
    docs: list[Document],
    batch_size: int = 16,
    use_cache: bool = True,
) -> list[AnalysedDoc]:
    """Analyse a batch of Documents efficiently with CPU batching and caching."""
    if not docs:
        return []

    texts = [f"{d.title}. {d.text}".strip() if d.text != d.title else d.title for d in docs]
    doc_ids = [d.id for d in docs]

    # 1. Sentiment analysis (FinBERT with CPU batching)
    sent_results = analyse_sentiment_batch(texts, doc_ids=doc_ids, batch_size=batch_size, use_cache=use_cache)

    # 2. Event type classification
    event_results = classify_event_batch(texts, doc_ids=doc_ids, use_cache=use_cache)

    analysed_docs: list[AnalysedDoc] = []

    for i, doc in enumerate(docs):
        sent = sent_results[i]
        event = event_results[i]

        # 3. Entity extraction
        entities = extract_entities(texts[i])

        # 4. Impact score calibration
        impact = _calculate_impact_score(
            event_type=event.event_type,
            sentiment_score=sent.score,
            company_count=len(entities.companies),
            sector_count=len(entities.sectors),
        )

        # 5. Audit evidence
        evidence = _extract_evidence(doc, event.reason)

        analysed = AnalysedDoc.from_document(
            doc=doc,
            sentiment_score=sent.score,
            sentiment_label=sent.label,
            sentiment_probs=sent.probs,
            event_type=event.event_type,
            event_confidence=event.confidence,
            event_reason=event.reason,
            entities=entities,
            impact_score=impact,
            evidence=evidence,
        )
        analysed_docs.append(analysed)

    return analysed_docs
