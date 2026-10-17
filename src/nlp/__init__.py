"""Ripple NLP Module: Sentiment, Event Classification, Entities, and Document Enrichment."""

from src.nlp.entities import ExtractedEntities, extract_entities
from src.nlp.events import EventClassificationResult, classify_event, classify_event_batch
from src.nlp.pipeline import analyse, analyse_batch
from src.nlp.schema import ALLOWED_EVENT_TYPES, AnalysedDoc
from src.nlp.sentiment import SentimentResult, analyse_sentiment, analyse_sentiment_batch

__all__ = [
    "analyse",
    "analyse_batch",
    "AnalysedDoc",
    "ALLOWED_EVENT_TYPES",
    "analyse_sentiment",
    "analyse_sentiment_batch",
    "SentimentResult",
    "classify_event",
    "classify_event_batch",
    "EventClassificationResult",
    "extract_entities",
    "ExtractedEntities",
]
