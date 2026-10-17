"""Tests for Phase 2 NLP pipeline: FinBERT sentiment, event classification, and entities."""

from __future__ import annotations

from src.ingest.schema import Document
from src.nlp.entities import extract_entities
from src.nlp.events import ALLOWED_EVENT_TYPES, classify_event
from src.nlp.pipeline import analyse, analyse_batch
from src.nlp.schema import AnalysedDoc
from src.nlp.sentiment import SentimentResult, analyse_sentiment


def test_sentiment_calculation():
    """Verify sentiment score equals P(pos) - P(neg) and values are in [-1, 1]."""
    text = "Company reports record profits and surging revenue growth."
    res = analyse_sentiment(text, use_cache=False)

    assert isinstance(res, SentimentResult)
    assert -1.0 <= res.score <= 1.0
    assert "positive" in res.probs
    assert "negative" in res.probs
    assert "neutral" in res.probs

    # Verify score formula: score = P(pos) - P(neg)
    expected_score = round(res.probs["positive"] - res.probs["negative"], 4)
    assert abs(res.score - expected_score) < 1e-3
    assert res.label in ("positive", "negative", "neutral")


def test_sentiment_caching(tmp_path, monkeypatch):
    """Verify sentiment result is written to and read from disk cache."""
    cache_dir = tmp_path / "cache"
    fake_config = {"nlp": {"cache_dir": str(cache_dir)}}
    monkeypatch.setattr("src.nlp.sentiment.load_config", lambda: fake_config)

    text = "Regional bank faces credit default following commercial mortgage slump."
    doc_id = "test_doc_sentiment_cache_001"

    # First call - computes and caches
    res1 = analyse_sentiment(text, doc_id=doc_id, use_cache=True)
    assert res1.label == "negative"

    # Verify cache file created
    cached_file = cache_dir / "sentiment" / f"{doc_id}.json"
    assert cached_file.exists()

    # Second call - should read from cache
    res2 = analyse_sentiment("Different text with same doc_id", doc_id=doc_id, use_cache=True)
    assert res2.score == res1.score
    assert res2.label == res1.label


def test_event_classification_types():
    """Verify event classification returns allowed event types with confidence and reason."""
    samples = [
        ("Major auto supplier misses coupon payment on senior notes", "Credit Event"),
        ("Federal Reserve signals interest rates will stay higher for longer", "Macroeconomic"),
        ("US announces 25% tariffs on imported steel and green tech", "Geopolitical"),
        ("Crude oil jumps past $90 on Gulf shipping corridor tensions", "Commodity/Supply Chain"),
        ("Antitrust regulators launch formal probe into big tech cloud licensing", "Regulatory/Legal"),
        ("Chevron agrees to acquire Hess Corp in $53bn all-stock transaction", "Merger/Acquisition"),
        ("Ransomware attack forces clearinghouse to process trades manually", "Operational/Cyber"),
        ("Tesla issues vehicle recall across North America", "Corporate/Product"),
    ]

    for headline, expected_type in samples:
        res = classify_event(headline, use_cache=False)
        assert res.event_type in ALLOWED_EVENT_TYPES
        assert res.event_type == expected_type
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.reason) > 5


def test_entity_extraction_companies_and_sectors():
    """Verify entity extraction matches catalog companies and maps sectors & countries."""
    text = "JPMorgan Chase and Barclays report rising debt defaults in German commercial real estate."
    entities = extract_entities(text)

    # Companies
    tickers = [c["ticker"] for c in entities.companies]
    assert "JPM" in tickers
    assert "BARC" in tickers

    # Sectors
    assert "Financials" in entities.sectors
    assert "Real Estate" in entities.sectors

    # Countries
    assert "Germany" in entities.countries
    assert "United States" in entities.countries
    assert "United Kingdom" in entities.countries


def test_analyse_doc_pipeline():
    """Verify analyse(doc) returns an AnalysedDoc with impact score and evidence."""
    doc = Document(
        id="test_pipeline_doc_100",
        source="newsapi",
        published_at="2026-10-16T12:00:00Z",
        title="Nordic construction conglomerate defaults on $750m private debt facility",
        text="Bondholders demand acceleration after missing scheduled coupon payment.",
        url="https://news.local/article-100",
        language="english",
        raw_meta={"domain": "ft.com"},
    )

    analysed = analyse(doc, use_cache=False)

    assert isinstance(analysed, AnalysedDoc)
    assert analysed.id == doc.id
    assert analysed.event_type == "Credit Event"
    assert analysed.sentiment_score < 0.0
    assert analysed.sentiment_label == "negative"
    assert 1.0 <= analysed.impact_score <= 10.0
    assert len(analysed.evidence) > 0


def test_analyse_batch_order():
    """Verify analyse_batch preserves input order and returns all items."""
    docs = [
        Document(
            id=f"batch_doc_{i}",
            source="gdelt",
            published_at="2026-10-16T12:00:00Z",
            title=f"Sample title {i} discussing interest rate inflation",
            text=f"Sample text {i}",
            url=f"https://news.local/{i}",
            language="english",
            raw_meta={},
        )
        for i in range(5)
    ]

    results = analyse_batch(docs, batch_size=2, use_cache=False)
    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.id == f"batch_doc_{i}"
