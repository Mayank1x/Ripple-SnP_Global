"""Tests for Phase 3 scoring: clustering, event aggregation, source divergence, impact, and store."""

from __future__ import annotations

import numpy as np

from src.nlp.schema import AnalysedDoc
from src.scoring.cluster import aggregate_cluster, cluster_documents, embed_texts
from src.scoring.impact import calculate_impact, compute_novelty
from src.scoring.schema import Signal
from src.scoring.store import get_signal_by_id, query_signals, save_signals


def _make_dummy_doc(
    doc_id: str,
    title: str,
    source: str = "gdelt",
    published_at: str = "2026-10-15T12:00:00Z",
    sentiment_score: float = -0.80,
    sentiment_label: str = "negative",
    event_type: str = "Credit Event",
    confidence: float = 0.90,
    entities: dict | None = None,
) -> AnalysedDoc:
    return AnalysedDoc(
        id=doc_id,
        source=source,
        published_at=published_at,
        title=title,
        text=title,
        url=f"https://news.local/{doc_id}",
        language="english",
        raw_meta={},
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        sentiment_probs={"positive": 0.05, "negative": 0.85, "neutral": 0.10},
        event_type=event_type,
        event_confidence=confidence,
        event_reason="Matched credit distress",
        entities=entities or {"companies": [], "sectors": ["Financials"], "countries": [], "commodities": []},
        impact_score=7.5,
        evidence=title,
    )


def test_embed_texts_shape_and_normalization():
    """Verify embeddings have shape (N, 384) and L2 norm equals 1."""
    texts = [
        "JPMorgan faces default risk on leveraged buyout loans",
        "European Central Bank holds interest rates steady",
    ]
    embs = embed_texts(texts)
    assert embs.shape == (2, 384)
    for i in range(len(texts)):
        norm = float(np.linalg.norm(embs[i]))
        assert abs(norm - 1.0) < 1e-3


def test_cluster_documents_similarity():
    """Verify documents with similar titles in the same 48h window merge into one Event."""
    doc1 = _make_dummy_doc("d1", "Regional bank defaults on senior debt obligations", published_at="2026-10-15T10:00:00Z")
    doc2 = _make_dummy_doc("d2", "Regional bank enters default following missed debt payments", published_at="2026-10-15T18:00:00Z")
    doc3 = _make_dummy_doc("d3", "China issues export curbs on rare earth minerals", event_type="Geopolitical", published_at="2026-10-15T12:00:00Z")

    events = cluster_documents([doc1, doc2, doc3], similarity_threshold=0.65, window_hours=48)
    assert len(events) == 2

    # One event should contain doc1 and doc2 (bank default)
    bank_events = [e for e in events if e.n_docs == 2]
    assert len(bank_events) == 1
    assert bank_events[0].event_type == "Credit Event"


def test_event_aggregation_weighted_sentiment():
    """Verify source weights affect aggregated sentiment score."""
    # News says negative (-0.80), tweets says positive (+0.40)
    # News weight 1.0, tweets weight 0.6 -> weighted avg = (-0.80*1.0 + 0.40*0.6) / 1.6 = -0.56 / 1.6 = -0.35
    d_news = _make_dummy_doc("d_n", "Bank defaults on notes", source="newsapi", sentiment_score=-0.80)
    d_soc = _make_dummy_doc("d_s", "Bank defaults on notes", source="tweets", sentiment_score=0.40)

    embs = embed_texts([d_news.title, d_soc.title])
    event = aggregate_cluster([d_news, d_soc], [embs[0], embs[1]])

    assert event.sentiment_score < 0.0
    # Expected: (-0.8*1.0 + 0.4*0.6) / 1.6 = -0.35
    assert abs(event.sentiment_score - (-0.35)) < 0.05


def test_source_divergence_flag():
    """Verify divergence_flag triggers when news and social sentiment differ significantly."""
    d_news = _make_dummy_doc("dn", "Tech firm revenue falls", source="gdelt", sentiment_score=-0.70)
    d_social = _make_dummy_doc("ds", "Tech firm revenue falls", source="tweets", sentiment_score=0.30)

    embs = embed_texts([d_news.title, d_social.title])
    event = aggregate_cluster([d_news, d_social], [embs[0], embs[1]])

    # Sentiment diff = |-0.70 - 0.30| = 1.00 > 0.40 threshold
    assert event.divergence_flag is True


def test_impact_calculation_formula():
    """Verify raw_impact formula components and 1-10 range."""
    doc = _make_dummy_doc("d1", "Major default", sentiment_score=-0.90, confidence=0.95, event_type="Credit Event")
    embs = embed_texts([doc.title])
    event = aggregate_cluster([doc], [embs[0]])

    raw_impact, impact_score, comps = calculate_impact(event, novelty=1.0)

    assert 0.0 <= raw_impact <= 1.0
    assert 1.0 <= impact_score <= 10.0
    # Credit Event prior is 0.95, so impact should be high
    assert impact_score >= 7.0
    assert "event_prior" in comps
    assert "sentiment_intensity" in comps
    assert "source_breadth" in comps
    assert "novelty" in comps


def test_novelty_computation():
    """Verify compute_novelty returns low score for repetitive events and 1.0 for new events."""
    embs = embed_texts([
        "Federal Reserve cuts interest rates by 50 basis points",
        "Federal Reserve cuts interest rates by 50 basis points",
        "Oil tanker attacked in Strait of Hormuz",
    ])
    # Duplicate event
    nov_dup = compute_novelty(embs[0:1], [embs[1]])
    assert nov_dup < 0.15

    # Dissimilar event
    nov_new = compute_novelty(embs[2:3], [embs[0]])
    assert nov_new > 0.60


def test_signal_storage_sqlite_and_jsonl(tmp_path):
    """Verify saving signals to SQLite and JSONL and querying by filters."""
    db_file = tmp_path / "test_signals.db"
    jsonl_file = tmp_path / "test_signals.jsonl"

    sig1 = Signal(
        event_id="evt_test_01",
        timestamp="2026-10-16T12:00:00Z",
        entities={"companies": [{"ticker": "JPM", "name": "JPMorgan Chase"}], "sectors": ["Financials"]},
        sentiment_score=-0.85,
        event_type="Credit Event",
        impact_score=8.5,
        confidence=0.92,
        divergence_flag=False,
        evidence=[{"url": "https://example.com/1", "title": "Title 1", "source": "gdelt", "snippet": "Snippet 1"}],
        model_versions={"sentiment": "finbert"},
    )
    sig2 = Signal(
        event_id="evt_test_02",
        timestamp="2026-10-15T08:00:00Z",
        entities={"companies": [], "sectors": ["Energy"]},
        sentiment_score=-0.20,
        event_type="Commodity/Supply Chain",
        impact_score=5.2,
        confidence=0.88,
        divergence_flag=True,
        evidence=[],
        model_versions={"sentiment": "finbert"},
    )

    save_signals([sig1, sig2], jsonl_path=jsonl_file, db_path=db_file)

    assert jsonl_file.exists()
    assert db_file.exists()

    # Query all
    all_sigs = query_signals(db_path=db_file)
    assert len(all_sigs) == 2

    # Filter by min_impact
    high_impact = query_signals(min_impact=8.0, db_path=db_file)
    assert len(high_impact) == 1
    assert high_impact[0].event_id == "evt_test_01"

    # Filter by event_type
    credit_sigs = query_signals(event_type="Credit Event", db_path=db_file)
    assert len(credit_sigs) == 1
    assert credit_sigs[0].event_id == "evt_test_01"

    # Filter by entity
    jpm_sigs = query_signals(entity="JPM", db_path=db_file)
    assert len(jpm_sigs) == 1
    assert jpm_sigs[0].event_id == "evt_test_01"

    # Fetch by ID
    single = get_signal_by_id("evt_test_01", db_path=db_file)
    assert single is not None
    assert single.event_id == "evt_test_01"
