"""Tests for the deduplication and language filtering module."""

from src.ingest.dedup import deduplicate
from src.ingest.schema import Document


def _make_doc(**kwargs) -> Document:
    """Shorthand to build a Document with defaults."""
    defaults = {
        "source": "test",
        "published_at": "2026-01-01",
        "title": "Headline",
        "text": "Body",
        "url": "http://example.com",
        "language": "english",
    }
    defaults.update(kwargs)
    return Document(**defaults)


def test_dedup_removes_duplicate_urls():
    """Two documents with the same URL should collapse to one."""
    docs = [
        _make_doc(url="http://a.com", title="A"),
        _make_doc(url="http://a.com", title="B"),  # same URL, different title
    ]
    result = deduplicate(docs)
    assert len(result) == 1
    assert result[0].title == "A"  # first one wins


def test_dedup_removes_duplicate_titles():
    """Two documents with the same title but different URLs collapse."""
    docs = [
        _make_doc(url="http://a.com", title="Same Headline"),
        _make_doc(url="http://b.com", title="Same Headline"),
    ]
    result = deduplicate(docs)
    assert len(result) == 1


def test_dedup_filters_non_english():
    """Non-English documents should be dropped."""
    docs = [
        _make_doc(language="english", title="A", url="http://a.com"),
        _make_doc(language="spanish", title="B", url="http://b.com"),
        _make_doc(language="English", title="C", url="http://c.com"),
    ]
    result = deduplicate(docs)
    assert len(result) == 2
    assert all("english" in d.language.lower() for d in result)


def test_dedup_keeps_unique_docs():
    """Unique documents should all survive."""
    docs = [
        _make_doc(url="http://a.com", title="A"),
        _make_doc(url="http://b.com", title="B"),
        _make_doc(url="http://c.com", title="C"),
    ]
    result = deduplicate(docs)
    assert len(result) == 3
