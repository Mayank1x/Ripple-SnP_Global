"""Tests for the Document schema."""

from src.ingest.schema import Document


def test_document_id_is_deterministic():
    """Same source+url should produce the same id every time."""
    a = Document(source="test", published_at="2026-01-01", title="T", text="body", url="http://example.com")
    b = Document(source="test", published_at="2026-01-01", title="T", text="body", url="http://example.com")
    assert a.id == b.id
    assert len(a.id) == 16


def test_document_roundtrip():
    """to_dict → from_dict should give back an equal Document."""
    original = Document(
        source="gdelt",
        published_at="2026-10-10T00:00:00Z",
        title="Test headline",
        text="Some body text",
        url="https://example.com/article",
        language="english",
        raw_meta={"domain": "example.com"},
    )
    rebuilt = Document.from_dict(original.to_dict())
    assert rebuilt.id == original.id
    assert rebuilt.title == original.title
    assert rebuilt.raw_meta == original.raw_meta


def test_document_id_differs_for_different_urls():
    """Documents with different URLs must get different ids."""
    a = Document(source="test", published_at="", title="T", text="", url="http://a.com")
    b = Document(source="test", published_at="", title="T", text="", url="http://b.com")
    assert a.id != b.id
