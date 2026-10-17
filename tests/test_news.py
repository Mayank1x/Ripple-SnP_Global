"""Tests for the news ingestor (NewsAPI + RSS fallback) using fixtures."""

import json
from datetime import date
from pathlib import Path

from src.ingest.news import fetch_news

# Minimal NewsAPI response fixture matching real API shape.
NEWSAPI_FIXTURE = [
    {
        "source": {"id": "bbc-news", "name": "BBC News"},
        "author": "J. Smith",
        "title": "Bank of England holds rates steady",
        "description": "The BoE voted 7-2 to hold the base rate at 5.25%.",
        "url": "https://bbc.co.uk/news/business-123",
        "urlToImage": "https://bbc.co.uk/img.jpg",
        "publishedAt": "2026-10-10T08:00:00Z",
        "content": "Full article body here...",
    },
]


def test_newsapi_from_cache(tmp_path: Path, monkeypatch):
    """Feed a cached NewsAPI fixture and verify output shape."""
    cache_dir = tmp_path / "newsapi" / "2026-10-10"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "newsapi_bank_OR_default.json"
    cache_file.write_text(json.dumps(NEWSAPI_FIXTURE))

    fake_config = {
        "ingest": {
            "newsapi": {
                "base_url": "https://newsapi.org/v2/everything",
                "page_size": 100,
                "language": "en",
                "sort_by": "relevancy",
            },
            "rss": {"feeds": []},
            "cache_dir": str(tmp_path),
            "dedup": {"enabled": True, "keep_language": "english"},
        }
    }
    monkeypatch.setattr("src.ingest.news.load_config", lambda: fake_config)
    monkeypatch.setattr("src.ingest.helpers.load_config", lambda: fake_config)
    monkeypatch.setattr("src.ingest.news.get_secret", lambda _: "fake-key")

    docs = fetch_news(
        query="bank OR default",
        start=date(2026, 10, 10),
        end=date(2026, 10, 11),
        use_cache=True,
    )

    assert len(docs) == 1
    assert docs[0].source == "newsapi"
    assert docs[0].title == "Bank of England holds rates steady"
    assert docs[0].raw_meta["source_name"] == "BBC News"


def test_rss_fallback_from_cache(tmp_path: Path, monkeypatch):
    """Verify RSS cache fallback when NEWSAPI_KEY is not set."""
    today_str = date.today().isoformat()
    cache_dir = tmp_path / "rss" / today_str
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "rss_test_feed.json"
    cache_file.write_text(
        json.dumps(
            [
                {
                    "title": "Fed rate decision preview",
                    "link": "https://example.com/fed-preview",
                    "published": "2026-10-10T10:00:00Z",
                    "summary": "Market expectations for upcoming FOMC meeting.",
                }
            ]
        )
    )

    fake_config = {
        "ingest": {
            "newsapi": {"enabled": True},
            "rss": {
                "feeds": [
                    {"name": "Test Feed", "url": "https://example.com/rss.xml"}
                ]
            },
            "cache_dir": str(tmp_path),
            "dedup": {"enabled": True, "keep_language": "english"},
        }
    }
    monkeypatch.setattr("src.ingest.news.load_config", lambda: fake_config)
    monkeypatch.setattr("src.ingest.helpers.load_config", lambda: fake_config)
    monkeypatch.setattr("src.ingest.news.get_secret", lambda _: "")

    docs = fetch_news(
        query="bank OR default",
        start=date(2026, 10, 10),
        end=date(2026, 10, 11),
        use_cache=True,
    )

    assert len(docs) == 1
    assert docs[0].source == "rss"
    assert docs[0].title == "Fed rate decision preview"
    assert docs[0].url == "https://example.com/fed-preview"

