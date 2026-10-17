"""Tests for the tweets CSV loader using a temporary CSV fixture."""

import csv
from datetime import date
from pathlib import Path

from src.ingest.tweets import load_tweets


def test_tweets_loader(tmp_path: Path, monkeypatch):
    """Write a small CSV, point the loader at it, check output."""
    tweets_dir = tmp_path / "tweets"
    tweets_dir.mkdir()

    csv_path = tweets_dir / "test_tweets.csv"
    rows = [
        {"text": "Fed hikes rates again!", "date": "2026-10-12", "user": "analyst1"},
        {"text": "Markets crash after default rumour", "date": "2026-10-13", "user": "trader99"},
        {"text": "", "date": "2026-10-13", "user": "empty"},  # should be skipped
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "date", "user"])
        writer.writeheader()
        writer.writerows(rows)

    # Patch config to point at our tmp directory
    fake_config = {
        "ingest": {
            "tweets": {"directory": str(tweets_dir)},
            "cache_dir": str(tmp_path),
            "dedup": {"enabled": True, "keep_language": "english"},
        }
    }
    monkeypatch.setattr("src.ingest.tweets.load_config", lambda: fake_config)

    docs = load_tweets()
    assert len(docs) == 2
    assert docs[0].source == "tweets"
    assert "hikes" in docs[0].text
    assert docs[1].raw_meta["csv_file"] == "test_tweets.csv"


def test_tweets_loader_missing_dir(tmp_path: Path, monkeypatch):
    """Loader should return empty list when directory doesn't exist."""
    fake_config = {
        "ingest": {
            "tweets": {"directory": str(tmp_path / "nonexistent")},
            "cache_dir": str(tmp_path),
            "dedup": {"enabled": True, "keep_language": "english"},
        }
    }
    monkeypatch.setattr("src.ingest.tweets.load_config", lambda: fake_config)

    docs = load_tweets()
    assert docs == []


def test_tweets_loader_date_filter(tmp_path: Path, monkeypatch):
    """Verify start and end date filtering works."""
    tweets_dir = tmp_path / "tweets"
    tweets_dir.mkdir()

    csv_path = tweets_dir / "dated_tweets.csv"
    rows = [
        {"text": "Early tweet", "date": "2026-10-09", "user": "u1"},
        {"text": "In range tweet 1", "date": "2026-10-12", "user": "u2"},
        {"text": "In range tweet 2", "date": "2026-10-15", "user": "u3"},
        {"text": "Late tweet", "date": "2026-10-20", "user": "u4"},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "date", "user"])
        writer.writeheader()
        writer.writerows(rows)

    fake_config = {
        "ingest": {
            "tweets": {"directory": str(tweets_dir)},
            "cache_dir": str(tmp_path),
            "dedup": {"enabled": True, "keep_language": "english"},
        }
    }
    monkeypatch.setattr("src.ingest.tweets.load_config", lambda: fake_config)

    docs = load_tweets(start=date(2026, 10, 10), end=date(2026, 10, 17))
    assert len(docs) == 2
    assert docs[0].text == "In range tweet 1"
    assert docs[1].text == "In range tweet 2"

