"""Tests for the GDELT ingestor using fixture data (no network calls)."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.ingest.gdelt import fetch_gdelt

# A minimal GDELT response matching the real API shape.
GDELT_FIXTURE = [
    {
        "url": "https://example.com/article-1",
        "url_mobile": "",
        "title": "Fed raises interest rates by 25 basis points",
        "seendate": "20261010T120000Z",
        "socialimage": "",
        "domain": "example.com",
        "language": "English",
        "sourcecountry": "United States",
    },
    {
        "url": "https://example.com/article-2",
        "url_mobile": "",
        "title": "Oil prices surge amid Middle East tensions",
        "seendate": "20261010T140000Z",
        "socialimage": "",
        "domain": "example.com",
        "language": "English",
        "sourcecountry": "United States",
    },
]


def test_gdelt_parses_fixture(tmp_path: Path, monkeypatch):
    """Feed a cached fixture to the GDELT ingestor and check output shape."""
    # Write fixture into the cache directory the ingestor expects
    cache_dir = tmp_path / "gdelt" / "2026-10-10"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "gdelt_bank_OR_default.json"
    cache_file.write_text(json.dumps(GDELT_FIXTURE))

    # Patch config so cache_dir points to our tmp_path
    fake_config = {
        "ingest": {
            "gdelt": {
                "base_url": "https://api.gdeltproject.org/api/v2/doc/doc",
                "max_records": 250,
                "source_lang": "english",
            },
            "cache_dir": str(tmp_path),
            "dedup": {"enabled": True, "keep_language": "english"},
        }
    }
    monkeypatch.setattr("src.ingest.gdelt.load_config", lambda: fake_config)
    monkeypatch.setattr("src.ingest.helpers.load_config", lambda: fake_config)

    docs = fetch_gdelt(
        query="bank OR default",
        start=date(2026, 10, 10),
        end=date(2026, 10, 11),
        use_cache=True,
    )

    assert len(docs) == 2
    assert docs[0].source == "gdelt"
    assert docs[0].title == "Fed raises interest rates by 25 basis points"
    assert docs[0].url == "https://example.com/article-1"
    assert docs[0].language == "english"
    assert "2026-10-10" in docs[0].published_at
