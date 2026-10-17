"""Loader for a Kaggle financial-tweets CSV.

Expected columns (case-insensitive): at least two of
  - "text" or "tweet" or "content"        → body text
  - "date" or "created_at" or "timestamp" → publish time
Optional:
  - "user" or "username"
  - "url" or "link"

Put one or more .csv files in data/raw/tweets/ and this module reads
them all, normalising each row into a Document.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

from src.ingest.helpers import load_config
from src.ingest.schema import Document


def _find_column(headers: list[str], candidates: list[str]) -> str | None:
    """Return the first header that matches any candidate (case-insensitive)."""
    lower_headers = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower_headers:
            return lower_headers[c.lower()]
    return None


def load_tweets(
    start: date | None = None,
    end: date | None = None,
) -> list[Document]:
    """Read all .csv files in the configured tweets directory.

    Optionally filter rows whose date falls in [start, end).
    If the CSV has no parseable date column, all rows are included.
    """
    cfg = load_config()["ingest"]["tweets"]
    tweets_dir = Path(cfg["directory"])

    if not tweets_dir.exists():
        print(f"  [tweets] Directory not found: {tweets_dir}")
        return []

    csv_files = list(tweets_dir.glob("*.csv"))
    if not csv_files:
        print(f"  [tweets] No .csv files in {tweets_dir}")
        return []

    documents: list[Document] = []

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue

            headers = list(reader.fieldnames)
            text_col = _find_column(headers, ["text", "tweet", "content"])
            date_col = _find_column(headers, ["date", "created_at", "timestamp"])
            user_col = _find_column(headers, ["user", "username", "screen_name"])
            url_col = _find_column(headers, ["url", "link"])

            if text_col is None:
                print(f"  [tweets] Skipping {csv_path.name}: no text column found")
                continue

            for row in reader:
                text = (row.get(text_col) or "").strip()
                if not text:
                    continue

                published = (row.get(date_col) or "") if date_col else ""
                if published and (start is not None or end is not None):
                    try:
                        row_date = date.fromisoformat(published[:10])
                        if start is not None and row_date < start:
                            continue
                        if end is not None and row_date >= end:
                            continue
                    except (ValueError, TypeError):
                        pass

                doc = Document(
                    source="tweets",
                    published_at=published,
                    title=text[:120],  # tweets have no separate title
                    text=text,
                    url=(row.get(url_col) or "") if url_col else "",
                    language="english",  # assume English for Kaggle dataset
                    raw_meta={
                        "user": (row.get(user_col) or "") if user_col else "",
                        "csv_file": csv_path.name,
                    },
                )
                documents.append(doc)

    return documents
