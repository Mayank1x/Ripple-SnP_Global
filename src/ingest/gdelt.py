"""Ingestor for the GDELT DOC 2.0 API (free, no key required).

API reference (verified Oct 2026):
  Base URL : https://api.gdeltproject.org/api/v2/doc/doc
  Key params:
    query           – search keywords / phrases, supports OR, quotes, operators
    mode            – "artlist" returns a flat article list
    format          – "json" for machine-readable output
    maxrecords      – up to 250 per call
    startdatetime   – YYYYMMDDHHMMSS  (UTC)
    enddatetime     – YYYYMMDDHHMMSS  (UTC)
    sourcelang      – e.g. "english"
  Response shape (JSON):
    { "articles": [ { url, url_mobile, title, seendate, socialimage,
                       domain, language, sourcecountry }, … ] }
    seendate format: "YYYYMMDDTHHMMSSZ"
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from typing import Any

from src.ingest.helpers import (
    fetch_json,
    load_config,
    read_cache,
    write_cache,
)
from src.ingest.schema import Document


def _date_to_gdelt_ts(d: date) -> str:
    """Convert a date to the GDELT timestamp format YYYYMMDDHHMMSS."""
    return d.strftime("%Y%m%d%H%M%S")


def _parse_seendate(raw: str) -> str:
    """Convert '20260731T041500Z' to ISO-8601."""
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
        return dt.isoformat() + "Z"
    except ValueError:
        return raw


def fetch_gdelt(
    query: str,
    start: date,
    end: date,
    *,
    use_cache: bool = True,
) -> list[Document]:
    """Fetch articles from GDELT DOC 2.0 for each day in [start, end).

    One API call per day keeps responses small and cache granular.
    Returns a flat list of Document objects.
    """
    cfg = load_config()["ingest"]["gdelt"]
    base_url: str = cfg["base_url"]
    max_records: int = cfg["max_records"]
    source_lang: str = cfg["source_lang"]

    documents: list[Document] = []
    current = start

    safe_query = re.sub(r"[^a-zA-Z0-9_-]", "_", query[:40])
    cache_file = f"gdelt_{safe_query}.json"

    while current < end:
        next_day = current + timedelta(days=1)
        date_str = current.isoformat()

        # Try cache first (replay mode)
        cached: Any = None
        if use_cache:
            cached = read_cache("gdelt", date_str, cache_file)

        if cached is not None:
            raw_articles = cached
        else:
            params: dict[str, Any] = {
                "query": f"{query} sourcelang:{source_lang}",
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records,
                "startdatetime": _date_to_gdelt_ts(current),
                "enddatetime": _date_to_gdelt_ts(next_day),
            }

            try:
                data = fetch_json(base_url, params=params)
                raw_articles = data.get("articles", []) if isinstance(data, dict) else []
                write_cache("gdelt", date_str, cache_file, raw_articles)
                time.sleep(6)  # GDELT requires at least 5s between requests
            except Exception as exc:
                print(f"  [gdelt] Warning for {date_str}: {exc}")
                raw_articles = []
                time.sleep(6)

        # Convert to Documents
        for art in raw_articles:
            doc = Document(
                source="gdelt",
                published_at=_parse_seendate(art.get("seendate", "")),
                title=art.get("title", ""),
                text=art.get("title", ""),  # GDELT artlist has no body
                url=art.get("url", ""),
                language=art.get("language", "English").lower(),
                raw_meta={
                    "domain": art.get("domain", ""),
                    "sourcecountry": art.get("sourcecountry", ""),
                    "socialimage": art.get("socialimage", ""),
                },
            )
            documents.append(doc)

        current = next_day

    return documents
