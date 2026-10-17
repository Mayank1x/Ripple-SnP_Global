"""News ingestor: tries NewsAPI.org first, falls back to RSS feeds.

NewsAPI /v2/everything (verified Oct 2026):
  Params : q, from, to, language, sortBy, pageSize, apiKey
  Response: { status, totalResults, articles: [
    { source: {id,name}, author, title, description,
      url, urlToImage, publishedAt, content }, … ] }
  Free tier: 100 requests/day, articles up to 1 month old.

RSS fallback: feedparser reads any Atom/RSS feed. We use business-news
feeds listed in config.yaml.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import feedparser

from src.ingest.helpers import (
    fetch_json,
    get_secret,
    load_config,
    read_cache,
    write_cache,
)
from src.ingest.schema import Document

# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------


def _fetch_newsapi(
    query: str,
    start: date,
    end: date,
    *,
    use_cache: bool = True,
) -> list[Document]:
    """Call NewsAPI /v2/everything for each day in [start, end)."""
    api_key = get_secret("NEWSAPI_KEY")
    if not api_key:
        return []  # caller will fall back to RSS

    cfg = load_config()["ingest"]["newsapi"]
    documents: list[Document] = []
    current = start

    safe_query = re.sub(r"[^a-zA-Z0-9_-]", "_", query[:40])
    cache_file = f"newsapi_{safe_query}.json"

    while current < end:
        next_day = current + timedelta(days=1)
        date_str = current.isoformat()

        cached: Any = None
        if use_cache:
            cached = read_cache("newsapi", date_str, cache_file)

        if cached is not None:
            raw_articles = cached
        else:
            params: dict[str, Any] = {
                "q": query,
                "from": date_str,
                "to": next_day.isoformat(),
                "language": cfg["language"],
                "sortBy": cfg["sort_by"],
                "pageSize": cfg["page_size"],
                "apiKey": api_key,
            }
            try:
                data = fetch_json(cfg["base_url"], params=params)
            except Exception as exc:
                print(f"  [newsapi] Error for {date_str}: {exc}")
                current = next_day
                continue

            raw_articles = data.get("articles", []) if isinstance(data, dict) else []
            write_cache("newsapi", date_str, cache_file, raw_articles)

        for art in raw_articles:
            doc = Document(
                source="newsapi",
                published_at=art.get("publishedAt", ""),
                title=art.get("title", ""),
                text=art.get("description", "") or art.get("content", ""),
                url=art.get("url", ""),
                language="english",
                raw_meta={
                    "author": art.get("author", ""),
                    "source_name": (art.get("source") or {}).get("name", ""),
                    "urlToImage": art.get("urlToImage", ""),
                },
            )
            documents.append(doc)

        current = next_day

    return documents


# ---------------------------------------------------------------------------
# RSS fallback
# ---------------------------------------------------------------------------


def _fetch_rss(*, use_cache: bool = True) -> list[Document]:
    """Parse RSS feeds listed in config.yaml. One cache file per feed."""
    cfg = load_config()["ingest"]["rss"]
    today_str = date.today().isoformat()
    documents: list[Document] = []

    for feed_info in cfg["feeds"]:
        name: str = feed_info["name"]
        url: str = feed_info["url"]
        safe_name = name.replace(" ", "_").lower()
        cache_file = f"rss_{safe_name}.json"

        cached: Any = None
        if use_cache:
            cached = read_cache("rss", today_str, cache_file)

        if cached is not None:
            raw_entries = cached
        else:
            try:
                parsed = feedparser.parse(url)
                raw_entries = [
                    {
                        "title": e.get("title", ""),
                        "link": e.get("link", ""),
                        "published": e.get("published", ""),
                        "summary": e.get("summary", ""),
                    }
                    for e in parsed.entries
                ]
            except Exception as exc:
                print(f"  [rss] Error fetching {name}: {exc}")
                continue

            write_cache("rss", today_str, cache_file, raw_entries)

        for entry in raw_entries:
            doc = Document(
                source="rss",
                published_at=entry.get("published", ""),
                title=entry.get("title", ""),
                text=entry.get("summary", ""),
                url=entry.get("link", ""),
                language="english",
                raw_meta={"feed_name": name},
            )
            documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_news(
    query: str,
    start: date,
    end: date,
    *,
    use_cache: bool = True,
) -> list[Document]:
    """Try NewsAPI first. If no key is configured, use RSS feeds instead."""
    docs = _fetch_newsapi(query, start, end, use_cache=use_cache)
    if docs:
        return docs

    print("  [news] No NEWSAPI_KEY set - falling back to RSS feeds.")
    return _fetch_rss(use_cache=use_cache)
