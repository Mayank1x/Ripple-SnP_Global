"""Deduplication and language filtering for ingested Documents.

Rules:
  1. Drop rows whose URL has already been seen (exact match).
  2. Drop rows whose title has already been seen (exact match, lowered).
  3. Keep only documents whose language field contains the configured
     keep_language string (default "english").
"""

from __future__ import annotations

from src.ingest.helpers import load_config
from src.ingest.schema import Document


def deduplicate(docs: list[Document]) -> list[Document]:
    """Remove exact-duplicate URLs and titles, keep English only.

    Returns a new list; does not mutate the input.
    """
    cfg = load_config()["ingest"]["dedup"]
    keep_lang = cfg.get("keep_language", "english").lower()

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    result: list[Document] = []

    for doc in docs:
        # Language filter
        if keep_lang and keep_lang not in doc.language.lower():
            continue

        # URL dedup
        if doc.url:
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)

        # Title dedup
        title_key = doc.title.strip().lower()
        if title_key:
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

        result.append(doc)

    return result
