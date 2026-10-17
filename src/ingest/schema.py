"""Common Document schema used across all ingestors.

Every ingestor converts its raw data into a list of Document dicts.
This keeps the rest of the pipeline source-agnostic.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """One piece of text from any source (article, tweet, etc.).

    Fields
    ------
    id          : SHA-256 of (source + url). Deterministic so dedup is easy.
    source      : Where it came from: "gdelt", "newsapi", "rss", "tweets".
    published_at: ISO-8601 timestamp string.
    title       : Headline or first sentence.
    text        : Full body text (may be truncated by the API).
    url         : Original URL. Empty string for tweets without one.
    language    : Two-letter or full language name as provided by the source.
    raw_meta    : Any extra fields the source returned, stored as-is.
    """

    source: str
    published_at: str
    title: str
    text: str
    url: str
    language: str = "english"
    raw_meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""  # filled by __post_init__

    def __post_init__(self) -> None:
        """Generate a deterministic id from source + url."""
        if not self.id:
            key = f"{self.source}:{self.url or self.title}"
            self.id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (safe for JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Reconstruct a Document from a dict (e.g. loaded from JSON)."""
        return cls(**data)
