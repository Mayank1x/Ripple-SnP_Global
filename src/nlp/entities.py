"""Entity extraction using alias dictionary with spaCy ORG/GPE fallback.

Matches entities across four categories:
  - Companies (with ticker, sector, country metadata)
  - Sectors
  - Countries
  - Commodities

Reference dictionary lives at data/reference/entities.csv.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Optional spaCy integration with graceful fallback
_SPACY_NLP: Any = None
_SPACY_TRIED = False


def _get_spacy_model():
    """Lazily load spaCy model if installed, else return None."""
    global _SPACY_NLP, _SPACY_TRIED
    if not _SPACY_TRIED:
        _SPACY_TRIED = True
        try:
            import spacy

            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = None
        except ImportError:
            _SPACY_NLP = None
    return _SPACY_NLP


@dataclass
class ExtractedEntities:
    """Entities identified in text."""

    companies: list[dict[str, str]] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    commodities: list[str] = field(default_factory=list)
    unmapped_orgs: list[str] = field(default_factory=list)
    unmapped_gpes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "companies": self.companies,
            "sectors": self.sectors,
            "countries": self.countries,
            "commodities": self.commodities,
            "unmapped_orgs": self.unmapped_orgs,
            "unmapped_gpes": self.unmapped_gpes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedEntities:
        return cls(
            companies=data.get("companies", []),
            sectors=data.get("sectors", []),
            countries=data.get("countries", []),
            commodities=data.get("commodities", []),
            unmapped_orgs=data.get("unmapped_orgs", []),
            unmapped_gpes=data.get("unmapped_gpes", []),
        )


class EntityCatalog:
    """In-memory dictionary of reference entities loaded from CSV."""

    def __init__(self, csv_path: str | Path = "data/reference/entities.csv") -> None:
        self.csv_path = Path(csv_path)
        self.entries: list[dict[str, Any]] = []
        self._patterns: list[tuple[re.Pattern, dict[str, Any]]] = []
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            return

        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entity_type = row["entity_type"].strip()
                entity_id = row["id"].strip()
                name = row["name"].strip()
                sector = row.get("sector", "").strip()
                country = row.get("country", "").strip()
                aliases_raw = row.get("aliases", "").strip()

                aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
                if name not in aliases:
                    aliases.insert(0, name)

                entry = {
                    "entity_type": entity_type,
                    "id": entity_id,
                    "name": name,
                    "sector": sector,
                    "country": country,
                    "aliases": aliases,
                }
                self.entries.append(entry)

                # Precompile regex for each alias with word boundary
                for alias in aliases:
                    # Case-sensitive for short 2-3 char acronyms (e.g. US, UK, BP, GE, GM, VW),
                    # case-insensitive for longer phrases.
                    is_short = len(alias) <= 3 and alias.isupper()
                    flags = 0 if is_short else re.IGNORECASE
                    pat = re.compile(rf"\b{re.escape(alias)}\b", flags)
                    self._patterns.append((pat, entry))

    def extract(self, text: str) -> ExtractedEntities:
        """Extract entities from input text using dictionary and spaCy fallback."""
        matched_companies: dict[str, dict[str, str]] = {}
        matched_sectors: set[str] = set()
        matched_countries: set[str] = set()
        matched_commodities: set[str] = set()

        # 1. Alias dictionary matching
        for pat, entry in self._patterns:
            if pat.search(text):
                etype = entry["entity_type"]
                eid = entry["id"]

                if etype == "company":
                    if eid not in matched_companies:
                        matched_companies[eid] = {
                            "ticker": eid,
                            "name": entry["name"],
                            "sector": entry["sector"],
                            "country": entry["country"],
                        }
                    # Automatically tag company's sector and country
                    if entry["sector"]:
                        matched_sectors.add(entry["sector"])
                    if entry["country"]:
                        matched_countries.add(entry["country"])

                elif etype == "sector":
                    matched_sectors.add(entry["name"])

                elif etype == "country":
                    matched_countries.add(entry["name"])

                elif etype == "commodity":
                    matched_commodities.add(entry["name"])

        # 2. spaCy NER Fallback
        unmapped_orgs: set[str] = set()
        unmapped_gpes: set[str] = set()

        nlp = _get_spacy_model()
        if nlp is not None:
            doc = nlp(text[:1000])  # limit length for speed
            for ent in doc.ents:
                clean_text = ent.text.strip()
                if not clean_text or len(clean_text) < 2:
                    continue

                if ent.label_ == "ORG":
                    if not any(
                        c["name"].lower() == clean_text.lower()
                        or c["ticker"].lower() == clean_text.lower()
                        for c in matched_companies.values()
                    ):
                        unmapped_orgs.add(clean_text)

                elif ent.label_ in ("GPE", "LOC"):
                    if not any(c.lower() == clean_text.lower() for c in matched_countries):
                        unmapped_gpes.add(clean_text)
        else:
            # Fast regex fallback for unknown organizations when spaCy is not installed
            cap_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", text)
            stopwords = {"The", "This", "That", "When", "What", "After", "Before", "While", "Under", "Over", "And", "For", "With"}
            org_suffixes = ("Corp", "Inc", "Bank", "Ltd", "Group", "LLC", "Fund", "AG", "SA", "Co", "Plc", "Holdings", "Capital", "Partners")
            for phrase in cap_phrases:
                phrase = phrase.strip()
                if len(phrase) <= 2 or phrase in stopwords:
                    continue
                already_known = (
                    any(c["name"].lower() == phrase.lower() or c["ticker"].lower() == phrase.lower() for c in matched_companies.values())
                    or any(c.lower() == phrase.lower() for c in matched_countries)
                    or any(s.lower() == phrase.lower() for s in matched_sectors)
                )
                if not already_known:
                    if any(phrase.endswith(s) for s in org_suffixes) or len(phrase.split()) >= 2:
                        unmapped_orgs.add(phrase)

        return ExtractedEntities(
            companies=list(matched_companies.values()),
            sectors=sorted(matched_sectors),
            countries=sorted(matched_countries),
            commodities=sorted(matched_commodities),
            unmapped_orgs=sorted(unmapped_orgs)[:10],
            unmapped_gpes=sorted(unmapped_gpes)[:10],
        )


_DEFAULT_CATALOG: EntityCatalog | None = None


def extract_entities(text: str) -> ExtractedEntities:
    """Convenience helper using the shared default entity catalog."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = EntityCatalog()
    return _DEFAULT_CATALOG.extract(text)
