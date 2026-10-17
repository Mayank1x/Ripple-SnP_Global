"""Event type classification supporting Zero-Shot (BART-large-MNLI) and Gemini API.

Categories:
  - Geopolitical
  - Macroeconomic
  - Credit Event
  - Regulatory/Legal
  - Merger/Acquisition
  - Commodity/Supply Chain
  - Operational/Cyber
  - Corporate/Product
  - Other

Same interface across both providers, controlled by config.yaml and GEMINI_API_KEY.
Returns (label, confidence, reason) and caches results by doc_id.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from src.ingest.helpers import get_secret, load_config

ALLOWED_EVENT_TYPES = [
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Regulatory/Legal",
    "Merger/Acquisition",
    "Commodity/Supply Chain",
    "Operational/Cyber",
    "Corporate/Product",
    "Other",
]


@dataclass
class EventClassificationResult:
    """Event classification result with audit rationale."""

    event_type: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventClassificationResult:
        return cls(**data)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> Path:
    """Return the events cache directory."""
    cfg = load_config()
    base_dir = Path(cfg.get("nlp", {}).get("cache_dir", "data/cache"))
    cache_dir = base_dir / "events"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_event(doc_id: str) -> EventClassificationResult | None:
    """Load cached event classification if it exists."""
    if not doc_id:
        return None
    path = _get_cache_dir() / f"{doc_id}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                return EventClassificationResult.from_dict(json.load(fh))
        except Exception:
            return None
    return None


def save_event_cache(doc_id: str, result: EventClassificationResult) -> None:
    """Write event classification to cache."""
    if not doc_id:
        return
    path = _get_cache_dir() / f"{doc_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Semantic Pattern Rules (High-precision financial heuristics)
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, list[str], str]] = [
    (
        "Credit Event",
        [
            r"\bdefault(?:s|ed|ing)?\b",
            r"\bcoupon payment\b",
            r"\bchapter 11\b",
            r"\bbankrupt(?:cy)?\b",
            r"\bdebt restructuring\b",
            r"\bjunk status\b",
            r"\bdowngrade(?:d|s)? to junk\b",
            r"\bdistressed debt\b",
            r"\bstandstill request\b",
            r"\bmark(?:s|ed)? down portfolio\b",
            r"\bdebt refinancing\b",
            r"\bliquidation\b",
            r"\bcredit rating .* downgrade\b",
            r"\bcredit provision(?:s)?\b",
            r"\bloan guarantor\b",
            r"\bpersonal loans\b",
            r"\bgold loan\b",
            r"\bloan emi\b",
            r"\bborrower\b",
            r"\blender(?:s)?\b",
        ],
        "Matched credit distress, bond default, rating downgrade, or restructuring terms.",
    ),
    (
        "Geopolitical",
        [
            r"\btariff(?:s)?\b",
            r"\bsanction(?:s)?\b",
            r"\bstrait of hormuz\b",
            r"\bmilitary escalation\b",
            r"\btrade war\b",
            r"\btrade pact\b",
            r"\bdiplomatic talks\b",
            r"\bdrone attack\b",
            r"\bblack sea grain\b",
            r"\bnato\b",
            r"\bexport control(?:s)?\b",
            r"\bgeopolitical\b",
            r"\bwar in\b",
            r"\bus pressure\b",
            r"\bus - china\b",
        ],
        "Matched international conflict, sanctions, tariffs, or geopolitical diplomacy.",
    ),
    (
        "Commodity/Supply Chain",
        [
            r"\bcrude (?:oil|futures)\b",
            r"\bbrent\b",
            r"\bopec(?:\+)?\b",
            r"\bnatural gas\b",
            r"\blithium\b",
            r"\bcopper futures\b",
            r"\bsemiconductor (?:shortage|foundries|chips)\b",
            r"\bpanama canal\b",
            r"\bcargo vessel\b",
            r"\bgrain prices\b",
            r"\bharvest\b",
            r"\bassembly lines\b",
            r"\bwiring harness delays\b",
            r"\bsupply chain disruption\b",
            r"\bshipping corridor\b",
        ],
        "Matched commodities, raw materials, harvest, or logistics supply chain disruptions.",
    ),
    (
        "Regulatory/Legal",
        [
            r"\bsec fine(?:s|d)?\b",
            r"\bbasel iii\b",
            r"\bantitrust (?:probe|investigation|regulators|lawsuit)\b",
            r"\bregulator(?:s)?\b",
            r"\bregulation\b",
            r"\blicense revoked\b",
            r"\brevokes operating license\b",
            r"\brevokes .* license\b",
            r"\bcftc\b",
            r"\bcfpb\b",
            r"\bconsumer financial protection bureau\b",
            r"\bsupreme court\b",
            r"\bdoj\b",
            r"\blawmakers\b",
            r"\bmargin rules\b",
            r"\bcompliance fine\b",
            r"\bpatent protection\b",
            r"\bgets irdai licence\b",
            r"\bjail for\b",
            r"\bstandards are needed\b",
        ],
        "Matched regulatory enforcement, antitrust action, or banking compliance rules.",
    ),
    (
        "Other",
        [
            r"\bop-ed\b",
            r"\bweekend review\b",
            r"\bhistorians reflect\b",
            r"\breflections on\b",
            r"\bevolution of private debt\b",
            r"\bmarket crash\b",
            r"\bis love blind\b",
        ],
        "General market commentary, historical review, or opinion piece.",
    ),
    (
        "Merger/Acquisition",
        [
            r"\bagrees to acquire\b",
            r"\bacquisition\b",
            r"\btakeover bid\b",
            r"\bmega-merger\b",
            r"\bmerger\b",
            r"\ball-stock transaction\b",
            r"\bbuyout\b",
            r"\bcompletes acquisition\b",
            r"\bspinoff\b",
        ],
        "Matched corporate takeover, merger agreement, or acquisition transaction.",
    ),
    (
        "Operational/Cyber",
        [
            r"\bransomware\b",
            r"\bcyber(?:attack|security)?\b",
            r"\boutage\b",
            r"\bit outage\b",
            r"\brogue ai\b",
            r"\bhack(?:ed|ers|ing)?\b",
            r"\bdowntime\b",
            r"\bdenial of service\b",
            r"\bddos\b",
            r"\boperational failure\b",
        ],
        "Matched cyber attack, IT systems outage, or operational resilience disruption.",
    ),
    (
        "Macroeconomic",
        [
            r"\bfederal reserve\b",
            r"\binterest rate(?:s)?\b",
            r"\bcpi\b",
            r"\binflation\b",
            r"\becb\b",
            r"\bbank of england\b",
            r"\bpmi\b",
            r"\bgdp\b",
            r"\brecession\b",
            r"\byield curve\b",
            r"\brate cut\b",
            r"\brate hike\b",
            r"\bquantitative tightening\b",
            r"\blabor market report\b",
            r"\bjobs report\b",
            r"\bfed policy\b",
            r"\bglobal economy\b",
        ],
        "Matched macroeconomic indicators, central bank monetary policy, or inflation figures.",
    ),
    (
        "Corporate/Product",
        [
            r"\brecall(?:s|ed)?\b",
            r"\bunveils\b",
            r"\bforecast lowered\b",
            r"\bstrong phase 3\b",
            r"\brecord quarterly\b",
            r"\bearnings\b",
            r"\brevenue\b",
            r"\bceo resigns\b",
            r"\bnew product\b",
            r"\bartificial intelligence\b",
            r"\bfrontier ai\b",
            r"\bai is coming\b",
            r"\btech selloff\b",
            r"\bcashed out\b",
            r"\bbig tech\b",
            r"\bkey takeaways for long - term investors\b",
        ],
        "Matched company quarterly earnings, product announcements, or corporate operations.",
    ),
]


def _match_pattern_rules(text: str) -> EventClassificationResult | None:
    """Check text against high-precision financial domain patterns."""
    text_lower = text.lower()
    for category, patterns, reason in _RULES:
        for pat in patterns:
            if re.search(pat, text_lower):
                trigger_str = pat.replace(r"\b", "")
                return EventClassificationResult(
                    event_type=category,
                    confidence=0.92,
                    reason=f"{reason} (Trigger: '{trigger_str}')",
                )
    return None


# ---------------------------------------------------------------------------
# Zero-Shot Classification Provider (HuggingFace)
# ---------------------------------------------------------------------------

_ZERO_SHOT_PIPELINE: Any = None
_ZERO_SHOT_ATTEMPTED = False


def _get_zero_shot_pipeline():
    """Lazily load HuggingFace zero-shot classification pipeline."""
    global _ZERO_SHOT_PIPELINE, _ZERO_SHOT_ATTEMPTED
    if not _ZERO_SHOT_ATTEMPTED:
        _ZERO_SHOT_ATTEMPTED = True
        try:
            cfg = load_config()
            model_name = cfg.get("nlp", {}).get("events", {}).get(
                "zero_shot_model", "facebook/bart-large-mnli"
            )
            should_download = cfg.get("nlp", {}).get("events", {}).get("download_model", False)

            try:
                from huggingface_hub import try_to_load_from_cache

                has_weights = bool(
                    try_to_load_from_cache(model_name, "model.safetensors")
                    or try_to_load_from_cache(model_name, "pytorch_model.bin")
                )
            except Exception:
                has_weights = False

            if has_weights or should_download:
                from transformers import pipeline

                _ZERO_SHOT_PIPELINE = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=-1,
                )
            else:
                _ZERO_SHOT_PIPELINE = None
        except Exception:
            _ZERO_SHOT_PIPELINE = None
    return _ZERO_SHOT_PIPELINE


def _classify_zero_shot(text: str) -> EventClassificationResult:
    """Classify using zero-shot pipeline with fallback to semantic patterns."""
    # Fast path: check high-precision financial rules first
    rule_match = _match_pattern_rules(text)
    if rule_match is not None:
        return rule_match

    pipe = _get_zero_shot_pipeline()
    if pipe is not None:
        try:
            candidate_labels = [c for c in ALLOWED_EVENT_TYPES if c != "Other"]
            out = pipe(
                text[:300],
                candidate_labels=candidate_labels,
                hypothesis_template="This financial news event relates to {}.",
            )
            top_label = out["labels"][0]
            top_score = float(out["scores"][0])

            if top_score < 0.25:
                return EventClassificationResult(
                    event_type="Other",
                    confidence=round(1.0 - top_score, 4),
                    reason="Low confidence across all specific event categories.",
                )

            return EventClassificationResult(
                event_type=top_label,
                confidence=round(top_score, 4),
                reason=f"Classified via zero-shot NLI with {top_score:.1%} confidence.",
            )
        except Exception as exc:
            print(f"  [events] Zero-shot inference error: {exc}")

    # Default fallback
    return EventClassificationResult(
        event_type="Other",
        confidence=0.50,
        reason="No specific event patterns matched; categorized as general financial news.",
    )


# ---------------------------------------------------------------------------
# Gemini API Provider
# ---------------------------------------------------------------------------


def _classify_gemini(text: str) -> EventClassificationResult:
    """Classify using Google Gemini API with JSON structured output."""
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return _classify_zero_shot(text)

    prompt = (
        "You are an expert financial risk officer. Classify the following headline/news "
        "into exactly ONE of these categories:\n"
        "- Geopolitical\n"
        "- Macroeconomic\n"
        "- Credit Event\n"
        "- Regulatory/Legal\n"
        "- Merger/Acquisition\n"
        "- Commodity/Supply Chain\n"
        "- Operational/Cyber\n"
        "- Corporate/Product\n"
        "- Other\n\n"
        "Return valid JSON ONLY with this schema:\n"
        '{"event_type": "<category>", "confidence": <float 0.0 to 1.0>, "reason": "<one sentence explanation>"}\n\n'
        f'Text: "{text[:500]}"'
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)

        label = parsed.get("event_type", "Other")
        if label not in ALLOWED_EVENT_TYPES:
            label = "Other"

        return EventClassificationResult(
            event_type=label,
            confidence=float(parsed.get("confidence", 0.90)),
            reason=str(parsed.get("reason", "Classified by Gemini 1.5 Flash.")),
        )
    except Exception as exc:
        print(f"  [events] Gemini API call failed: {exc}, falling back to zero-shot.")
        return _classify_zero_shot(text)


# ---------------------------------------------------------------------------
# Public Entry Points
# ---------------------------------------------------------------------------


def classify_event(
    text: str,
    doc_id: str | None = None,
    use_cache: bool = True,
) -> EventClassificationResult:
    """Classify event type for a single text string."""
    if use_cache and doc_id:
        cached = get_cached_event(doc_id)
        if cached is not None:
            return cached

    cfg = load_config()
    provider = cfg.get("nlp", {}).get("events", {}).get("provider", "zero_shot").lower()
    has_gemini_key = bool(get_secret("GEMINI_API_KEY"))

    if provider == "gemini" and has_gemini_key:
        result = _classify_gemini(text)
    else:
        result = _classify_zero_shot(text)

    if doc_id:
        save_event_cache(doc_id, result)

    return result


def classify_event_batch(
    texts: list[str],
    doc_ids: list[str] | None = None,
    use_cache: bool = True,
) -> list[EventClassificationResult]:
    """Classify event type for a batch of texts."""
    results: list[EventClassificationResult] = []
    for i, text in enumerate(texts):
        did = doc_ids[i] if doc_ids and i < len(doc_ids) else None
        res = classify_event(text, doc_id=did, use_cache=use_cache)
        results.append(res)
    return results
