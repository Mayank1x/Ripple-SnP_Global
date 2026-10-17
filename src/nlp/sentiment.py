"""Sentiment analysis using ProsusAI/finbert with CPU batching and disk caching.

Calculates:
  score = P(positive) - P(negative)  [-1.0 to +1.0]
Also returns individual probabilities for positive, negative, and neutral classes.
Results are cached by document ID under data/cache/sentiment/<doc_id>.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ingest.helpers import load_config

# Global lazy singletons for model & tokenizer
_FINBERT_TOKENIZER: Any = None
_FINBERT_MODEL: Any = None
_MODEL_LOAD_ATTEMPTED = False


@dataclass
class SentimentResult:
    """Sentiment output for a document."""

    score: float  # P(pos) - P(neg), in [-1.0, 1.0]
    label: str  # "positive", "negative", "neutral"
    probs: dict[str, float]  # {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
    confidence: float  # max probability

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentimentResult:
        return cls(**data)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> Path:
    """Return the sentiment cache directory."""
    cfg = load_config()
    base_dir = Path(cfg.get("nlp", {}).get("cache_dir", "data/cache"))
    cache_dir = base_dir / "sentiment"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_sentiment(doc_id: str) -> SentimentResult | None:
    """Load cached sentiment result if it exists."""
    if not doc_id:
        return None
    path = _get_cache_dir() / f"{doc_id}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                return SentimentResult.from_dict(json.load(fh))
        except Exception:
            return None
    return None


def save_sentiment_cache(doc_id: str, result: SentimentResult) -> None:
    """Write sentiment result to cache."""
    if not doc_id:
        return
    path = _get_cache_dir() / f"{doc_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def _load_finbert():
    """Lazily load ProsusAI/finbert model and tokenizer onto CPU."""
    global _FINBERT_TOKENIZER, _FINBERT_MODEL, _MODEL_LOAD_ATTEMPTED
    if not _MODEL_LOAD_ATTEMPTED:
        _MODEL_LOAD_ATTEMPTED = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "ProsusAI/finbert"
            _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
            _FINBERT_MODEL.eval()
        except Exception as exc:
            print(f"  [sentiment] FinBERT not available ({exc}), using financial lexicon fallback.")
            _FINBERT_TOKENIZER = None
            _FINBERT_MODEL = None
    return _FINBERT_TOKENIZER, _FINBERT_MODEL


# Financial domain lexicon fallback for offline/resilience
_POS_WORDS = {
    "surge", "surges", "surging", "profit", "profits", "gain", "gains", "growth",
    "rebound", "rebounds", "rally", "rallies", "upgrade", "upgrades", "beat",
    "beats", "strong", "bullish", "record", "soar", "soars", "boost", "boosts",
    "acquire", "acquires", "acquisition", "refinancing", "recovery", "settlement"
}
_NEG_WORDS = {
    "default", "defaults", "defaulting", "downgrade", "downgrades", "loss",
    "losses", "crash", "crashes", "slump", "slumps", "drop", "drops", "plunge",
    "plunges", "fall", "falls", "crisis", "recession", "bankruptcy", "bankrupt",
    "restructuring", "sanction", "sanctions", "tariff", "tariffs", "inflation",
    "probe", "investigation", "fine", "fined", "fraud", "outage", "ransomware",
    "risk", "risks", "debt", "warning", "warns", "cut", "cuts", "decline", "declines"
}


def _lexicon_sentiment(text: str) -> SentimentResult:
    """Fallback financial lexicon sentiment."""
    words = [w.strip(".,;:!?\"'()[]") for w in text.lower().split()]
    pos_count = sum(1 for w in words if w in _POS_WORDS)
    neg_count = sum(1 for w in words if w in _NEG_WORDS)
    total = pos_count + neg_count

    if total == 0:
        p_pos, p_neg, p_neu = 0.1, 0.1, 0.8
    else:
        p_pos = round(pos_count / (total + 1.0), 4)
        p_neg = round(neg_count / (total + 1.0), 4)
        p_neu = round(max(0.0, 1.0 - (p_pos + p_neg)), 4)

    score = round(p_pos - p_neg, 4)
    probs = {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
    label = max(probs, key=probs.get)
    return SentimentResult(
        score=score,
        label=label,
        probs=probs,
        confidence=probs[label],
    )


def analyse_sentiment(
    text: str,
    doc_id: str | None = None,
    use_cache: bool = True,
) -> SentimentResult:
    """Analyse sentiment for a single text string."""
    if use_cache and doc_id:
        cached = get_cached_sentiment(doc_id)
        if cached is not None:
            return cached

    results = analyse_sentiment_batch([text], doc_ids=[doc_id] if doc_id else None, use_cache=use_cache)
    return results[0]


def analyse_sentiment_batch(
    texts: list[str],
    doc_ids: list[str] | None = None,
    batch_size: int = 16,
    use_cache: bool = True,
) -> list[SentimentResult]:
    """Analyse sentiment for a batch of texts on CPU.

    Checks cache first for items that have doc_ids. Un-cached items are
    batched through FinBERT on CPU, and cached.
    """
    if not texts:
        return []

    results: list[SentimentResult | None] = [None] * len(texts)
    uncached_indices: list[int] = []

    # 1. Check cache
    for i, text in enumerate(texts):
        did = doc_ids[i] if doc_ids and i < len(doc_ids) else None
        if use_cache and did:
            cached = get_cached_sentiment(did)
            if cached is not None:
                results[i] = cached
                continue
        uncached_indices.append(i)

    if not uncached_indices:
        return [r for r in results if r is not None]

    # 2. Run model or fallback for uncached items
    tokenizer, model = _load_finbert()

    if tokenizer is not None and model is not None:
        import torch

        # Process in CPU batches
        for start_idx in range(0, len(uncached_indices), batch_size):
            batch_item_indices = uncached_indices[start_idx : start_idx + batch_size]
            batch_texts = [texts[idx][:512] for idx in batch_item_indices]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs_tensor = torch.softmax(logits, dim=-1)

            # FinBERT config: 0 -> positive, 1 -> negative, 2 -> neutral
            id2label = model.config.id2label
            for b_i, global_idx in enumerate(batch_item_indices):
                p_vec = probs_tensor[b_i].tolist()
                p_dict = {
                    id2label.get(cls_idx, str(cls_idx)).lower(): round(p_val, 4)
                    for cls_idx, p_val in enumerate(p_vec)
                }
                p_pos = p_dict.get("positive", 0.0)
                p_neg = p_dict.get("negative", 0.0)
                p_neu = p_dict.get("neutral", 0.0)

                score = round(p_pos - p_neg, 4)
                label = max(p_dict, key=p_dict.get)

                res = SentimentResult(
                    score=score,
                    label=label,
                    probs={"positive": p_pos, "negative": p_neg, "neutral": p_neu},
                    confidence=max(p_pos, p_neg, p_neu),
                )
                results[global_idx] = res

                # Write to cache
                did = doc_ids[global_idx] if doc_ids and global_idx < len(doc_ids) else None
                if did:
                    save_sentiment_cache(did, res)
    else:
        # Fallback lexicon
        for global_idx in uncached_indices:
            res = _lexicon_sentiment(texts[global_idx])
            results[global_idx] = res
            did = doc_ids[global_idx] if doc_ids and global_idx < len(doc_ids) else None
            if did:
                save_sentiment_cache(did, res)

    return [r for r in results if r is not None]
