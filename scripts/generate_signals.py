"""Generate actionable risk signals from cached articles using Phase 2 NLP + Phase 3 Scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.schema import Document
from src.nlp.pipeline import analyse_batch
from src.scoring.pipeline import process_signals
from src.scoring.store import query_signals


def main() -> None:
    print("=" * 65)
    print("RIPPLE RISK ENGINE: Generating Signals from Cached Data")
    print("=" * 65)

    source_path = Path("data/raw/merged_articles.jsonl")
    if not source_path.exists():
        print(f"Error: {source_path} does not exist. Run scripts/fetch_history.py first.")
        sys.exit(1)

    raw_docs: list[Document] = []
    with open(source_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                raw_docs.append(Document.from_dict(json.loads(line)))

    print(f"Loaded {len(raw_docs)} cached raw documents.")

    # 1. Run NLP pipeline (sentiment, event classification, entities)
    print("Running NLP analysis (using FinBERT + Event classifier cache) ...")
    analysed_docs = analyse_batch(raw_docs, batch_size=32, use_cache=True)
    print(f"Analysed {len(analysed_docs)} documents.")

    # 2. Run clustering, impact scoring, and signal persistence
    print("Clustering documents into real-world events and computing impact scores ...")
    signals = process_signals(analysed_docs, save_to_disk=True)

    print("\n" + "-" * 65)
    print(f"Total Risk Signals Generated: {len(signals)}")
    print("-" * 65)

    # 3. Print breakdown by event type
    type_counts: dict[str, int] = {}
    for s in signals:
        type_counts[s.event_type] = type_counts.get(s.event_type, 0) + 1

    print("Signals by Event Type:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  - {etype:<25}: {count}")

    # 4. Print top 5 high-impact signals
    top_signals = sorted(signals, key=lambda s: -s.impact_score)[:5]
    print("\nTop High-Impact Signals:")
    for idx, s in enumerate(top_signals, start=1):
        div_str = " [DIVERGENCE FLAG]" if s.divergence_flag else ""
        print(f"\n[{idx}] {s.event_id} | {s.event_type} | Impact: {s.impact_score}/10{div_str}")
        print(f"    Sentiment: {s.sentiment_score:+.2f} | Conf: {s.confidence:.2f} | Time: {s.timestamp}")
        comps = [c["name"] for c in s.entities.get("companies", [])]
        sectors = s.entities.get("sectors", [])
        print(f"    Entities : Companies: {comps} | Sectors: {sectors}")
        if s.evidence:
            print(f"    Headline : {s.evidence[0]['title'][:85]}")

    # Verify SQLite queries
    db_signals = query_signals(limit=10)
    print(f"\nSQLite Verification: successfully retrieved {len(db_signals)} signals from database.")
    print("=" * 65)


if __name__ == "__main__":
    main()
