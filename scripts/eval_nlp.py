"""Evaluation script for NLP pipeline against hand-labeled benchmark headlines.

Calculates and prints:
  - Event Type: Accuracy, Macro F1, Weighted F1, and per-class breakdown
  - Sentiment Agreement: Accuracy, Macro F1, Weighted F1
  - Summary metrics table

Usage:
  python scripts/eval_nlp.py [--eval-file data/eval/labels.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.ingest.schema import Document
from src.nlp.pipeline import analyse_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NLP pipeline on benchmark dataset.")
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/labels.csv",
        help="Path to CSV containing ground truth labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_file)

    if not eval_path.exists():
        print(f"Error: Eval file not found at {eval_path}")
        sys.exit(1)

    # Read ground truth
    items: list[dict[str, str]] = []
    with open(eval_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            items.append(row)

    print(f"Loaded {len(items)} evaluation samples from {eval_path}")
    print("Running NLP pipeline across benchmark headlines ...\n")

    # Create Document instances
    docs = [
        Document(
            id=item["id"],
            source="eval",
            published_at="2026-10-17T00:00:00Z",
            title=item["title"],
            text=item["title"],
            url=f"https://eval.local/{item['id']}",
            language="english",
            raw_meta={},
        )
        for item in items
    ]

    # Run analysis
    analysed_docs = analyse_batch(docs, batch_size=16, use_cache=True)

    y_true_events = [item["true_event_type"].strip() for item in items]
    y_pred_events = [d.event_type.strip() for d in analysed_docs]

    y_true_sent = [item["true_sentiment"].strip().lower() for item in items]
    y_pred_sent = [d.sentiment_label.strip().lower() for d in analysed_docs]

    # Metrics - Event Classification
    event_acc = accuracy_score(y_true_events, y_pred_events)
    event_macro_f1 = f1_score(y_true_events, y_pred_events, average="macro", zero_division=0)
    event_weighted_f1 = f1_score(y_true_events, y_pred_events, average="weighted", zero_division=0)

    # Metrics - Sentiment Classification
    sent_acc = accuracy_score(y_true_sent, y_pred_sent)
    sent_macro_f1 = f1_score(y_true_sent, y_pred_sent, average="macro", zero_division=0)
    sent_weighted_f1 = f1_score(y_true_sent, y_pred_sent, average="weighted", zero_division=0)

    # Print Report
    print("=" * 68)
    print(" " * 20 + "RIPPLE NLP EVALUATION REPORT")
    print("=" * 68)
    print(f"Total benchmark headlines: {len(items)}")
    print("-" * 68)
    print("1. EVENT TYPE CLASSIFICATION METRICS:")
    print(f"   Accuracy    : {event_acc:.2%}")
    print(f"   Macro F1    : {event_macro_f1:.4f}")
    print(f"   Weighted F1 : {event_weighted_f1:.4f}")
    print("\n   Per-Class Breakdown:")
    report_event = classification_report(y_true_events, y_pred_events, zero_division=0)
    for line in report_event.split("\n"):
        print(f"   {line}")

    print("-" * 68)
    print("2. SENTIMENT AGREEMENT METRICS:")
    print(f"   Accuracy    : {sent_acc:.2%}")
    print(f"   Macro F1    : {sent_macro_f1:.4f}")
    print(f"   Weighted F1 : {sent_weighted_f1:.4f}")
    print("\n   Per-Class Breakdown:")
    report_sent = classification_report(y_true_sent, y_pred_sent, zero_division=0)
    for line in report_sent.split("\n"):
        print(f"   {line}")

    print("=" * 68)


if __name__ == "__main__":
    main()
