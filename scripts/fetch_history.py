"""Fetch and cache articles for a date window.

Usage:
  python scripts/fetch_history.py --start 2026-10-10 --end 2026-10-17 \\
      --query "bank OR default OR inflation"

After the first run, the same command reads from cache (replay mode).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# Add project root to sys.path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.dedup import deduplicate
from src.ingest.gdelt import fetch_gdelt
from src.ingest.helpers import load_config
from src.ingest.news import fetch_news
from src.ingest.tweets import load_tweets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    cfg = load_config()
    default_query = cfg["ingest"]["default_query"]

    parser = argparse.ArgumentParser(
        description="Fetch financial-news articles and cache them locally."
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="Start date (YYYY-MM-DD, inclusive).",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="End date (YYYY-MM-DD, exclusive).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=default_query,
        help=f'Search query (default: "{default_query}").',
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached files and re-download everything.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all ingestors, deduplicate, print summary."""
    args = parse_args()
    use_cache = not args.no_cache

    print(f"Query : {args.query}")
    print(f"Window: {args.start} -> {args.end}")
    print(f"Cache : {'ON (replay mode)' if use_cache else 'OFF (fresh download)'}")
    print()

    # --- GDELT ---
    print("[1/3] Fetching GDELT articles ...")
    gdelt_docs = fetch_gdelt(args.query, args.start, args.end, use_cache=use_cache)
    print(f"      => {len(gdelt_docs)} raw articles")

    # --- News (NewsAPI or RSS) ---
    print("[2/3] Fetching news articles ...")
    news_docs = fetch_news(args.query, args.start, args.end, use_cache=use_cache)
    print(f"      => {len(news_docs)} raw articles")

    # --- Tweets CSV ---
    print("[3/3] Loading tweets CSV ...")
    tweet_docs = load_tweets(start=args.start, end=args.end)
    print(f"      => {len(tweet_docs)} tweets")

    # --- Merge & deduplicate ---
    all_docs = gdelt_docs + news_docs + tweet_docs
    print(f"\nTotal raw documents: {len(all_docs)}")

    clean_docs = deduplicate(all_docs)
    print(f"After dedup + language filter: {len(clean_docs)}")

    # --- Summary by source ---
    counts = Counter(doc.source for doc in clean_docs)
    print("\nCounts by source:")
    for source, count in sorted(counts.items()):
        print(f"  {source:10s}: {count}")

    # --- Write merged output ---
    out_path = Path("data/raw/merged_articles.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n" for doc in clean_docs)

    print(f"\nMerged output written to {out_path} ({len(clean_docs)} docs)")


if __name__ == "__main__":
    main()
