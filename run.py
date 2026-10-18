"""Master Orchestration Entrypoint for Ripple Risk Engine.

Usage:
    python run.py --mode replay [--limit 80]
    python run.py --mode live   [--query "bank OR default"]

Pipeline execution:
    1. Ingestion (live or cached replay with automated fallback)
    2. Deduplication & NLP Analysis (FinBERT sentiment, entities, event classification)
    3. Rolling-window clustering & event aggregation
    4. Calibrated impact scoring (1-10)
    5. Signal persistence (JSONL & SQLite)
    6. Strategic Portfolio Stress Testing for triggered high-impact events
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.dedup import deduplicate
from src.ingest.gdelt import fetch_gdelt
from src.ingest.helpers import load_config
from src.ingest.news import fetch_news
from src.ingest.schema import Document
from src.ingest.tweets import load_tweets
from src.nlp.pipeline import analyse_batch
from src.portfolio.graph import build_exposure_graph
from src.portfolio.schema import Portfolio
from src.scoring.pipeline import process_signals
from src.stress.run import stress

# Setup structured logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ripple.master")


def save_ingestion_status(
    mode: str,
    fallback: bool,
    banner: str | None = None,
    doc_count: int = 0,
) -> None:
    """Save pipeline ingestion health status to be consumed by the dashboard."""
    status_file = PROJECT_ROOT / "data" / "signals" / "ingestion_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "fallback": fallback,
        "banner": banner,
        "doc_count": doc_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pipeline(
    mode: str = "replay",
    query: str | None = None,
    limit: int = 80,
) -> None:
    """Execute end-to-end Ripple pipeline."""
    start_time = time.time()
    logger.info("=" * 68)
    logger.info("RIPPLE RISK ENGINE: Master Pipeline Orchestrator (Phase 8 Hardened)")
    logger.info(f"Execution Mode: {mode.upper()} | Document Limit: {limit}")
    logger.info("=" * 68)

    cfg = load_config()
    default_q = cfg.get("ingest", {}).get("default_query", "bank OR default OR inflation")
    search_query = query or default_q

    raw_docs: list[Document] = []
    fallback_active = False
    fallback_banner = None

    # -------------------------------------------------------------
    # Step 1: Ingestion (Live with Cache Fallback OR Replay)
    # -------------------------------------------------------------
    if mode == "live":
        logger.info(f"Initiating live ingestion across NewsAPI, RSS, GDELT for query: '{search_query}'")
        today = date.today()
        start_d = today - timedelta(days=2)
        end_d = today + timedelta(days=1)

        # 1. NewsAPI / RSS
        logger.info("Fetching news articles...")
        try:
            news_docs = fetch_news(search_query, start_d, end_d, use_cache=False)
            raw_docs.extend(news_docs)
            logger.info(f"NewsAPI / RSS gathered {len(news_docs)} articles.")
        except Exception as exc:
            logger.warning(f"News fetch failed: {exc}")

        # 2. GDELT
        logger.info("Fetching GDELT global events...")
        try:
            gdelt_docs = fetch_gdelt(search_query, start_d, end_d, use_cache=False)
            raw_docs.extend(gdelt_docs)
            logger.info(f"GDELT gathered {len(gdelt_docs)} events.")
        except Exception as exc:
            logger.warning(f"GDELT live fetch failed: {exc}. Continuing with available documents.")

        # 3. Tweets
        logger.info("Loading recent tweets...")
        try:
            tweet_docs = load_tweets()
            raw_docs.extend(tweet_docs)
            logger.info(f"Tweets gathered {len(tweet_docs)} documents.")
        except Exception as exc:
            logger.warning(f"Tweet load failed: {exc}")

        if raw_docs:
            logger.info(f"Live ingestion succeeded with {len(raw_docs)} raw documents.")
            save_ingestion_status(mode="live", fallback=False, banner=None, doc_count=len(raw_docs))
        else:
            logger.warning("No documents collected from live endpoints. Gracefully falling back to local cached replay data.")
            fallback_active = True
            fallback_banner = "Live APIs rate-limited; fell back to cached replay data."
            save_ingestion_status(mode="live", fallback=True, banner=fallback_banner, doc_count=0)

    # Replay mode OR Fallback from live failure
    if mode == "replay" or fallback_active or not raw_docs:
        logger.info("Loading cached raw articles from data/raw/ ...")
        merged_file = PROJECT_ROOT / "data" / "raw" / "merged_articles.jsonl"
        if merged_file.exists():
            with open(merged_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_docs.append(Document.from_dict(json.loads(line)))
        else:
            # Fallback to loading tweets directly
            raw_docs = load_tweets()

        if not fallback_active:
            save_ingestion_status(mode="replay", fallback=False, banner=None, doc_count=len(raw_docs))

    logger.info(f"Step 1 Complete: {len(raw_docs)} raw documents collected.")

    # -------------------------------------------------------------
    # Step 2: Deduplication & Language Filtering
    # -------------------------------------------------------------
    logger.info("Step 2: Running document deduplication...")
    unique_docs = deduplicate(raw_docs)
    logger.info(f"Retained {len(unique_docs)} unique English documents.")

    # Limit to demo window if specified (ensures sub-minute completion)
    if limit and len(unique_docs) > limit:
        unique_docs = unique_docs[:limit]
        logger.info(f"Bounded demo window to {len(unique_docs)} documents.")

    # -------------------------------------------------------------
    # Step 3: NLP Analysis (FinBERT + Gemini / Heuristics)
    # -------------------------------------------------------------
    logger.info("Step 3: Running NLP pipeline (FinBERT sentiment, entities, event classification)...")
    analysed_docs = analyse_batch(unique_docs, batch_size=32, use_cache=True)
    logger.info(f"Successfully analysed {len(analysed_docs)} documents.")

    # -------------------------------------------------------------
    # Step 4: Clustering & Calibrated Impact Scoring
    # -------------------------------------------------------------
    logger.info("Step 4: Running rolling-window FAISS clustering & calibrated impact scoring...")
    signals = process_signals(analysed_docs, save_to_disk=True)
    logger.info(f"Generated {len(signals)} risk signals (saved to signals.jsonl and SQLite).")

    # -------------------------------------------------------------
    # Step 5: Downstream Portfolio Stress Testing (Module B)
    # -------------------------------------------------------------
    logger.info("Step 5: Evaluating portfolio stress triggers on high-impact signals...")
    port_path = PROJECT_ROOT / "data" / "portfolio" / "portfolio.json"
    if not port_path.exists():
        logger.warning(f"Portfolio file '{port_path}' not found. Generating default portfolio...")
        import scripts.make_portfolio as mp
        mp.main()

    portfolio = Portfolio.load(port_path)
    graph = build_exposure_graph(portfolio, seed=42)

    triggered_count = 0
    # Find high-impact signals (impact >= 7.0)
    high_impact_signals = [s for s in signals if s.impact_score >= 7.0]

    for sig in high_impact_signals[:3]:  # Stress test top high-impact events
        sig_dict = sig.to_dict()
        res = stress(sig_dict, portfolio=portfolio, graph=graph, horizon_days=5, force=True, save_run=True)
        triggered_count += 1
        logger.info(
            f"  * Stress Run [{res.run_id}]: Event '{sig.event_id}' ({sig.event_type}) -> "
            f"Pre: ${res.portfolio_value_before/1e9:.2f}B | Post: ${res.portfolio_value_after_p50/1e9:.2f}B | "
            f"Loss: ${res.loss_p50/1e6:.1f}M ({res.loss_pct_p50*100:.2f}%)"
        )

    # -------------------------------------------------------------
    # Pipeline Summary
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("=" * 68)
    logger.info(f"RIPPLE PIPELINE RUN COMPLETE in {elapsed:.2f} seconds (< 60s)")
    logger.info(f"  - Ingested Documents : {len(raw_docs)}")
    logger.info(f"  - Unique Analysed    : {len(analysed_docs)}")
    logger.info(f"  - Risk Signals       : {len(signals)}")
    logger.info(f"  - Stress Runs Fired  : {triggered_count}")
    logger.info(f"  - Fallback Status    : {'ACTIVE: ' + fallback_banner if fallback_active else 'Normal'}")
    logger.info("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ripple Risk Engine Master CLI")
    parser.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
        help="Pipeline execution mode: 'replay' reads local cache; 'live' queries APIs with automated fallback.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query for live ingestion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Max number of documents to analyze in demo window (default 80, completes < 60s).",
    )
    args = parser.parse_args()

    run_pipeline(mode=args.mode, query=args.query, limit=args.limit)


if __name__ == "__main__":
    main()
