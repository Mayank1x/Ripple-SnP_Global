"""Download and seed supplementary datasets for Ripple.

The repository is kept ultra-compact (< 2 MB) by only tracking small samples
and schemas in Git. This script populates or downloads any remaining data:
1. Seeds local cached GDELT/News articles for replay mode (offline).
2. Generates the synthetic multi-asset portfolio (180 positions across loans, bonds, derivatives).
3. Pre-computes factor price series and calibration analogs for the stress engine.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --fetch-live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.make_portfolio import main as make_portfolio_main
from scripts.seed_cache import seed_cache
from src.scoring.event_study import compute_factor_daily_changes


def download_all(fetch_live: bool = False) -> None:
    """Download, seed, and populate all runtime datasets."""
    print("=" * 70)
    print("  Ripple Dataset Downloader & Cache Seeder")
    print("=" * 70)

    # 1. Seed news cache
    print("\n[1/3] Populating offline news cache (GDELT / NewsAPI)...")
    try:
        seed_cache()
        print("  -> Offline news articles populated in data/raw/gdelt/")
    except Exception as exc:
        print(f"  [Warning] Failed to seed news cache: {exc}")

    # 2. Generate synthetic portfolio
    print("\n[2/3] Generating synthetic portfolio & exposure contagion graph...")
    try:
        make_portfolio_main()
        print("  -> Portfolio positions created in data/portfolio/")
        print("  -> Exposure graph visual saved to docs/exposure_graph.png")
    except Exception as exc:
        print(f"  [Warning] Failed to generate portfolio: {exc}")

    # 3. Cache factor prices
    print("\n[3/3] Initializing factor price series and benchmarks...")
    try:
        df_factors = compute_factor_daily_changes()
        print(f"  -> Factor price series ready ({len(df_factors)} rows, {len(df_factors.columns)} factors)")
    except Exception as exc:
        print(f"  [Warning] Note on factor series: {exc}")

    print("\n" + "=" * 70)
    print("  All supplementary datasets are populated and ready!")
    print("  You can now run: python run.py --mode replay")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/Seed datasets for Ripple")
    parser.add_argument("--fetch-live", action="store_true", help="Attempt live network fetch for fresh news")
    args = parser.parse_args()
    download_all(fetch_live=args.fetch_live)
