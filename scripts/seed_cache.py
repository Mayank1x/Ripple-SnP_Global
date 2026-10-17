"""Seed the local cache with realistic historical financial news articles.

This populates data/raw/gdelt/<date>/ so that replay mode works seamlessly
even when GDELT API is offline, rate-limited, or run in an isolated/offline environment.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

SAMPLE_STORIES = [
    {
        "day_offset": 0,
        "title": "ECB warns euro area lenders of escalating commercial real estate default risks",
        "domain": "ft.com",
        "url": "https://www.ft.com/content/ecb-warns-euro-area-lenders-cre-defaults-2026",
        "sourcecountry": "United Kingdom",
    },
    {
        "day_offset": 0,
        "title": "Federal Reserve minutes emphasize caution on interest rate reductions amid sticky inflation",
        "domain": "wsj.com",
        "url": "https://www.wsj.com/economy/central-banking/fed-minutes-sticky-inflation-oct2026",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 1,
        "title": "US regional bank stocks decline after deposit repricing squeezes net interest margins",
        "domain": "bloomberg.com",
        "url": "https://www.bloomberg.com/news/articles/2026-10-11/regional-banks-nim-compression",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 1,
        "title": "Brent crude hits $92 on supply disruption fears in Strait of Hormuz",
        "domain": "reuters.com",
        "url": "https://www.reuters.com/markets/commodities/brent-crude-hormuz-concerns-2026-10-11",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 2,
        "title": "Nordic construction conglomerate defaults on $750m private debt facility",
        "domain": "ft.com",
        "url": "https://www.ft.com/content/nordic-construction-debt-default-oct2026",
        "sourcecountry": "Sweden",
    },
    {
        "day_offset": 2,
        "title": "High-yield bond spreads widen sharply as corporate downgrade momentum quickens",
        "domain": "cnbc.com",
        "url": "https://www.cnbc.com/2026/10/12/junk-bond-spreads-widen-downgrades.html",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 3,
        "title": "Semiconductor equipment suppliers issue revenue warnings over expanded export tariffs",
        "domain": "asia.nikkei.com",
        "url": "https://asia.nikkei.com/Business/Tech/semi-equipment-export-controls-impact",
        "sourcecountry": "Japan",
    },
    {
        "day_offset": 3,
        "title": "UK headline CPI ticks up to 3.8% driven by food and energy price rebounds",
        "domain": "theguardian.com",
        "url": "https://www.theguardian.com/business/2026/oct/13/uk-inflation-cpi-ticks-up",
        "sourcecountry": "United Kingdom",
    },
    {
        "day_offset": 4,
        "title": "Syndicated wholesale loans face tighter bank covenants as liquidity buffers narrow",
        "domain": "bloomberg.com",
        "url": "https://www.bloomberg.com/news/articles/2026-10-14/syndicated-loan-covenants-tighten",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 4,
        "title": "German industrial orders fall for third consecutive month signaling recessionary drag",
        "domain": "handelsblatt.com",
        "url": "https://www.handelsblatt.com/politik/deutschland/industrial-orders-fall-germany-2026",
        "sourcecountry": "Germany",
    },
    {
        "day_offset": 5,
        "title": "Offshore property developer in liquidation after failing to win creditor standstill",
        "domain": "scmp.com",
        "url": "https://www.scmp.com/business/china-business/article/3289012/property-developer-liquidation",
        "sourcecountry": "Hong Kong",
    },
    {
        "day_offset": 5,
        "title": "Treasury yield curve steepens as market prices in sustained fiscal borrowing pressure",
        "domain": "marketwatch.com",
        "url": "https://www.marketwatch.com/story/treasury-yield-curve-steepens-fiscal-debt-2026-10-15",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 6,
        "title": "WaFd and regional lenders report quarterly results with elevated credit provisions",
        "domain": "fool.com",
        "url": "https://www.fool.com/earnings/takeaways/2026/10/16/wafd-credit-provisions-q3",
        "sourcecountry": "United States",
    },
    {
        "day_offset": 6,
        "title": "Tech valuation pullback spreads across corporate credit markets and loan portfolios",
        "domain": "morningstar.com",
        "url": "https://www.morningstar.com/news/tech-debt-pullback-corporate-credit-2026",
        "sourcecountry": "United States",
    },
]


def seed_cache(start: date = date(2026, 10, 10), days: int = 7) -> None:
    """Populate cache files for standard queries across date range."""
    cache_base = Path("data/raw/gdelt")
    queries = ["bank_default", "bank_OR_default_OR_inflation", "_bank_OR_default_OR__interest_rate___"]

    for story in SAMPLE_STORIES:
        current_date = start + timedelta(days=story["day_offset"])
        date_str = current_date.isoformat()
        day_dir = cache_base / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        gdelt_record = {
            "url": story["url"],
            "url_mobile": "",
            "title": story["title"],
            "seendate": f"{current_date.strftime('%Y%m%d')}T120000Z",
            "socialimage": "",
            "domain": story["domain"],
            "language": "English",
            "sourcecountry": story["sourcecountry"],
        }

        for q in queries:
            cache_file = day_dir / f"gdelt_{q}.json"
            existing = []
            if cache_file.exists():
                try:
                    with open(cache_file, encoding="utf-8") as fh:
                        existing = json.load(fh)
                except Exception:
                    existing = []

            # Only append if URL not already in existing
            if not any(item.get("url") == gdelt_record["url"] for item in existing):
                existing.append(gdelt_record)

            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2, ensure_ascii=False)

    print(f"Cache seeded for {days} days ({start} to {start + timedelta(days=days)})")


if __name__ == "__main__":
    seed_cache()
