# Ripple System Architecture

This document provides a concise architectural reference for the Ripple AI/NLP Risk Engine & Strategic Stress-Testing Platform.

For detailed mathematical models, valuation formulas, and parameters, see [methodology.md](methodology.md).  
For technical limitations and operational assumptions, see [limitations.md](limitations.md).  
For the 7-slide pitch deck guide, see [presentation_outline.md](presentation_outline.md).

---

## Architecture Diagram

![System Architecture](architecture.png)

---

## Architectural Layers

1. **Ingestion Layer (`src/ingest/`):**
   - Ingests feeds from GDELT 2.0, NewsAPI / RSS feeds, and Twitter feeds.
   - Saves responses deterministically under `data/raw/<source>/<YYYY-MM-DD>/` for offline replay.
   - Automatically falls back to cached articles when rate limits (HTTP 429) or network errors occur.

2. **NLP Engine (`src/nlp/`):**
   - Deduplicates incoming text via URL hashes and title Jaccard word similarity.
   - Computes financial sentiment $[-1.0, +1.0]$ using `ProsusAI/finbert`.
   - Maps tickers, 11 GICS sectors, countries, and commodities via dictionary and regex rules.
   - Classifies events into 10 crisis taxonomies using Gemini 2.0 Flash with local rule-based fallbacks.

3. **Scoring & Calibration Layer (`src/scoring/`):**
   - Clusters events across a rolling 7-day window using FAISS cosine distance over `all-MiniLM-L6-v2` embeddings.
   - Tracks source sentiment divergence between institutional news and retail social media.
   - Maps raw impact to observed absolute abnormal stock and factor moves ($|CAR_{0,1}|$) via Isotonic Regression, yielding a calibrated impact score ($1.0$ to $10.0$).
   - Persists all risk signals to SQLite (`data/signals/signals.db`) and JSONL (`data/signals/signals.jsonl`).

4. **Strategic Stress Engine - Module B (`src/stress/`):**
   - Triggers when an incoming signal reaches $\text{impact\_score} \ge 7.0$.
   - Finds historical crisis analogs in FAISS; applies Bayesian shrinkage toward documented baseline priors when analogs $< 3$.
   - Propagates shocks through the NetworkX exposure graph with distance decay ($\delta = 0.5$, max 2 hops).
   - Revalues the banking book ($8.54B book: 70 loans, 60 corporate bonds, 50 OTC derivatives) across median ($\text{p50}$), mild ($\text{p5}$), and severe ($\text{p95}$) factor moves.
   - Logs every stress execution to `data/stress_runs/run_*.json`.

5. **Delivery & Interface Layer (`src/dashboard/`, `src/api/`):**
   - **Interactive Dashboard:** Streamlit + Plotly application running 100% offline with 5 dedicated views (Live Signals, Stress Test, Contagion Map, Model Check, Time Machine Replay).
   - **REST API:** FastAPI microservice providing `/signals`, `/signals/{id}`, and `/health` endpoints.
