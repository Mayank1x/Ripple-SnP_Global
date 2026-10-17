# Ripple — AI/NLP Risk Engine & Stress-Test Platform
# S&P Global & CRISIL Campus Hackathon 2026

**Status:** Phase 8 Complete — Hardening, Master Orchestration Runner (`run.py`), Replay & Live Modes with Automated Fallback, Sub-Minute Replay, Stress Engine (Module B), and Interactive Dashboard.

---

## What is Ripple?

Ripple is an enterprise-grade AI/NLP risk intelligence platform designed to ingest unstructured financial text (news feeds, social media, macro filings), extract calibrated risk events, compute empirical factor shocks, and propagate contagion across wholesale banking books.

### Key Capabilities
- **Multi-Source Ingestion**: GDELT 2.0, NewsAPI / RSS feeds, and Kaggle financial tweets with automatic offline cache fallback.
- **Financial NLP Engine**: FinBERT domain sentiment, NER entity recognition (companies, sectors, countries, commodities), and zero-shot/few-shot Gemini event classification with local rules fallback.
- **Rolling Clustering & Novelty**: Sentence-Transformers + FAISS sliding-window clustering with source sentiment divergence tracking.
- **Empirical Isotonic Calibration**: Isotonic regression mapping raw impact to realized absolute abnormal returns ($|CAR_{0,1}|$ and volatility shifts).
- **Module B Strategic Stress Engine**: NetworkX contagion propagation, empirical factor shocks ($p5/p50/p95$), Bayesian shrinkage toward documented crisis priors, and multi-asset position revaluation (loans, corporate bonds, derivatives).
- **Interactive Offline Dashboard**: Streamlit + Plotly visual interface with 5 dedicated views (Live Signals, Stress Test, Contagion Map, Model Check, Time Machine Replay).

---

## One-Command Master Execution

Ripple provides a unified master CLI runner that executes the full end-to-end pipeline (ingestion -> deduplication -> NLP analysis -> clustering -> calibration -> signal generation -> strategic stress test) in a single command:

```bash
# Replay Mode (100% Offline, runs demo window in ~45 seconds)
python run.py --mode replay

# Live Mode (attempts live APIs with graceful fallback to cached data if rate-limited or offline)
python run.py --mode live --query "bank OR default OR inflation"
```

If an external API (like GDELT or Gemini) fails or is rate-limited in live mode, Ripple:
1. Automatically falls back to cached articles.
2. Writes an alert banner payload to `data/signals/ingestion_status.json`.
3. Displays a prominent warning banner at the top of the interactive dashboard.

---

## Fresh-Clone Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Mayank1x/Ripple-SnP_Global.git
cd "Ripple-SnP_Global"

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows PowerShell / CMD
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template (optional API keys for NewsAPI / Gemini)
copy .env.example .env       # Windows
# cp .env.example .env       # Linux / macOS

# 5. Populate offline sample cache & generate synthetic portfolio (< 10 seconds)
python scripts/download_data.py

# 6. Run master replay pipeline (< 60 seconds)
python run.py --mode replay

# 7. Launch Interactive Dashboard (Streamlit + Plotly, 100% offline)
streamlit run src/dashboard/app.py
# URL: http://localhost:8501
```

---

## Individual Module Scripts

For deep-dive testing and inspection of individual components:

```bash
# Ingestion: Fetch 7-day news window
python scripts/fetch_history.py --start 2026-10-10 --end 2026-10-17

# NLP: Run FinBERT & Gemini evaluation report
python scripts/eval_nlp.py

# Signals: Generate & store signals in SQLite and JSONL
python scripts/generate_signals.py

# Calibration: Run Event Study & Backtest report (writes docs/calibration.png)
python scripts/backtest_report.py

# Portfolio: Generate synthetic 180-position portfolio & contagion graph
python scripts/make_portfolio.py

# Stress Engine: Execute stress tests on stored signals
python scripts/run_stress.py

# FastAPI Service: Serve signals and health check REST endpoints
uvicorn src.api.main:app --reload --port 8000
```

---

## Testing & Code Quality

Ripple maintains a 100% passing test suite across all 8 development phases:

```bash
# Run all 54 pytest tests
pytest -v

# Run fast replay pipeline benchmark (< 60 seconds)
pytest -v tests/test_master_run.py

# Run linter
ruff check --select E,F,W --ignore E501,E402 run.py src/ tests/ scripts/
```

---

## Repository Size & Data Policy

- **Total Git Tracked Size:** **~1.7 MiB** (fast to clone and lightweight).
- Only small representative samples, schemas, and lookup tables are stored in `data/`.
- Heavy runtime caches, model checkpoints, and generated logs are excluded via `.gitignore`.
- Running `python scripts/download_data.py` instantly seeds all offline caches.

---

## Project Layout

```
run.py                ← Master CLI runner (--mode replay / --mode live)
config.yaml           ← Centralized tunable parameters (thresholds, weights, priors)
requirements.txt      ← Production dependencies
src/
  ingest/             ← Ingestion modules (GDELT, NewsAPI, RSS, tweets CSV)
  nlp/                ← FinBERT sentiment, NER entity recognition, Gemini event classifier
  scoring/            ← Rolling FAISS clustering, isotonic impact calibration, SQLite/JSONL store
  portfolio/          ← Synthetic wholesale book generator (180 positions), Exposure Graph
  stress/             ← Factor shocks (p5/p50/p95), Bayesian prior shrinkage, multi-asset revaluation
  api/                ← FastAPI endpoints (/signals, /health, /stress)
  dashboard/          ← Streamlit + Plotly interactive analytics dashboard
data/
  raw/                ← Cached API payloads (GDELT, RSS, prices)
  portfolio/          ← Generated portfolio CSV/JSON ($8.54B book)
  calibration/        ← Isotonic regression tables and crisis analog factors
  signals/            ← Persisted risk signals (signals.jsonl, signals.db)
  stress_runs/        ← Auditable JSON outputs of executed stress tests
docs/                 ← Architecture, methodology, calibration charts, exposure graph
tests/                ← Pytest test suite (54 unit and integration tests)
scripts/              ← CLI helpers (download_data, make_portfolio, eval_nlp, etc.)
```

---

## License

MIT — see [LICENSE](file:///LICENSE).
