# Ripple — AI/NLP Risk Engine & Stress-Test Platform
# S&P Global & CRISIL Campus Hackathon 2026

**Status:** Phase 3 complete — Events clustering, Impact scoring, Risk Signals, and FastAPI server.

## What is Ripple?

Ripple ingests financial news from multiple sources, runs NLP to detect risk
events (sentiment, classification, impact), clusters them into real-world events,
generates actionable risk signals, and exposes them via FastAPI for downstream stress testing.

## Quick start

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy .env.example to .env, fill in optional keys
copy .env.example .env

# 4. Fetch 7 days of articles and cache them locally (Phase 1)
python scripts/fetch_history.py --start 2026-10-10 --end 2026-10-17 --query "bank OR default OR inflation"

# 5. Run NLP Evaluation & Metrics Report (Phase 2)
python scripts/eval_nlp.py

# 6. Generate Risk Signals & Populate SQLite / JSONL (Phase 3)
python scripts/generate_signals.py

# 7. Start the FastAPI Service (Phase 3)
uvicorn src.api.main:app --reload --port 8000
# Endpoints: http://127.0.0.1:8000/signals, http://127.0.0.1:8000/health

# 8. Run full test suite (29 tests)
pytest -v
```

## Project layout

```
src/
  ingest/     ← data fetchers (GDELT, NewsAPI/RSS, tweets CSV)
  nlp/        ← sentiment, entities, event classification (Phase 2)
  scoring/    ← impact calibration, historical analogs  (Phase 3)
  portfolio/  ← synthetic portfolio generator            (Phase 4)
  stress/     ← shock application, exposure graph        (Phase 5)
  api/        ← FastAPI endpoints                        (Phase 6)
  dashboard/  ← HTML + JS + Chart.js front-end           (Phase 6)
data/
  raw/        ← cached API responses (git-ignored)
  synthetic/  ← seeded portfolio data
  calibration/← event-study lookup tables
  audit/      ← stress-run logs
docs/         ← architecture notes
tests/        ← pytest suite
scripts/      ← CLI helpers (fetch_history, generate_portfolio, …)
config.yaml   ← all tunables (no secrets)
.env.example  ← template for secrets
```

## Replay mode

Every HTTP response is saved under `data/raw/<source>/<YYYY-MM-DD>/`.
Re-running `fetch_history.py` for the same date window reads from cache
instead of hitting the network. This means the whole pipeline works offline
once you have fetched data once.

## License

MIT — see LICENSE.
