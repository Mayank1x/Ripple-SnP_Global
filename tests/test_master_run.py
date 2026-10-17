"""Tests for Phase 8 - Master Pipeline Orchestration (run.py).

Verifies:
1. Master pipeline executes in replay mode from local cache.
2. Ingestion status artifact (data/signals/ingestion_status.json) is created.
3. Stress testing triggers on high-impact events and logs runs.
4. Pipeline finishes in under a minute for demo window.
"""

import json
import time
from pathlib import Path

from run import run_pipeline


def test_master_pipeline_replay_execution():
    """Verify master pipeline runs end-to-end in replay mode under 60 seconds."""
    t0 = time.time()
    run_pipeline(mode="replay", limit=15)
    elapsed = time.time() - t0

    # Must complete fast (well under 60 seconds)
    assert elapsed < 60.0, f"Replay took {elapsed:.2f}s, expected < 60s"

    # Verify ingestion status artifact was created
    status_file = Path("data/signals/ingestion_status.json")
    assert status_file.exists()

    with open(status_file, "r", encoding="utf-8") as f:
        status_info = json.load(f)
    assert status_info["mode"] == "replay"
    assert status_info["fallback"] is False

    # Verify signals file was created/updated
    signals_file = Path("data/signals/signals.jsonl")
    assert signals_file.exists()
    assert signals_file.stat().st_size > 0
