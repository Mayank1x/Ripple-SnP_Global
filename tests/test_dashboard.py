"""Tests for Phase 7 - Interactive Dashboard (Streamlit + Plotly).

Verifies that all 5 pages of the dashboard execute completely offline from local cache
without any exceptions or missing dependencies:
1. Live Signals Feed
2. Portfolio Stress Test
3. Network Contagion Map
4. Model Check & Validation
5. Time Machine Replay
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest


def test_dashboard_all_six_pages_offline():
    """Verify that all 6 dashboard views render offline in Dark and Light modes without errors."""
    pages = ["Overview", "Signals", "Stress test", "Rebalancer", "Model check", "Time machine"]

    for mode in ["Dark", "Light"]:
        for page in pages:
            at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
            at.run()
            assert len(at.exception) == 0, f"Startup Exception: {at.exception}"
            # Select Theme
            at.radio[1].set_value(mode).run()
            assert len(at.exception) == 0, f"Theme Exception in {mode}: {at.exception}"
            # Navigate to Page
            at.radio[2].set_value(page).run()
            assert len(at.exception) == 0, f"Exception on {page} in {mode} mode: {at.exception}"


