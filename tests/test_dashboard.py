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


def test_dashboard_all_five_pages_offline():
    """Verify that all 5 dashboard views render offline without errors."""
    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()

    # Page 1: Live Signals Feed
    assert len(at.exception) == 0, f"Page 1 Exception: {at.exception}"

    # Page 2: Portfolio Stress Test
    at.sidebar.radio[1].set_value("2. Portfolio Stress Test").run()
    assert len(at.exception) == 0, f"Page 2 Exception: {at.exception}"

    # Page 3: Network Contagion Map
    at.sidebar.radio[1].set_value("3. Network Contagion Map").run()
    assert len(at.exception) == 0, f"Page 3 Exception: {at.exception}"

    # Page 4: Model Check & Validation
    at.sidebar.radio[1].set_value("4. Model Check & Validation").run()
    assert len(at.exception) == 0, f"Page 4 Exception: {at.exception}"

    # Page 5: Time Machine Replay
    at.sidebar.radio[1].set_value("5. Time Machine Replay").run()
    assert len(at.exception) == 0, f"Page 5 Exception: {at.exception}"

    # Page 6: Tactical Rebalancer (Module A)
    at.sidebar.radio[1].set_value("6. Tactical Rebalancer (Module A)").run()
    assert len(at.exception) == 0, f"Page 6 Exception: {at.exception}"

