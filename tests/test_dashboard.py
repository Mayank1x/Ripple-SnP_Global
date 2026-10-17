"""Tests for Phase 7 - Interactive Dashboard (Streamlit + Plotly).

Verifies that all 6 views of the dashboard execute completely offline from local cache
without any exceptions or missing dependencies in both Dark and Light modes:
1. Overview
2. Signals
3. Stress test (both Ranked Contagion Cascade and Network Topology Map)
4. Rebalancer
5. Model check
6. Time machine
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

            # If on Stress test page, also test toggling contagion view
            if page == "Stress test":
                contagion_radios = [r for r in at.radio if "Ranked Contagion Cascade" in r.options]
                if contagion_radios:
                    contagion_radios[0].set_value("Network Topology Map").run()
                    assert len(at.exception) == 0, f"Exception switching contagion view in {mode} mode: {at.exception}"
