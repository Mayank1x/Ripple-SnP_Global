"""Tests for Feature 5: Risk Committee Audit Pack Generator."""

import json
from pathlib import Path
import pytest

from src.dashboard.report import generate_audit_pack_html, save_audit_pack_to_disk
from src.stress.revalue import compute_regulatory_capital


def test_regulatory_capital_calculation():
    """Verify Basel III capital adequacy computation."""
    pre_val = 8_500_000_000.0  # $8.5B
    loss = 150_000_000.0        # $150M

    cap = compute_regulatory_capital(pre_val, loss, base_cet1_ratio=0.1350)
    assert cap["base_cet1"] == 0.1350
    assert cap["stressed_cet1"] < 0.1350
    assert cap["depletion_bps"] > 0.0
    assert cap["min_requirement"] == 0.1050
    assert cap["is_compliant"] is True
    assert "Adequate" in cap["status"]


def test_generate_audit_pack_html():
    """Verify that generate_audit_pack_html produces a valid, complete HTML document."""
    mock_run = {
        "run_id": "stress_test_run_123",
        "timestamp": "2026-10-18T10:00:00Z",
        "portfolio_value_before": 8_538_147_288.10,
        "portfolio_value_after_p50": 8_387_795_248.26,
        "loss_p50": 150_352_039.84,
        "loss_pct_p50": 0.0176,
        "band": {
            "p5_loss": 138_000_000.0,
            "p95_loss": 165_000_000.0,
        },
        "signal_summary": {
            "event_id": "evt_test_01",
            "title": "US Treasury yield curve steepens amid inflation concerns",
            "event_type": "Macro Shock",
            "impact_score": 8.1,
            "sentiment_score": -0.65,
            "confidence": 0.88,
        },
        "loss_by_asset_class": {
            "Loans": 95_000_000.0,
            "Bonds": 42_000_000.0,
            "Derivatives": 13_352_039.84,
        },
        "top_10_positions": [
            {
                "counterparty": "Test Bank Corp",
                "asset_class": "Loan",
                "sector": "Financials",
                "country": "United States",
                "pre_value": 150_000_000.0,
                "loss": 8_500_000.0,
                "loss_pct": 0.0567,
            }
        ],
        "graph_shock_nodes": {
            "Financials": 0.85,
            "Technology": 0.45,
        },
    }

    html = generate_audit_pack_html(mock_run, is_custom=False)
    assert "<!DOCTYPE html>" in html
    assert "Ripple Risk Analytics" in html
    assert "stress_test_run_123" in html
    assert "Common Equity Tier 1 (CET1)" in html
    assert "Test Bank Corp" in html
    assert "window.print()" in html


def test_save_audit_pack_to_disk(tmp_path):
    """Verify saving the audit pack to a custom directory."""
    mock_run = {
        "run_id": "disk_test_001",
        "portfolio_value_before": 1_000_000.0,
        "loss_p50": 50_000.0,
    }

    path = save_audit_pack_to_disk(mock_run, output_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".html"
    content = path.read_text(encoding="utf-8")
    assert "disk_test_001" in content
