"""Tests for FastAPI endpoints: /health, /signals, and /signals/{id}."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.scoring.schema import Signal
from src.scoring.store import save_signals


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a TestClient with a temporary SQLite database."""
    test_db = tmp_path / "test_api_signals.db"
    test_jsonl = tmp_path / "test_api_signals.jsonl"
    fake_config = {
        "scoring": {
            "storage": {
                "signals_db": str(test_db),
                "signals_jsonl": str(test_jsonl),
            }
        }
    }
    monkeypatch.setattr("src.scoring.store.load_config", lambda: fake_config)

    # Seed sample signals
    s1 = Signal(
        event_id="evt_test_api_01",
        timestamp="2026-10-16T15:00:00Z",
        entities={"companies": [{"ticker": "BARC", "name": "Barclays"}], "sectors": ["Financials"]},
        sentiment_score=-0.75,
        event_type="Credit Event",
        impact_score=8.7,
        confidence=0.94,
        divergence_flag=False,
        evidence=[{"url": "https://barc.local/1", "title": "Debt restructuring", "source": "newsapi", "snippet": "Notes"}],
        model_versions={"sentiment": "finbert"},
    )
    s2 = Signal(
        event_id="evt_test_api_02",
        timestamp="2026-10-15T10:00:00Z",
        entities={"companies": [], "sectors": ["Technology"]},
        sentiment_score=0.20,
        event_type="Corporate/Product",
        impact_score=4.5,
        confidence=0.85,
        divergence_flag=True,
        evidence=[],
        model_versions={"sentiment": "finbert"},
    )
    save_signals([s1, s2], jsonl_path=test_jsonl, db_path=test_db)

    return TestClient(app)


def test_api_health(client):
    """Verify /health returns 200 with status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "ripple-risk-engine"


def test_api_list_signals(client):
    """Verify /signals returns seeded signals and respects filters."""
    # All signals
    resp = client.get("/signals")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2

    # Filter min_impact >= 8.0
    resp_impact = client.get("/signals?min_impact=8.0")
    assert resp_impact.status_code == 200
    high_items = resp_impact.json()
    assert len(high_items) == 1
    assert high_items[0]["event_id"] == "evt_test_api_01"

    # Filter event_type
    resp_type = client.get("/signals?event_type=Credit%20Event")
    assert resp_type.status_code == 200
    assert len(resp_type.json()) == 1

    # Filter entity
    resp_entity = client.get("/signals?entity=Barclays")
    assert resp_entity.status_code == 200
    assert len(resp_entity.json()) == 1


def test_api_get_signal_by_id(client):
    """Verify /signals/{id} returns single signal or 404 on missing."""
    resp = client.get("/signals/evt_test_api_01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_id"] == "evt_test_api_01"
    assert data["impact_score"] == 8.7

    # 404 case
    resp_missing = client.get("/signals/non_existent_id")
    assert resp_missing.status_code == 404
