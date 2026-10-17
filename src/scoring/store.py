"""Persistence for risk signals: JSON Lines and SQLite storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.ingest.helpers import load_config
from src.scoring.schema import Signal


def _get_storage_paths() -> tuple[Path, Path]:
    """Retrieve signals.jsonl and signals.db paths from config."""
    cfg = load_config()
    storage_cfg = cfg.get("scoring", {}).get("storage", {})
    jsonl_path = Path(storage_cfg.get("signals_jsonl", "data/signals/signals.jsonl"))
    db_path = Path(storage_cfg.get("signals_db", "data/signals/signals.db"))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return jsonl_path, db_path


def init_db(db_path: Path | str | None = None) -> Path:
    """Initialize SQLite signals schema with indexes."""
    if db_path is None:
        _, target_db = _get_storage_paths()
    else:
        target_db = Path(db_path)
    target_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                impact_score REAL NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                divergence_flag INTEGER NOT NULL,
                entities_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                model_versions_json TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_event_type ON signals(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_impact ON signals(impact_score)")
        conn.commit()

    return target_db


def save_signals(
    signals: list[Signal],
    jsonl_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Write signals to both JSONL file and SQLite database."""
    if not signals:
        return

    def_jsonl, def_db = _get_storage_paths()
    target_jsonl = Path(jsonl_path) if jsonl_path else def_jsonl
    target_db = init_db(db_path or def_db)

    # 1. Update SQLite database (UPSERT)
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        for s in signals:
            raw_dict = s.to_dict()
            cursor.execute(
                """
                INSERT OR REPLACE INTO signals (
                    event_id, timestamp, event_type, impact_score,
                    sentiment_score, confidence, divergence_flag,
                    entities_json, evidence_json, model_versions_json, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s.event_id,
                    s.timestamp,
                    s.event_type,
                    s.impact_score,
                    s.sentiment_score,
                    s.confidence,
                    1 if s.divergence_flag else 0,
                    json.dumps(s.entities, ensure_ascii=False),
                    json.dumps(s.evidence, ensure_ascii=False),
                    json.dumps(s.model_versions, ensure_ascii=False),
                    json.dumps(raw_dict, ensure_ascii=False),
                ),
            )
        conn.commit()

    # 2. Append/write to signals.jsonl
    # Load existing IDs to avoid duplicate lines in jsonl
    existing_ids = set()
    if target_jsonl.exists():
        try:
            with open(target_jsonl, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        item = json.loads(line)
                        if "event_id" in item:
                            existing_ids.add(item["event_id"])
        except Exception:
            existing_ids = set()

    with open(target_jsonl, "a", encoding="utf-8") as fh:
        for s in signals:
            if s.event_id not in existing_ids:
                fh.write(s.to_json() + "\n")
                existing_ids.add(s.event_id)


def query_signals(
    since: str | None = None,
    event_type: str | None = None,
    min_impact: float | None = None,
    entity: str | None = None,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[Signal]:
    """Query signals from SQLite with optional filtering."""
    _, def_db = _get_storage_paths()
    target_db = Path(db_path) if db_path else def_db
    if not target_db.exists():
        return []

    conditions: list[str] = []
    params: list[Any] = []

    if since:
        conditions.append("timestamp >= ?")
        params.append(since)
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type)
    if min_impact is not None:
        conditions.append("impact_score >= ?")
        params.append(float(min_impact))
    if entity:
        conditions.append("entities_json LIKE ?")
        params.append(f"%{entity}%")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"SELECT raw_json FROM signals{where_clause} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    results: list[Signal] = []
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            try:
                data = json.loads(row[0])
                results.append(Signal.from_dict(data))
            except Exception:
                continue

    return results


def get_signal_by_id(
    signal_id: str,
    db_path: Path | str | None = None,
) -> Signal | None:
    """Fetch single signal by its event_id."""
    _, def_db = _get_storage_paths()
    target_db = Path(db_path) if db_path else def_db
    if not target_db.exists():
        return None

    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT raw_json FROM signals WHERE event_id = ?", (signal_id,))
        row = cursor.fetchone()
        if row:
            try:
                data = json.loads(row[0])
                return Signal.from_dict(data)
            except Exception:
                return None
    return None
