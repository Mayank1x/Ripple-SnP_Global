"""Sentiment-Driven Tactical Index Rebalancer (Module A).

Rebalances a mock 15-stock S&P 100 index by applying news and social risk tilts:
1. Input: risk signals from Ripple engine (sentiment_score, calibrated impact_score, confidence, entities).
2. Computes decayed tilt per constituent:
       tilt = sentiment_score * (impact_score / 10.0) * confidence * (0.5 ** (dt / half_life))
3. Modulates base weights:
       w_target = base_weight * (1 + tilt_strength * net_tilt)
4. Enforces min/max weight bounds, turnover caps, and normalises to 100%.
5. Persists historical rebalance executions to data/rebalance/history.jsonl.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.ingest.helpers import load_config


@dataclass
class RebalanceRecord:
    """Historical audit record of an index rebalance event."""
    timestamp: str
    weights: dict[str, float]
    weight_changes: dict[str, float]
    turnover: float
    signals_applied: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_constituents(filepath: str | Path | None = None) -> list[dict[str, Any]]:
    """Load index constituent stocks with equal initial weights."""
    if filepath is None:
        cfg = load_config()
        filepath = cfg.get("rebalance", {}).get(
            "constituents_file", "data/index/constituents.csv"
        )
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Constituents file not found: {p}")

    constituents = []
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            constituents.append({
                "ticker": row["ticker"].strip().upper(),
                "name": row["name"].strip(),
                "sector": row["sector"].strip(),
                "base_weight": float(row.get("base_weight", 1.0 / 15.0)),
            })
    return constituents


class IndexRebalancer:
    """Tactical sentiment-driven index rebalancing engine."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_config()
        reb_cfg = cfg.get("rebalance", {})

        self.half_life_days = float(reb_cfg.get("half_life_days", 2.0))
        self.tilt_strength = float(reb_cfg.get("tilt_strength", 0.50))
        self.min_weight = float(reb_cfg.get("min_weight", 0.02))
        self.max_weight = float(reb_cfg.get("max_weight", 0.15))
        self.turnover_cap = float(reb_cfg.get("turnover_cap", 0.03))
        self.tx_cost_bps = float(reb_cfg.get("transaction_cost_bps", 10.0))
        self.history_file = Path(reb_cfg.get("history_file", "data/rebalance/history.jsonl"))

        constituents_file = reb_cfg.get("constituents_file", "data/index/constituents.csv")
        self.constituents = load_constituents(constituents_file)
        self.tickers = [c["ticker"] for c in self.constituents]
        self.base_weights = {c["ticker"]: c["base_weight"] for c in self.constituents}
        self.current_weights = dict(self.base_weights)

    def _parse_ts(self, ts_str: str | None) -> datetime:
        """Parse ISO timestamp with UTC fallback."""
        if not ts_str:
            return datetime.now(timezone.utc)
        ts_clean = str(ts_str).strip()
        if ts_clean.endswith("Z"):
            ts_clean = ts_clean[:-1] + "+00:00"
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(ts_clean, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return datetime.now(timezone.utc)

    def compute_signals_tilt(
        self,
        signals: list[dict[str, Any]],
        as_of_time: datetime | str | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Compute decayed net sentiment tilt per constituent stock.

        Tilt formula per signal s for stock i:
            tilt_s = sentiment_score * (impact_score / 10.0) * confidence
            decay = 0.5 ** (dt_days / half_life_days)
            effective_tilt = tilt_s * decay
        """
        as_of_dt = self._parse_ts(as_of_time) if isinstance(as_of_time, str) else (as_of_time or datetime.now(timezone.utc))

        stock_tilts = {t: 0.0 for t in self.tickers}
        applied_signals: list[dict[str, Any]] = []

        # Build lookup maps for matching
        ticker_set = set(self.tickers)
        sector_map: dict[str, list[str]] = {}
        for c in self.constituents:
            sector_map.setdefault(c["sector"].lower(), []).append(c["ticker"])

        for sig in signals:
            sentiment = float(sig.get("sentiment_score", 0.0))
            impact = float(sig.get("impact_score", 5.0))
            confidence = float(sig.get("confidence", 0.8))
            sig_ts = self._parse_ts(sig.get("timestamp"))

            # Calculate elapsed time in days
            dt_days = max(0.0, (as_of_dt - sig_ts).total_seconds() / 86400.0)
            decay = 0.5 ** (dt_days / self.half_life_days)
            raw_tilt = sentiment * (impact / 10.0) * confidence
            effective_tilt = raw_tilt * decay

            # Identify target constituents
            matched_tickers = set()
            entities = sig.get("entities", {})

            # 1. Company match
            for comp in entities.get("companies", []):
                sym = (comp.get("ticker", "") if isinstance(comp, dict) else str(comp)).upper()
                if sym in ticker_set:
                    matched_tickers.add(sym)

            # 2. Sector match (if no direct company match, distribute fractional tilt)
            if not matched_tickers:
                for sec in entities.get("sectors", []):
                    s_lower = str(sec).lower()
                    if s_lower in sector_map:
                        for st_sym in sector_map[s_lower]:
                            matched_tickers.add(st_sym)

            if matched_tickers:
                title = sig.get("title", "")
                if not title and "evidence" in sig and sig["evidence"]:
                    title = sig["evidence"][0].get("title", "")

                for sym in matched_tickers:
                    stock_tilts[sym] += effective_tilt

                applied_signals.append({
                    "event_id": sig.get("event_id", ""),
                    "title": title[:100],
                    "matched_tickers": sorted(matched_tickers),
                    "sentiment_score": round(sentiment, 3),
                    "impact_score": round(impact, 2),
                    "effective_tilt": round(effective_tilt, 4),
                    "age_days": round(dt_days, 2),
                })

        # Bound net tilts to [-1.0, 1.0] to prevent extreme single-stock distortions
        for t in self.tickers:
            stock_tilts[t] = float(np.clip(stock_tilts[t], -1.0, 1.0))

        return stock_tilts, applied_signals

    def rebalance(
        self,
        signals: list[dict[str, Any]],
        as_of_time: datetime | str | None = None,
        save_history: bool = True,
    ) -> RebalanceRecord:
        """Execute portfolio rebalance given current signals.

        Steps:
        1. Compute decayed tilts per stock.
        2. Adjust target weights: w_target = base_weight * (1 + tilt_strength * tilt).
        3. Enforce turnover cap: |w_new - w_curr| <= turnover_cap.
        4. Enforce min/max bounds: [min_weight, max_weight].
        5. Renormalise strictly to sum = 1.0.
        """
        as_of_dt = self._parse_ts(as_of_time) if isinstance(as_of_time, str) else (as_of_time or datetime.now(timezone.utc))
        tilts, applied_signals = self.compute_signals_tilt(signals, as_of_time=as_of_dt)

        # Invariance check: if no signals are applied, current weights remain exactly unchanged
        if not applied_signals or all(abs(tilts[t]) < 1e-7 for t in self.tickers):
            rec = RebalanceRecord(
                timestamp=as_of_dt.isoformat(),
                weights=dict(self.current_weights),
                weight_changes={t: 0.0 for t in self.tickers},
                turnover=0.0,
                signals_applied=[],
            )
            if save_history:
                self._save_record(rec)
            return rec

        # Step 2: Compute raw target weights
        target_weights = {}
        for t in self.tickers:
            w_base = self.base_weights[t]
            tilt = tilts[t]
            target_weights[t] = w_base * (1.0 + self.tilt_strength * tilt)

        # Step 3 & 4: Bound clipping and turnover constraints
        new_weights = {}
        for t in self.tickers:
            w_curr = self.current_weights[t]
            w_targ = target_weights[t]

            # Turnover cap relative to current weight
            w_bounded = np.clip(w_targ, w_curr - self.turnover_cap, w_curr + self.turnover_cap)
            # Absolute min/max bounds
            w_bounded = np.clip(w_bounded, self.min_weight, self.max_weight)
            new_weights[t] = float(w_bounded)

        # Step 5: Iterative projection to renormalise sum to 1.0 while respecting bounds
        final_weights = self._project_weights(new_weights)

        # Calculate weight changes and one-way turnover
        weight_changes = {t: round(final_weights[t] - self.current_weights[t], 6) for t in self.tickers}
        turnover = float(0.5 * sum(abs(weight_changes[t]) for t in self.tickers))

        # Update current weights state
        self.current_weights = dict(final_weights)

        rec = RebalanceRecord(
            timestamp=as_of_dt.isoformat(),
            weights={t: round(final_weights[t], 6) for t in self.tickers},
            weight_changes=weight_changes,
            turnover=round(turnover, 6),
            signals_applied=applied_signals,
        )

        if save_history:
            self._save_record(rec)

        return rec

    def _project_weights(self, weights: dict[str, float], max_iter: int = 10) -> dict[str, float]:
        """Iterative bound projection ensuring exact sum=1.0 and min/max compliance."""
        w = np.array([weights[t] for t in self.tickers], dtype=np.float64)
        n = len(w)
        low = self.min_weight
        high = self.max_weight

        for _ in range(max_iter):
            w = np.clip(w, low, high)
            diff = 1.0 - np.sum(w)
            if abs(diff) < 1e-7:
                break
            # Distribute excess proportionally among non-saturated weights
            adjustable = (w > low + 1e-6) & (w < high - 1e-6)
            if not np.any(adjustable):
                w += diff / n
                break
            w[adjustable] += diff / np.sum(adjustable)

        w = w / np.sum(w)
        return {self.tickers[i]: float(w[i]) for i in range(n)}

    def _save_record(self, record: RebalanceRecord) -> None:
        """Append rebalance record to history.jsonl."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
