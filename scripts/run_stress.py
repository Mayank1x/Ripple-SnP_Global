"""CLI Runner for Ripple Stress Engine (Module B).

Usage:
    python scripts/run_stress.py [--signal-id <id>] [--horizon 5] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stress.run import stress


def main() -> None:
    parser = argparse.ArgumentParser(description="Ripple Stress Engine Runner")
    parser.add_argument("--signal-id", type=str, default=None, help="Target event signal ID from signals.jsonl")
    parser.add_argument("--horizon", type=int, default=5, help="Forward shock horizon in days (1 or 5, default 5)")
    parser.add_argument("--force", action="store_true", help="Force execution even if impact_score < 7.0")
    args = parser.parse_args()

    signals_path = Path("data/signals/signals.jsonl")
    if not signals_path.exists():
        print(f"Error: {signals_path} not found. Run scripts/generate_signals.py first.")
        return

    # Load signals
    signals: list[dict] = []
    with open(signals_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                signals.append(json.loads(line))

    if not signals:
        print("No signals found in data/signals/signals.jsonl.")
        return

    # Select target signal
    target_signal = None
    if args.signal_id:
        for s in signals:
            if s.get("event_id") == args.signal_id:
                target_signal = s
                break
        if not target_signal:
            print(f"Signal ID '{args.signal_id}' not found.")
            return
    else:
        # Pick highest impact signal
        target_signal = max(signals, key=lambda s: float(s.get("impact_score", 0.0)))

    print("=" * 72)
    print("RIPPLE STRESS ENGINE: Wholesale Portfolio Scenario Evaluation")
    print("=" * 72)
    sig_summary = target_signal
    title = sig_summary.get("title", "")
    if not title and "evidence" in sig_summary and sig_summary["evidence"]:
        title = sig_summary["evidence"][0].get("title", "")

    print(f"Target Event ID    : {sig_summary.get('event_id')}")
    print(f"Event Headline     : {title[:65]}...")
    print(f"Event Type         : {sig_summary.get('event_type')}")
    print(f"Impact Score       : {sig_summary.get('impact_score')} / 10.0")
    print(f"Sentiment Score    : {sig_summary.get('sentiment_score'):+.4f}")
    print(f"Horizon Days       : {args.horizon} trading days")
    print("-" * 72)

    # Run stress engine
    result = stress(target_signal, horizon_days=args.horizon, force=args.force)

    if not result.triggered:
        print(f"[SKIPPED] {result.trigger_reason}")
        return

    print(f"[TRIGGERED] {result.trigger_reason}")
    print(f"Run ID             : {result.run_id}")
    print(f"Artifact Saved To  : {result.saved_filepath}")
    print("-" * 72)
    print(f"Factor Shocks Info : {result.shocks.get('message', '')}")
    if result.shocks.get("analogs"):
        print("Historical Analogs Matched:")
        for a in result.shocks["analogs"][:3]:
            print(f"  * [{a.get('event_id')}] (sim: {a.get('similarity'):.3f}) {a.get('title')[:55]}...")

    print("=" * 72)
    print("                      PORTFOLIO VALUATION & LOSS BAND")
    print("=" * 72)
    pre = result.portfolio_value_before
    post_50 = result.portfolio_value_after_p50
    loss_50 = result.loss_p50
    loss_pct_50 = result.loss_pct_p50 * 100.0

    p5_loss = result.band.get("p5_loss", 0.0)
    p95_loss = result.band.get("p95_loss", 0.0)

    print(f"Pre-Stress Value (Base)     : ${pre / 1e9:8.3f} Billion")
    print(f"Post-Stress Value (p50)     : ${post_50 / 1e9:8.3f} Billion")
    print(f"Total Portfolio Loss (p50)  : ${loss_50 / 1e6:8.2f} Million ({loss_pct_50:.2f}%)")
    print(f"  - Mild Band (p5) Loss     : ${p5_loss / 1e6:8.2f} Million")
    print(f"  - Severe Band (p95) Loss  : ${p95_loss / 1e6:8.2f} Million")
    print("-" * 72)

    print("Loss by Asset Class:")
    for ac, loss_amt in result.loss_by_asset_class.items():
        pct = (loss_amt / loss_50 * 100.0) if loss_50 > 0 else 0.0
        print(f"  * {ac:<15}: ${loss_amt / 1e6:7.2f}M  ({pct:5.1f}%)")

    print("\nTop 5 Sector Losses:")
    for sec, loss_amt in list(result.loss_by_sector.items())[:5]:
        pct = (loss_amt / loss_50 * 100.0) if loss_50 > 0 else 0.0
        print(f"  * {sec:<22}: ${loss_amt / 1e6:7.2f}M  ({pct:5.1f}%)")

    print("\nTop 5 Impacted Counterparties:")
    for p in result.top_10_positions[:5]:
        cp = p.get("counterparty", "")[:24]
        ac = p.get("asset_class", "")
        ploss = p.get("loss", 0.0)
        print(f"  * {cp:<25} ({ac:<10}): ${ploss / 1e6:6.2f}M")

    if result.graph_shock_nodes:
        print("\nKey Graph Shock Carriers (Contagion):")
        carriers = sorted(result.graph_shock_nodes.items(), key=lambda x: -x[1])[:5]
        for node, intensity in carriers:
            print(f"  * {node:<25} : intensity {intensity:.4f}")

    print("=" * 72)


if __name__ == "__main__":
    main()
