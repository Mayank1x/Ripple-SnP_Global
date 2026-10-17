"""Generate the system architecture diagram for Ripple.

Saves high-resolution PNG to docs/architecture.png using matplotlib.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_architecture_diagram(output_path: str = "docs/architecture.png") -> None:
    """Generate high-resolution architecture flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background color
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Header Title
    ax.text(
        50, 97.0,
        "Ripple: AI/NLP Risk Engine & Strategic Stress-Testing Platform",
        ha="center", va="center",
        fontsize=19, fontweight="bold", color="#0f172a",
        family="sans-serif",
    )
    ax.text(
        50, 94.2,
        "S&P Global & CRISIL Campus Hackathon 2026 — End-to-End System Architecture",
        ha="center", va="center",
        fontsize=11.5, color="#475569",
        family="sans-serif",
    )

    def draw_card(x, y, w, h, title, subtitle, bg_color, border_color, items=None, text_color="#1e293b"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h - 2.5, title,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=border_color,
            zorder=3, family="sans-serif",
        )
        if subtitle:
            ax.text(
                x + w / 2, y + h - 5.2, subtitle,
                ha="center", va="center",
                fontsize=8.5, fontstyle="italic", color="#64748b",
                zorder=3, family="sans-serif",
            )
        if items:
            start_y = y + h - (7.6 if subtitle else 5.4)
            for i, itm in enumerate(items):
                ax.text(
                    x + 1.8, start_y - (i * 2.8), f"• {itm}",
                    ha="left", va="center",
                    fontsize=8.7, color=text_color,
                    zorder=3, family="sans-serif",
                )

    def draw_arrow(x1, y1, x2, y2, label=""):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            color="#334155",
            linewidth=1.8,
            zorder=4,
        )
        ax.add_patch(arrow)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(
                mid_x, mid_y + 1.4, label,
                ha="center", va="center",
                fontsize=7.8, fontweight="bold", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.8),
                zorder=5, family="sans-serif",
            )

    # 1. LAYER 1: INGESTION (Top Left, X: 3 to 31, Y: 68 to 91)
    draw_card(
        3, 67, 28, 24,
        "1. Ingestion Engine",
        "Multi-source raw financial feeds",
        "#f1f5f9", "#0284c7",
        [
            "GDELT 2.0 API (Global events / tone)",
            "NewsAPI & Financial RSS feeds",
            "Kaggle Tweets (Social sentiment feed)",
            "Deterministic disk caching for replay",
            "Auto-fallback on HTTP 429 rate limits",
        ],
    )

    # 2. LAYER 2: NLP PIPELINE (Top Center, X: 36 to 64, Y: 67 to 91)
    draw_card(
        36, 67, 28, 24,
        "2. NLP Risk Engine",
        "Domain sentiment & classification",
        "#f8fafc", "#0d9488",
        [
            "Deduplication (Exact hash & Jaccard)",
            "FinBERT Model (ProsusAI, [-1, +1])",
            "NER Mapping (Tickers, Sectors, GPE)",
            "Gemini 2.0 Flash / Local Rule Fallback",
            "10 Financial Crisis Taxonomies",
        ],
    )

    # 3. LAYER 3: STRESS ENGINE - MODULE B (Top Right, X: 69 to 97, Y: 53 to 91)
    draw_card(
        69, 53, 28, 38,
        "4. Strategic Stress Engine",
        "Module B Portfolio Revaluation",
        "#fef2f2", "#dc2626",
        [
            "Trigger Rule: impact_score >= 7.0",
            "Analog Retrieval: FAISS crisis index",
            "Bayesian Shrinkage toward prior table",
            "Empirical Factor Shocks (p5, p50, p95)",
            "NetworkX Contagion Propagation",
            "Loans: dPD = PD * exp(k * intensity)",
            "Bonds: dP = -ModDur*dy - SpreadDur*ds",
            "Derivatives: FX, IRS (DV01), Commodity",
            "Multi-Asset Aggregation (EL & MTM loss)",
            "Auditable JSON outputs (data/stress_runs/)",
        ],
    )

    # 4. STORAGE & AUDIT (Mid Left, X: 3 to 31, Y: 39 to 63)
    draw_card(
        3, 39, 28, 24,
        "Persistence & Audit Store",
        "Queryable signal database",
        "#fdf4ff", "#a855f7",
        [
            "SQLite Signal Store (signals.db)",
            "JSONL Immutable Stream (signals.jsonl)",
            "Fast Querying by Entity & Sector",
            "Calibration metadata & divergence tags",
            "Ingestion health status (banner payload)",
        ],
    )

    # 5. SCORING & CALIBRATION (Mid Center, X: 36 to 64, Y: 39 to 63)
    draw_card(
        36, 39, 28, 24,
        "3. Scoring & Calibration",
        "Clustering & Empirical Isotonic Fit",
        "#f8fafc", "#d97706",
        [
            "FAISS Embeddings (all-MiniLM-L6-v2)",
            "Rolling 7-Day Temporal Clustering",
            "Source Divergence Flag (News vs Social)",
            "Event Study Market Model (^GSPC / ^NSEI)",
            "Isotonic Calibrator (|CAR| -> Score 1-10)",
        ],
    )

    # 6. DASHBOARD & UI (Bottom Left, X: 3 to 51, Y: 6 to 34)
    draw_card(
        3, 6, 48, 28,
        "5. Interactive Analytics Dashboard",
        "Streamlit + Plotly (100% Offline Capable)",
        "#eff6ff", "#2563eb",
        [
            "Live Signals Feed: Real-time cards, divergence tags, and source evidence",
            "Stress Test View: Before/after $8.54B book, p5-p95 confidence bands & waterfall",
            "Contagion Map: Interactive network graph with dynamic node shock sizing",
            "Model Check: Calibration curve (|CAR| vs Score) and classification metrics",
            "Time Machine Replay: Historical timeline slider with active portfolio stress",
            "Fallback Banner: Alerts user when live API degrades to offline cache",
        ],
    )

    # 7. DELIVERY & CONSUMPTION (Bottom Right, X: 54 to 97, Y: 6 to 34)
    draw_card(
        54, 6, 43, 28,
        "6. Delivery & Consumption Layer",
        "FastAPI Service & Wholesale Portfolio Book",
        "#f0fdf4", "#16a34a",
        [
            "FastAPI REST Microservice (uvicorn)",
            "GET /signals: Query by date, entity, min_impact",
            "GET /signals/{id}: Retrieve signal payload & evidence",
            "GET /health: System health and component status",
            "Wholesale Banking Portfolio (180 positions: Loans, Bonds, Swaps)",
            "One-Command Master CLI: python run.py --mode replay / --mode live",
        ],
    )

    # Connective Dataflow Arrows
    # 1. Ingestion -> NLP
    draw_arrow(31, 79, 36, 79, "Raw Docs")
    # 2. NLP -> Scoring
    draw_arrow(50, 67, 50, 63, "Analysed Docs")
    # 3. Scoring -> Storage
    draw_arrow(36, 51, 31, 51, "Signals")
    # 4. Scoring -> Stress Engine
    draw_arrow(64, 51, 69, 65, "Score >= 7.0")
    # 5. Storage -> Dashboard
    draw_arrow(17, 39, 17, 34, "Signal Stream")
    # 6. Stress -> Dashboard
    draw_arrow(73, 53, 48, 34, "Stress Results")
    # 7. Storage / Portfolio -> Delivery Layer
    draw_arrow(75, 53, 75, 34, "Portfolio Book")

    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Architecture diagram successfully generated and saved to: {output_file.resolve()}")


if __name__ == "__main__":
    create_architecture_diagram()
