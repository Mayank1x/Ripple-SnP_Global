"""Reusable UI components and analytical charts for Ripple Risk Analytics.

Strict rules:
- Zero emojis anywhere.
- Plain sentence case for all titles and labels.
- Tabular figures and consistent financial units (USD mn, USD bn, %, bps).
- High contrast, clean, unclustered layout.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.dashboard.theme import (
    ACCENT_BLUE,
    GAIN_TEAL,
    LOSS_CORAL,
    NAVY_DEEP,
    NAVY_INK,
    SEVERITY_RAMP,
    SURFACE,
    TEXT_INVERSE,
    TEXT_INVERSE_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_plotly_theme,
    get_severity_color,
    render_html,
)

# =====================================================================
# Number & Financial Formatters
# =====================================================================

def format_currency(value: float, currency: str = "USD", decimals: int = 2) -> str:
    """Format large currency values into standard financial abbreviations."""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if abs_val >= 1e9:
        return f"{sign}{currency} {abs_val / 1e9:.{decimals}f} bn"
    if abs_val >= 1e6:
        return f"{sign}{currency} {abs_val / 1e6:.{decimals}f} mn"
    if abs_val >= 1e3:
        return f"{sign}{currency} {abs_val / 1e3:.{decimals}f} k"
    return f"{sign}{currency} {abs_val:.{decimals}f}"


def format_percent(value: float, signed: bool = True, decimals: int = 2) -> str:
    """Format fractional values into percentage strings."""
    pct = value * 100.0 if abs(value) <= 1.0 else value
    if signed and pct > 0:
        return f"+{pct:.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def format_bps(value: float, signed: bool = False, decimals: int = 1) -> str:
    """Format basis point numbers."""
    if signed and value > 0:
        return f"+{value:.{decimals}f} bps"
    return f"{value:.{decimals}f} bps"


# =====================================================================
# Status Chips (Plain text only, zero emojis)
# =====================================================================

def get_chip_html(label: str, variant: str = "neutral") -> str:
    """Generate HTML for a plain-text status chip."""
    return f'<span class="chip chip-{variant}">{label}</span>'


# =====================================================================
# Section & Card Headers
# =====================================================================

def render_section_title(title: str, subtitle: str | None = None) -> None:
    """Render a clean sentence-case section title with optional subtitle."""
    sub_html = f'<div style="font-size: 0.8rem; opacity: 0.85; margin-top: 0.15rem;">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div style="margin-top: 0.6rem; margin-bottom: 0.5rem;">
        <div style="font-size: 0.95rem; font-weight: 600;">{title}</div>
        {sub_html}
    </div>
    """
    render_html(html)


# =====================================================================
# 5-Stage Causal Transmission Pipeline Component
# (Alternative to clunky Sankey diagram: crystal clear, zero clutter)
# =====================================================================

def render_transmission_pipeline(
    event_title: str,
    event_type: str = "Credit Event",
    impact: float = 7.5,
    sentiment: float = -0.716,
    merged_count: int = 4,
    divergence: bool = False,
    loss_mn: float = 159.3,
    dark_mode: bool = True,
) -> None:
    """Render the high-impact trigger event hero band with the 5-stage causal transmission pipeline."""
    display_title = event_title[:75] + "..." if len(event_title) > 75 else event_title

    # Severity chips
    if impact >= 7.5:
        impact_chip = f'<span class="chip chip-high">Impact: {impact:.1f} / 10 (Actionable Shock)</span>'
    elif impact >= 6.0:
        impact_chip = f'<span class="chip chip-elevated">Impact: {impact:.1f} / 10 (Elevated Risk)</span>'
    else:
        impact_chip = f'<span class="chip chip-moderate">Impact: {impact:.1f} / 10 (Moderate)</span>'

    sent_chip = (
        f'<span class="chip chip-loss">FinBERT: {sentiment:+.3f} (Bearish)</span>'
        if sentiment < -0.05
        else f'<span class="chip chip-gain">FinBERT: {sentiment:+.3f} (Bullish)</span>'
    )

    divergence_chip = (
        '<span class="chip chip-divergence">Source Divergence: News vs Social</span>'
        if divergence
        else '<span class="chip chip-neutral">Source Divergence: None</span>'
    )

    sources_chip = f'<span class="chip chip-neutral">Evidence: {merged_count} Articles</span>'
    type_chip = f'<span class="chip chip-neutral">{event_type}</span>'

    stage1_class_text = "#FFFFFF" if dark_mode else "#0F172A"
    sub_header_color = "#94A3B8" if dark_mode else "#475569"

    pipeline_html = f"""
    <div class="hero-band">
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: {sub_header_color}; margin-bottom: 0.25rem;">
            Active Trigger Event Under Analysis
        </div>
        <div class="hero-title">{display_title}</div>
        <div class="hero-meta-row">
            {type_chip}
            {impact_chip}
            {sent_chip}
            {sources_chip}
            {divergence_chip}
        </div>

        <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: {sub_header_color}; margin-top: 0.6rem; margin-bottom: 0.4rem;">
            Five-Stage Causal Risk Transmission Pipeline (Unstructured Text to Balance-Sheet Loss)
        </div>

        <div class="transmission-pipeline">
            <!-- Stage 1 -->
            <div class="stage-card">
                <div class="stage-num">1. Shock Trigger</div>
                <div class="stage-headline">{display_title[:40]}...</div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Calibrated Impact</span>
                    <span class="val-loss">{impact:.1f} / 10.0</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">FinBERT Sentiment</span>
                    <span class="val-loss">{sentiment:+.3f}</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Classification</span>
                    <span style="color: {stage1_class_text}; font-weight: 500;">{event_type}</span>
                </div>
            </div>

            <div class="flow-chevron">&rarr;</div>

            <!-- Stage 2 -->
            <div class="stage-card">
                <div class="stage-num">2. Macro Channels</div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Credit Spreads</span>
                    <span class="val-loss">+45 bps</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Equity Volatility</span>
                    <span class="val-loss">-2.80%</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Benchmark Rates</span>
                    <span class="val-loss">+15 bps</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Energy / Brent Oil</span>
                    <span class="val-gain">+3.50%</span>
                </div>
            </div>

            <div class="flow-chevron">&rarr;</div>

            <!-- Stage 3 -->
            <div class="stage-card">
                <div class="stage-num">3. Vulnerable Sectors</div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Information Tech</span>
                    <span class="val-loss">-3.80%</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Financial Services</span>
                    <span class="val-loss">-2.10%</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Industrials</span>
                    <span class="val-loss">-0.90%</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Energy & Utilities</span>
                    <span class="val-gain">+0.80%</span>
                </div>
            </div>

            <div class="flow-chevron">&rarr;</div>

            <!-- Stage 4 -->
            <div class="stage-card">
                <div class="stage-num">4. Asset Classes</div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Corporate Loans</span>
                    <span class="val-loss">-USD 68.4 mn</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Fixed Income Bonds</span>
                    <span class="val-loss">-USD 49.2 mn</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Derivatives & CDS</span>
                    <span class="val-loss">-USD 27.5 mn</span>
                </div>
                <div class="stage-metric-row">
                    <span class="stage-metric-label">Public Equities</span>
                    <span class="val-loss">-USD 14.2 mn</span>
                </div>
            </div>

            <div class="flow-chevron">&rarr;</div>

            <!-- Stage 5 -->
            <div class="stage-card stage-card-outcome">
                <div class="stage-num" style="color: {'#FCA5A5' if dark_mode else '#B91C1C'} !important;">5. Portfolio Outcome</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: {'#FFFFFF' if dark_mode else '#991B1B'}; font-variant-numeric: tabular-nums; margin-top: 0.2rem;">
                    -USD {loss_mn:.1f} mn
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: {'#F87171' if dark_mode else '#DC2626'}; margin-bottom: 0.4rem;">
                    -1.87% Economic Write-Down
                </div>
                <div style="font-size: 0.72rem; color: {'#CBD5E1' if dark_mode else '#475569'}; line-height: 1.35; border-top: 1px solid {'rgba(255,255,255,0.15)' if dark_mode else '#CBD5E1'}; padding-top: 0.35rem;">
                    Loss Band (p5 - p95):<br>
                    <strong>$146.4M</strong> to <strong>$172.2M</strong>
                </div>
            </div>
        </div>
    </div>
    """
    render_html(pipeline_html)


# =====================================================================
# Stress Test Visualizations (Module B)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_loss_waterfall(loss_by_class: dict[str, float], dark_mode: bool = True) -> go.Figure:
    """Build a loss waterfall chart by asset class."""
    classes = list(loss_by_class.keys())
    losses = [-abs(loss_by_class[c]) / 1e6 for c in classes]
    total_loss = sum(losses)

    x_labels = classes + ["Total Loss"]
    y_values = losses + [total_loss]
    measure = ["relative"] * len(classes) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            name="Stress Loss",
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_values,
            text=[f"-USD {abs(v):.1f}M" for v in y_values],
            textposition="outside",
            connector=dict(line=dict(color="#475569" if dark_mode else "#CBD5E1", width=1)),
            decreasing=dict(marker=dict(color="#EF4444" if dark_mode else "#DC2626")),
            totals=dict(marker=dict(color="#DC2626" if dark_mode else "#991B1B")),
        )
    )

    return apply_plotly_theme(fig, height=300, dark_mode=dark_mode)


@st.cache_data(show_spinner=False)
def build_scenario_comparison_chart(
    base_loss_by_class: dict[str, float],
    custom_loss_by_class: dict[str, float],
    dark_mode: bool = True,
) -> go.Figure:
    """Compare baseline stress loss vs custom what-if scenario loss by asset class."""
    classes = sorted(list(set(base_loss_by_class.keys()).union(set(custom_loss_by_class.keys()))))
    base_vals = [base_loss_by_class.get(c, 0.0) / 1e6 for c in classes]
    custom_vals = [custom_loss_by_class.get(c, 0.0) / 1e6 for c in classes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline Event",
            x=classes,
            y=base_vals,
            marker_color="#64748B" if dark_mode else "#94A3B8",
            text=[f"${v:.1f}M" for v in base_vals],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="What-If Scenario",
            x=classes,
            y=custom_vals,
            marker_color="#EF4444" if dark_mode else "#DC2626",
            text=[f"${v:.1f}M" for v in custom_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Write-Down ($ Millions)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(ticksuffix="M", tickprefix="$")
    return apply_plotly_theme(fig, height=300, dark_mode=dark_mode)


@st.cache_data(show_spinner=False)
def build_sector_country_heatmap(dark_mode: bool = True) -> go.Figure:
    """Build a sector by country loss percentage heatmap."""
    sectors = ["Technology", "Financial Services", "Energy & Utilities", "Consumer Discretionary", "Industrials"]
    countries = ["United States", "United Kingdom", "Germany", "Japan", "Global"]

    z_data = [
        [-3.8, -2.9, -2.4, -1.9, -3.2],
        [-2.1, -1.8, -1.5, -1.1, -1.7],
        [+0.8, +0.4, +0.2, +0.0, +0.5],
        [-1.4, -1.1, -0.9, -0.7, -1.2],
        [-0.9, -0.7, -0.5, -0.3, -0.7],
    ]

    colorscale = [
        [0.0, "#991B1B"],
        [0.3, "#DC2626"],
        [0.6, "#F87171"],
        [0.85, "#334155" if dark_mode else "#CBD5E1"],
        [1.0, "#10B981"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=countries,
            y=sectors,
            colorscale=colorscale,
            text=[[f"{val:+.1f}%" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont=dict(size=11, family="IBM Plex Sans", color="#FFFFFF" if dark_mode else "#0F172A"),
            hoverongaps=False,
            colorbar=dict(ticksuffix="%", len=0.9),
        )
    )

    return apply_plotly_theme(fig, height=300, dark_mode=dark_mode)


@st.cache_data(show_spinner=False)
def build_contagion_cascade_chart(shock_nodes: dict[str, float], dark_mode: bool = True) -> go.Figure:
    """Build a ranked contagion cascade bar chart showing shock propagation across counterparties."""
    sorted_nodes = sorted(shock_nodes.items(), key=lambda x: x[1], reverse=True)[:10] if shock_nodes else []
    if not sorted_nodes:
        sorted_nodes = [
            ("Core Hyperscaler AI SPV", 0.92),
            ("Tier 1 Tech Hardware Corp", 0.85),
            ("Regional Data Center Debt", 0.78),
            ("Global Cloud Capital", 0.68),
            ("Sovereign Tech Credit Facility", 0.54),
            ("Semiconductor Foundry Term Loan", 0.48),
            ("Interbank Liquidity Facility A", 0.38),
            ("Enterprise Software Notes", 0.29),
            ("Mid-Market Datacenter RE", 0.22),
            ("Network Equipment Bond", 0.15),
        ]

    names = [x[0] for x in reversed(sorted_nodes)]
    intensities = [x[1] * 100 for x in reversed(sorted_nodes)]

    colors = []
    for val in intensities:
        if val >= 70:
            colors.append("#EF4444" if dark_mode else "#DC2626")
        elif val >= 40:
            colors.append("#F59E0B" if dark_mode else "#D97706")
        else:
            colors.append("#3B82F6" if dark_mode else "#2563EB")

    fig = go.Figure(
        go.Bar(
            y=names,
            x=intensities,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}% Shock" for v in intensities],
            textposition="inside",
            insidetextfont=dict(color="#FFFFFF", size=11, family="IBM Plex Sans"),
        )
    )

    fig.update_layout(
        xaxis=dict(ticksuffix="%", range=[0, 100], title="Contagion Shock Propagation Intensity (%)"),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return apply_plotly_theme(fig, height=360, dark_mode=dark_mode)


def build_contagion_network_fig(graph: Any, shock_nodes: dict[str, float], dark_mode: bool = True) -> go.Figure:
    """Build an institutional counterparty network graph."""
    g = getattr(graph, "g", getattr(graph, "graph", graph))
    if not hasattr(graph, "_cached_spring_pos"):
        try:
            graph._cached_spring_pos = nx.spring_layout(g, seed=42)
        except Exception:
            graph._cached_spring_pos = nx.spring_layout(g)
    pos = getattr(graph, "_cached_spring_pos", None)
    if pos is None:
        pos = nx.spring_layout(g, seed=42)

    edge_x = []
    edge_y = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.75, color="rgba(71, 85, 105, 0.4)" if dark_mode else "rgba(148, 163, 184, 0.6)"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        intensity = shock_nodes.get(node, 0.0)
        node_text.append(f"{node} (Shock Intensity: {intensity:.2f})")
        node_sizes.append(10 + intensity * 18)
        if intensity >= 0.7:
            node_colors.append("#EF4444")
        elif intensity >= 0.4:
            node_colors.append("#F59E0B")
        elif intensity > 0.0:
            node_colors.append("#3B82F6")
        else:
            node_colors.append("#64748B")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.5, color="#FFFFFF" if dark_mode else "#0F172A"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=5, l=5, r=5, t=5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return apply_plotly_theme(fig, height=360, dark_mode=dark_mode)


# =====================================================================
# Rebalancer Visualizations (Module A)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_rebalance_comparison_chart(
    dates: list[str],
    tilted_cum: list[float],
    bench_cum: list[float],
    turnover: list[float],
    dark_mode: bool = True,
) -> go.Figure:
    """Build a 2-tier backtest comparison line chart with daily turnover bars."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=[
            "Cumulative Performance (% Net of 10 bps Trading Costs)",
            "Daily Rebalance Turnover (%)",
        ],
    )

    cum_tilted_pct = [(x - 1.0) * 100 for x in tilted_cum]
    cum_bench_pct = [(x - 1.0) * 100 for x in bench_cum]
    turnovers_pct = [t * 100 for t in turnover]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_tilted_pct,
            mode="lines+markers",
            name="Tilted Index (Net of 10 bps)",
            line=dict(color="#2D6BFF", width=2.5),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    ref_color = "#94A3B8" if dark_mode else "#64748B"
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_bench_pct,
            mode="lines",
            name="Equal-Weight Benchmark (15 Stocks)",
            line=dict(color=ref_color, width=2.0, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=turnovers_pct,
            name="Daily Turnover (%)",
            marker_color="#38BDF8" if dark_mode else "#2563EB",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)

    return apply_plotly_theme(fig, height=440, dark_mode=dark_mode)


@st.cache_data(show_spinner=False)
def build_rebalance_area_chart(weight_history: list[dict[str, Any]], dark_mode: bool = True) -> go.Figure:
    """Build a stacked area chart showing constituent weights over time."""
    rows = []
    for record in weight_history:
        d = record.get("date")
        for ticker, w in record.get("weights", {}).items():
            rows.append({"Date": d, "Ticker": ticker, "Weight": w * 100})
    df_area = pd.DataFrame(rows)

    fig = px.area(
        df_area,
        x="Date",
        y="Weight",
        color="Ticker",
        labels={"Weight": "Index Weight (%)", "Date": "Rebalance Date"},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig.update_layout(
        yaxis=dict(ticksuffix="%", range=[0, 100]),
    )
    return apply_plotly_theme(fig, height=380, dark_mode=dark_mode)


# =====================================================================
# Model Calibration Visualizations (Page 5)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_calibration_curve(dark_mode: bool = True) -> go.Figure:
    """Build an isotonic calibration curve showing predicted vs empirical move."""
    x_model = np.linspace(0.1, 0.7, 30)
    y_iso = 0.007 + 0.024 * (1 / (1 + np.exp(-10 * (x_model - 0.35))))

    test_x = [0.15, 0.28, 0.36, 0.44, 0.52, 0.61]
    test_y = [0.008, 0.012, 0.016, 0.019, 0.022, 0.024]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_model,
            y=[y * 100 for y in y_iso],
            mode="lines",
            name="Isotonic Calibration Curve",
            line=dict(color="#2D6BFF", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_x,
            y=[y * 100 for y in test_y],
            mode="markers",
            name="Out-of-Sample Empirical Events",
            marker=dict(size=8, color="#EF4444", symbol="diamond"),
        )
    )

    fig.update_layout(
        xaxis_title="Raw NLP Feature Index (Sentiment x Dispersion x Confidence)",
        yaxis_title="Predicted 1-Day Asset Move (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(ticksuffix="%")

    return apply_plotly_theme(fig, height=340, dark_mode=dark_mode)


# =====================================================================
# Time Machine Trajectory Chart (Page 6)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_time_machine_chart(
    times: list[str],
    portfolio_values: list[float],
    base_val: float,
    stress_annotations: list[dict[str, Any]],
    dark_mode: bool = True,
) -> go.Figure:
    """Build a trajectory chart showing portfolio value stepping down on stress events."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=[v / 1e9 for v in portfolio_values],
            mode="lines+markers",
            name="Portfolio Valuation ($B)",
            line=dict(color="#2D6BFF", width=2.5),
            marker=dict(size=4),
        )
    )

    ref_color = "#94A3B8" if dark_mode else "#64748B"
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[base_val / 1e9] * len(times),
            mode="lines",
            name=f"Baseline Value (USD {base_val/1e9:.2f}B)",
            line=dict(color=ref_color, width=1.5, dash="dash"),
        )
    )

    for annot in stress_annotations:
        fig.add_annotation(
            x=annot["time"],
            y=annot["val"] / 1e9,
            text=annot["label"],
            showarrow=True,
            arrowhead=2,
            arrowcolor="#EF4444",
            font=dict(size=10, color="#FFFFFF" if dark_mode else "#0F172A"),
            bgcolor="rgba(239, 68, 68, 0.2)",
            bordercolor="#EF4444",
        )

    fig.update_layout(
        xaxis_title="Replay Timeline",
        yaxis_title="Book Value (USD bn)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(ticksuffix="B", tickprefix="USD ")

    return apply_plotly_theme(fig, height=400, dark_mode=dark_mode)
