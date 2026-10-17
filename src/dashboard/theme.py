"""Theme tokens, custom CSS injection, and Plotly templates for Ripple Risk Analytics.

Strict design rules:
- Zero emojis anywhere.
- Flawless contrast in both Light and Dark institutional modes.
- Self-hosted IBM Plex Sans fonts with tabular numbers.
- Fixed chart geometries and zero layout shifts.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# =====================================================================
# Light Theme Tokens
# =====================================================================

LIGHT_CANVAS = "#F1F5F9"
LIGHT_SURFACE = "#FFFFFF"
LIGHT_SURFACE_SUBTLE = "#F8FAFC"
LIGHT_BORDER_STRONG = "#CBD5E1"
LIGHT_BORDER_SUBTLE = "#E2E8F0"
LIGHT_TEXT_PRIMARY = "#0F172A"
LIGHT_TEXT_SECONDARY = "#475569"

# =====================================================================
# Dark Theme Tokens (High Contrast)
# =====================================================================

DARK_CANVAS = "#070E1A"
DARK_SURFACE = "#0E1A2E"
DARK_SURFACE_SUBTLE = "#14243D"
DARK_BORDER_STRONG = "#2B4366"
DARK_BORDER_SUBTLE = "#1E355B"
DARK_TEXT_PRIMARY = "#F8FAFC"
DARK_TEXT_SECONDARY = "#94A3B8"

# Shared Accent Tokens
NAVY_INK = "#0B1B33"
NAVY_DEEP = "#14305A"
ACCENT_BLUE = "#2D6BFF"
GAIN_TEAL = "#10B981"
LOSS_CORAL = "#EF4444"
DIVERGENCE_AMBER = "#F59E0B"

# Default Aliases
CANVAS = LIGHT_CANVAS
SURFACE = LIGHT_SURFACE
SURFACE_SUBTLE = LIGHT_SURFACE_SUBTLE
BORDER_STRONG = LIGHT_BORDER_STRONG
BORDER_SUBTLE = LIGHT_BORDER_SUBTLE
TEXT_PRIMARY = LIGHT_TEXT_PRIMARY
TEXT_SECONDARY = LIGHT_TEXT_SECONDARY
TEXT_INVERSE = "#FFFFFF"
TEXT_INVERSE_MUTED = "#94A3B8"


SEVERITY_RAMP = {
    "low": "#64748B",
    "moderate": "#334155",
    "elevated": "#D97706",
    "high": "#E5484D",
}


def get_severity_color(impact: float) -> str:
    """Return sequential ramp color for calibrated impact score."""
    if impact >= 7.5:
        return SEVERITY_RAMP["high"]
    if impact >= 6.0:
        return SEVERITY_RAMP["elevated"]
    if impact >= 4.0:
        return SEVERITY_RAMP["moderate"]
    return SEVERITY_RAMP["low"]


def get_sentiment_color(sentiment: float) -> str:
    """Return color for signed sentiment."""
    if sentiment > 0.05:
        return GAIN_TEAL
    if sentiment < -0.05:
        return LOSS_CORAL
    return "#64748B"


# =====================================================================
# Font Loader (Offline Base64 Encoding)
# =====================================================================

@st.cache_data
def load_font_face_css() -> str:
    """Load local IBM Plex Sans TTF files and encode as base64 data URIs."""
    fonts_dir = Path(__file__).parent / "static" / "fonts"
    css_rules = []
    font_files = {
        ("400", "normal"): fonts_dir / "ibm-plex-sans-400.ttf",
        ("500", "normal"): fonts_dir / "ibm-plex-sans-500.ttf",
        ("600", "normal"): fonts_dir / "ibm-plex-sans-600.ttf",
    }

    for (weight, style), font_path in font_files.items():
        if font_path.exists():
            try:
                b64_data = base64.b64encode(font_path.read_bytes()).decode("utf-8")
                css_rules.append(
                    f"""
                    @font-face {{
                        font-family: 'IBM Plex Sans';
                        font-style: {style};
                        font-weight: {weight};
                        font-display: swap;
                        src: url('data:font/truetype;charset=utf-8;base64,{b64_data}') format('truetype');
                    }}
                    """
                )
            except Exception:
                pass

    return "\n".join(css_rules)


# =====================================================================
# Dynamic Theme CSS Injection
# =====================================================================

def inject_theme(dark_mode: bool = True) -> None:
    """Inject institutional CSS styles with strict contrast controls."""
    font_css = load_font_face_css()

    c_canvas = DARK_CANVAS if dark_mode else LIGHT_CANVAS
    c_surface = DARK_SURFACE if dark_mode else LIGHT_SURFACE
    c_surface_subtle = DARK_SURFACE_SUBTLE if dark_mode else LIGHT_SURFACE_SUBTLE
    c_border_strong = DARK_BORDER_STRONG if dark_mode else LIGHT_BORDER_STRONG
    c_border_subtle = DARK_BORDER_SUBTLE if dark_mode else LIGHT_BORDER_SUBTLE
    c_text_primary = DARK_TEXT_PRIMARY if dark_mode else LIGHT_TEXT_PRIMARY
    c_text_secondary = DARK_TEXT_SECONDARY if dark_mode else LIGHT_TEXT_SECONDARY

    custom_css = f"""
    <style>
    {font_css}

    /* Hide Streamlit default chrome */
    #MainMenu, footer, header, div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
        visibility: hidden !important;
        display: none !important;
    }}

    /* Global typography and canvas */
    html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: {c_canvas} !important;
        color: {c_text_primary} !important;
        font-feature-settings: 'tnum' 1, 'cv05' 1 !important;
    }}

    /* Explicit text contrast overrides */
    p, span, div, h1, h2, h3, h4, h5, h6, label {{
        color: {c_text_primary};
    }}

    /* Force tabular numerals for all financial data */
    .tabular-num, td, th, [data-testid="stMetricValue"], [data-testid="stDataFrame"] {{
        font-variant-numeric: tabular-nums !important;
    }}

    /* Container Spacing */
    .block-container {{
        padding-top: 0.75rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}

    /* Top Application Bar */
    .app-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: {c_surface};
        color: {c_text_primary};
        padding: 0.65rem 1.25rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
        border-left: 4px solid {ACCENT_BLUE};
        border-right: 1px solid {c_border_subtle};
        border-top: 1px solid {c_border_subtle};
        border-bottom: 1px solid {c_border_subtle};
    }}

    /* Navigation Radio Pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        gap: 0.35rem !important;
        background-color: {c_surface} !important;
        padding: 0.35rem 0.6rem !important;
        border-radius: 4px !important;
        border: 1px solid {c_border_subtle} !important;
    }}
    div[data-testid="stRadio"] label p {{
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        color: {c_text_secondary} !important;
    }}

    /* Hero Band Container */
    .hero-band {{
        background-color: {DARK_SURFACE if dark_mode else "#0B1B33"};
        border: 1px solid {DARK_BORDER_SUBTLE if dark_mode else "#14305A"};
        border-radius: 4px;
        padding: 1.25rem 1.5rem;
        color: #FFFFFF !important;
        margin-bottom: 0.85rem;
    }}
    .hero-band p, .hero-band span, .hero-band div {{
        color: #FFFFFF;
    }}
    .hero-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem;
        line-height: 1.35;
    }}
    .hero-meta-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        align-items: center;
    }}

    /* Transmission Pipeline Grid */
    .transmission-pipeline {{
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr auto 1.15fr;
        gap: 0.5rem;
        align-items: stretch;
        margin-top: 0.5rem;
    }}
    .stage-card {{
        background-color: {"rgba(20, 36, 61, 0.7)" if dark_mode else "rgba(255, 255, 255, 0.08)"};
        border: 1px solid {"rgba(43, 67, 102, 0.6)" if dark_mode else "rgba(255, 255, 255, 0.15)"};
        border-radius: 4px;
        padding: 0.75rem 0.85rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .stage-card-outcome {{
        background-color: {"rgba(239, 68, 68, 0.14)" if dark_mode else "rgba(229, 72, 77, 0.2)"};
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .stage-num {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8 !important;
        margin-bottom: 0.35rem;
    }}
    .stage-headline {{
        font-size: 0.82rem;
        font-weight: 600;
        color: #FFFFFF !important;
        line-height: 1.3;
        margin-bottom: 0.5rem;
    }}
    .stage-metric-row {{
        display: flex;
        justify-content: space-between;
        font-size: 0.76rem;
        padding: 0.18rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .stage-metric-label {{
        color: #CBD5E1 !important;
    }}
    .val-loss {{
        color: #F87171 !important;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}
    .val-gain {{
        color: #34D399 !important;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}
    .flow-chevron {{
        display: flex;
        align-items: center;
        justify-content: center;
        color: #64748B;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0 0.1rem;
    }}

    /* White / Dark Content Card */
    .white-card {{
        background-color: {c_surface};
        border: 1px solid {c_border_subtle};
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }}
    .card-title {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {c_text_primary} !important;
        margin-bottom: 0.65rem;
        border-bottom: 1px solid {c_border_subtle};
        padding-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* Status Chips */
    .chip {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.18rem 0.55rem;
        border-radius: 3px;
        letter-spacing: 0.1px;
    }}
    .chip-high {{
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .chip-elevated {{
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24 !important;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }}
    .chip-moderate {{
        background-color: rgba(100, 116, 139, 0.2);
        color: #CBD5E1 !important;
        border: 1px solid rgba(100, 116, 139, 0.4);
    }}
    .chip-divergence {{
        background-color: rgba(245, 158, 11, 0.22);
        color: #FCD34D !important;
        border: 1px solid #F59E0B;
    }}
    .chip-gain {{
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    .chip-loss {{
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .chip-neutral {{
        background-color: rgba(148, 163, 184, 0.15);
        color: #E2E8F0 !important;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }}

    /* Streamlit Metrics */
    div[data-testid="stMetric"] {{
        background-color: {c_surface} !important;
        border: 1px solid {c_border_subtle} !important;
        border-radius: 4px !important;
        padding: 0.65rem 0.85rem !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: {c_text_secondary} !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: {c_text_primary} !important;
    }}
    div[data-testid="stMetricDelta"] {{
        font-size: 0.75rem !important;
    }}

    /* Tables & Dataframes */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {c_border_subtle} !important;
        border-radius: 4px !important;
        background-color: {c_surface} !important;
    }}

    /* Inputs, Selectboxes, Sliders */
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {c_surface} !important;
        color: {c_text_primary} !important;
        border-color: {c_border_strong} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {c_surface} !important;
        color: {c_text_primary} !important;
        border: 1px solid {c_border_strong} !important;
        border-radius: 3px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.75rem !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {c_surface} !important;
        border: 1px solid {c_border_subtle} !important;
        border-radius: 4px !important;
        margin-bottom: 0.6rem !important;
    }}
    div[data-testid="stExpander"] summary span {{
        color: {c_text_primary} !important;
    }}

    /* Dividers */
    hr {{
        border: none !important;
        border-top: 1px solid {c_border_subtle} !important;
        margin: 0.85rem 0 !important;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =====================================================================
# Plotly Custom Institutional Template (Light & Dark)
# =====================================================================

def get_plotly_layout_defaults(dark_mode: bool = True) -> dict[str, Any]:
    """Return standard layout dictionary for institutional Plotly figures."""
    c_text = DARK_TEXT_PRIMARY if dark_mode else LIGHT_TEXT_PRIMARY
    c_text_sec = DARK_TEXT_SECONDARY if dark_mode else LIGHT_TEXT_SECONDARY
    c_grid = DARK_BORDER_SUBTLE if dark_mode else LIGHT_BORDER_SUBTLE
    c_line = DARK_BORDER_STRONG if dark_mode else LIGHT_BORDER_STRONG

    return dict(
        font=dict(
            family="IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            size=12,
            color=c_text,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            showgrid=True,
            gridcolor=c_grid,
            gridwidth=1,
            linecolor=c_line,
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=11, color=c_text_sec),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=c_grid,
            gridwidth=1,
            linecolor=c_line,
            linewidth=1,
            zeroline=True,
            zerolinecolor=c_line,
            zerolinewidth=1,
            tickfont=dict(size=11, color=c_text_sec),
        ),
        hoverlabel=dict(
            bgcolor=DARK_SURFACE if dark_mode else NAVY_INK,
            font_size=12,
            font_family="IBM Plex Sans, sans-serif",
            font_color="#FFFFFF",
            bordercolor=DARK_BORDER_STRONG if dark_mode else NAVY_DEEP,
        ),
    )


def apply_plotly_theme(fig: go.Figure, height: int = 340, dark_mode: bool = True) -> go.Figure:
    """Apply custom institutional theme and fixed geometry to a Plotly figure."""
    defaults = get_plotly_layout_defaults(dark_mode=dark_mode)
    fig.update_layout(
        **defaults,
        height=height,
    )
    return fig
