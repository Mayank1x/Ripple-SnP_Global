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
# Light Theme Tokens (High Contrast Financial Terminal)
# =====================================================================

LIGHT_CANVAS = "#F1F5F9"
LIGHT_SURFACE = "#FFFFFF"
LIGHT_SURFACE_SUBTLE = "#F8FAFC"
LIGHT_BORDER_STRONG = "#94A3B8"
LIGHT_BORDER_SUBTLE = "#CBD5E1"
LIGHT_TEXT_PRIMARY = "#0F172A"
LIGHT_TEXT_SECONDARY = "#334155"
LIGHT_TEXT_MUTED = "#64748B"

# =====================================================================
# Dark Theme Tokens (Deep Navy / Slate Institutional)
# =====================================================================

DARK_CANVAS = "#070E1A"
DARK_SURFACE = "#0E1A2E"
DARK_SURFACE_SUBTLE = "#14243D"
DARK_BORDER_STRONG = "#2B4366"
DARK_BORDER_SUBTLE = "#1E355B"
DARK_TEXT_PRIMARY = "#F8FAFC"
DARK_TEXT_SECONDARY = "#94A3B8"
DARK_TEXT_MUTED = "#64748B"

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
# Clean HTML Renderer Helper
# Ensures zero leading whitespace on any line to prevent Markdown code blocks
# =====================================================================

def render_html(html: str) -> None:
    """Render clean HTML without markdown indentation interpretation."""
    cleaned = "\n".join(line.strip() for line in html.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# =====================================================================
# Font Loader (Offline Base64 Encoding)
# =====================================================================

def load_font_face_css() -> str:
    """Return lightweight font-face declarations pointing to static served fonts with system fallbacks."""
    return """
@font-face {
  font-family: 'IBM Plex Sans';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url('/app/static/fonts/ibm-plex-sans-400.ttf') format('truetype'),
       local('IBM Plex Sans'),
       local('Segoe UI');
}
@font-face {
  font-family: 'IBM Plex Sans';
  font-style: normal;
  font-weight: 500;
  font-display: swap;
  src: url('/app/static/fonts/ibm-plex-sans-500.ttf') format('truetype'),
       local('IBM Plex Sans Medium'),
       local('Segoe UI Semibold');
}
@font-face {
  font-family: 'IBM Plex Sans';
  font-style: normal;
  font-weight: 600;
  font-display: swap;
  src: url('/app/static/fonts/ibm-plex-sans-600.ttf') format('truetype'),
       local('IBM Plex Sans SemiBold'),
       local('Segoe UI Bold');
}
"""



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
    c_text_muted = DARK_TEXT_MUTED if dark_mode else LIGHT_TEXT_MUTED

    # Chips colors tailored for maximum contrast in each mode
    if dark_mode:
        chip_high_bg = "rgba(239, 68, 68, 0.22)"
        chip_high_border = "#EF4444"
        chip_high_color = "#FCA5A5"

        chip_elev_bg = "rgba(245, 158, 11, 0.22)"
        chip_elev_border = "#F59E0B"
        chip_elev_color = "#FCD34D"

        chip_mod_bg = "rgba(100, 116, 139, 0.25)"
        chip_mod_border = "#64748B"
        chip_mod_color = "#E2E8F0"

        chip_gain_bg = "rgba(16, 185, 129, 0.22)"
        chip_gain_border = "#10B981"
        chip_gain_color = "#6EE7B7"

        chip_loss_bg = "rgba(239, 68, 68, 0.22)"
        chip_loss_border = "#EF4444"
        chip_loss_color = "#FCA5A5"

        chip_neut_bg = "rgba(30, 53, 91, 0.65)"
        chip_neut_border = "#2B4366"
        chip_neut_color = "#E2E8F0"

        stage_card_bg = "#14243D"
        stage_card_border = "#2B4366"
        stage_num_color = "#94A3B8"
        stage_headline_color = "#FFFFFF"
        stage_label_color = "#CBD5E1"
        stage_border_color = "rgba(255, 255, 255, 0.08)"
        stage_outcome_bg = "rgba(239, 68, 68, 0.16)"
        stage_outcome_border = "rgba(239, 68, 68, 0.45)"
        stage_outcome_text = "#FFFFFF"
        stage_outcome_sub = "#F87171"
    else:
        chip_high_bg = "#FEE2E2"
        chip_high_border = "#EF4444"
        chip_high_color = "#991B1B"

        chip_elev_bg = "#FEF3C7"
        chip_elev_border = "#F59E0B"
        chip_elev_color = "#92400E"

        chip_mod_bg = "#F1F5F9"
        chip_mod_border = "#94A3B8"
        chip_mod_color = "#1E293B"

        chip_gain_bg = "#D1FAE5"
        chip_gain_border = "#10B981"
        chip_gain_color = "#065F46"

        chip_loss_bg = "#FEE2E2"
        chip_loss_border = "#EF4444"
        chip_loss_color = "#991B1B"

        chip_neut_bg = "#F1F5F9"
        chip_neut_border = "#CBD5E1"
        chip_neut_color = "#1E293B"

        stage_card_bg = "#F8FAFC"
        stage_card_border = "#CBD5E1"
        stage_num_color = "#475569"
        stage_headline_color = "#0F172A"
        stage_label_color = "#475569"
        stage_border_color = "#E2E8F0"
        stage_outcome_bg = "#FEF2F2"
        stage_outcome_border = "#FCA5A5"
        stage_outcome_text = "#991B1B"
        stage_outcome_sub = "#B91C1C"

    raw_css = f"""
<style>
{font_css}

/* Root Theme Variables */
:root {{
    --primary-color: {ACCENT_BLUE} !important;
    --background-color: {c_canvas} !important;
    --secondary-background-color: {c_surface} !important;
    --text-color: {c_text_primary} !important;
    --font: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* Hide Streamlit default chrome */
#MainMenu, footer, header, div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
    visibility: hidden !important;
    display: none !important;
}}

/* Global canvas and app background */
html, body, .stApp, div[data-testid="stAppViewContainer"] {{
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: {c_canvas} !important;
    color: {c_text_primary} !important;
    font-feature-settings: 'tnum' 1, 'cv05' 1 !important;
}}

/* Container Spacing */
.block-container {{
    padding-top: 0.75rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    max-width: 100% !important;
}}

/* Navigation Radio Pills */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    gap: 0.35rem !important;
    background-color: {c_surface} !important;
    padding: 0.3rem 0.6rem !important;
    border-radius: 4px !important;
    border: 1px solid {c_border_subtle} !important;
}}
div[data-testid="stRadio"] label p {{
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: {c_text_primary} !important;
}}

/* Hero Band Container */
.hero-band {{
    background-color: {c_surface};
    border: 1px solid {c_border_subtle};
    border-radius: 4px;
    padding: 1.15rem 1.35rem;
    color: {c_text_primary};
    margin-bottom: 0.85rem;
    box-shadow: { 'none' if dark_mode else '0 1px 3px rgba(0,0,0,0.05)' };
}}
.hero-title {{
    font-size: 1.2rem;
    font-weight: 600;
    color: {c_text_primary} !important;
    margin-bottom: 0.5rem;
    line-height: 1.35;
}}
.hero-meta-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    font-size: 0.8rem;
    margin-bottom: 0.85rem;
    align-items: center;
}}

/* Transmission Pipeline Grid */
.transmission-pipeline {{
    display: grid;
    grid-template-columns: 1.15fr auto 1.05fr auto 1.05fr auto 1.05fr auto 1.15fr;
    gap: 0.4rem;
    align-items: stretch;
    margin-top: 0.5rem;
}}
.stage-card {{
    background-color: {stage_card_bg};
    border: 1px solid {stage_card_border};
    border-radius: 4px;
    padding: 0.75rem 0.85rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}
.stage-card-outcome {{
    background-color: {stage_outcome_bg} !important;
    border: 1px solid {stage_outcome_border} !important;
}}
.stage-num {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: {stage_num_color} !important;
    margin-bottom: 0.35rem;
}}
.stage-headline {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {stage_headline_color} !important;
    line-height: 1.3;
    margin-bottom: 0.5rem;
}}
.stage-metric-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.76rem;
    padding: 0.22rem 0;
    border-bottom: 1px solid {stage_border_color};
}}
.stage-metric-label {{
    color: {stage_label_color} !important;
}}
.val-loss {{
    color: {'#F87171' if dark_mode else '#DC2626'} !important;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}
.val-gain {{
    color: {'#34D399' if dark_mode else '#059669'} !important;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}
.flow-chevron {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: {c_text_muted};
    font-size: 1.15rem;
    font-weight: 700;
    padding: 0 0.15rem;
}}

/* Content Cards */
.white-card {{
    background-color: {c_surface};
    border: 1px solid {c_border_subtle};
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    color: {c_text_primary};
    box-shadow: { 'none' if dark_mode else '0 1px 3px rgba(0,0,0,0.04)' };
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
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    letter-spacing: 0.1px;
}}
.chip-high {{
    background-color: {chip_high_bg};
    color: {chip_high_color} !important;
    border: 1px solid {chip_high_border};
}}
.chip-elevated {{
    background-color: {chip_elev_bg};
    color: {chip_elev_color} !important;
    border: 1px solid {chip_elev_border};
}}
.chip-moderate {{
    background-color: {chip_mod_bg};
    color: {chip_mod_color} !important;
    border: 1px solid {chip_mod_border};
}}
.chip-divergence {{
    background-color: {chip_elev_bg};
    color: {chip_elev_color} !important;
    border: 1px solid {chip_elev_border};
}}
.chip-gain {{
    background-color: {chip_gain_bg};
    color: {chip_gain_color} !important;
    border: 1px solid {chip_gain_border};
}}
.chip-loss {{
    background-color: {chip_loss_bg};
    color: {chip_loss_color} !important;
    border: 1px solid {chip_loss_border};
}}
.chip-neutral {{
    background-color: {chip_neut_bg};
    color: {chip_neut_color} !important;
    border: 1px solid {chip_neut_border};
}}

/* Streamlit Metrics */
div[data-testid="stMetric"] {{
    background-color: {c_surface} !important;
    border: 1px solid {c_border_subtle} !important;
    border-radius: 4px !important;
    padding: 0.65rem 0.85rem !important;
    box-shadow: { 'none' if dark_mode else '0 1px 3px rgba(0,0,0,0.04)' } !important;
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
    font-variant-numeric: tabular-nums !important;
}}

/* Tables & Glide Data Grid */
div[data-testid="stDataFrame"] {{
    border: 1px solid {c_border_subtle} !important;
    border-radius: 4px !important;
    background-color: {c_surface} !important;
}}
div[data-testid="stDataFrame"], [class*="gdg"] {{
    --gdg-bg-cell: {c_surface} !important;
    --gdg-bg-header: {c_surface_subtle} !important;
    --gdg-bg-header-has-focus: {c_surface_subtle} !important;
    --gdg-text-dark: {c_text_primary} !important;
    --gdg-text-medium: {c_text_secondary} !important;
    --gdg-text-light: {c_text_secondary} !important;
    --gdg-border-color: {c_border_subtle} !important;
    --gdg-accent-color: {ACCENT_BLUE} !important;
}}

/* Inputs & Selectboxes */
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background-color: {c_surface} !important;
    color: {c_text_primary} !important;
    border-color: {c_border_strong} !important;
}}

/* Captions */
div[data-testid="stCaptionContainer"] p {{
    color: {c_text_secondary} !important;
    font-size: 0.78rem !important;
    line-height: 1.45 !important;
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

/* Institutional Buttons */
div[data-testid="stButton"] button {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    padding: 0.38rem 0.85rem !important;
    transition: all 0.15s ease-in-out !important;
    border: 1px solid {c_border_strong} !important;
    background-color: {c_surface} !important;
    color: {c_text_primary} !important;
}}
div[data-testid="stButton"] button:hover {{
    border-color: {ACCENT_BLUE} !important;
    color: {ACCENT_BLUE} !important;
    box-shadow: 0 2px 6px rgba(45, 107, 255, 0.15) !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background-color: {NAVY_DEEP if dark_mode else "#1E40AF"} !important;
    border: 1px solid {ACCENT_BLUE} !important;
    color: #FFFFFF !important;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{
    background-color: {ACCENT_BLUE} !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 8px rgba(45, 107, 255, 0.3) !important;
}}

/* Institutional Download Buttons */
div[data-testid="stDownloadButton"] button {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    padding: 0.42rem 1rem !important;
    border: 1px solid {ACCENT_BLUE} !important;
    background-color: {'rgba(45, 107, 255, 0.12)' if dark_mode else '#EFF6FF'} !important;
    color: {'#60A5FA' if dark_mode else '#1D4ED8'} !important;
    transition: all 0.15s ease-in-out !important;
}}
div[data-testid="stDownloadButton"] button:hover {{
    background-color: {ACCENT_BLUE} !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 8px rgba(45, 107, 255, 0.25) !important;
}}

/* Institutional Sliders */
div[data-testid="stSlider"] div[data-baseweb="slider"] {{
    padding-top: 0.3rem !important;
    padding-bottom: 0.3rem !important;
}}
div[data-testid="stSlider"] label p {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {c_text_primary} !important;
}}

/* Radio Navigation Tabs Active State */
div[data-testid="stRadio"] label:has(input:checked) {{
    background-color: {'rgba(45, 107, 255, 0.16)' if dark_mode else '#DBEAFE'} !important;
    border-radius: 3px !important;
}}
div[data-testid="stRadio"] label:has(input:checked) p {{
    color: {'#60A5FA' if dark_mode else '#1D4ED8'} !important;
    font-weight: 600 !important;
}}

/* Dividers */
hr {{
    border: none !important;
    border-top: 1px solid {c_border_subtle} !important;
    margin: 0.85rem 0 !important;
}}
</style>
"""
    render_html(raw_css)


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
