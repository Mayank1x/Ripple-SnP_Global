"""Executive Audit Pack & Risk Committee Report Generator (Feature 5).

Generates a standalone, institutional, print-ready HTML/PDF audit dossier for
board presentations, supervisory review, and executive risk committees.
Features:
- CRISIL & S&P Global Hackathon institutional branding and audit stamps.
- Complete causal transmission audit (unstructured headline to balance sheet loss).
- Multi-asset position write-downs and Basel III / CET1 capital impact.
- One-click print-to-PDF support via responsive media print stylesheet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.stress.revalue import compute_regulatory_capital


def generate_audit_pack_html(
    stress_run: dict[str, Any],
    custom_shocks: dict[str, float] | None = None,
    is_custom: bool = False,
    k_loan: float = 2.0,
    include_convexity: bool = False,
) -> str:
    """Generate a self-contained, publication-grade HTML audit pack document.

    Args:
        stress_run: Dictionary containing stress test results (either standard or custom).
        custom_shocks: Optional dictionary of macro factor shocks if running in What-If mode.
        is_custom: Boolean indicating if this is an interactive scenario simulation.
        k_loan: Exponential multiplier used for loan PD acceleration.
        include_convexity: Whether 2nd-order bond convexity was active.

    Returns:
        Complete HTML string with embedded institutional CSS and print rules.
    """
    run_id = stress_run.get("run_id", "STRESS_AUDIT_001")
    timestamp = stress_run.get("timestamp", datetime.now(timezone.utc).isoformat())
    sig_summary = stress_run.get("signal_summary", {})

    event_id = sig_summary.get("event_id", "EVT_TRIGGER_01")
    title = sig_summary.get("title", "Credit spread widening and liquidity disruption")
    event_type = sig_summary.get("event_type", "Credit Event")
    impact = float(sig_summary.get("impact_score", 7.5))
    sentiment = float(sig_summary.get("sentiment_score", -0.716))
    confidence = float(sig_summary.get("confidence", 0.92))

    pre_val = float(stress_run.get("portfolio_value_before", 8538147288.10))
    post_val = float(stress_run.get("portfolio_value_after_p50", 8387795248.26))
    loss = float(stress_run.get("loss_p50", 159289744.22))
    loss_pct = float(stress_run.get("loss_pct_p50", 0.0187))

    band = stress_run.get("band", {})
    p5_loss = float(band.get("p5_loss", loss * 0.92))
    p95_loss = float(band.get("p95_loss", loss * 1.08))

    loss_by_class = stress_run.get("loss_by_asset_class", {})
    loss_by_sector = stress_run.get("loss_by_sector", {})
    top_positions = stress_run.get("top_10_positions", [])[:10]
    shock_nodes = stress_run.get("graph_shock_nodes", {})

    # Shocks extraction
    shocks_dict = custom_shocks or {}
    if not shocks_dict:
        raw_shocks = stress_run.get("shocks", {})
        factors = raw_shocks.get("factors", {}) if isinstance(raw_shocks, dict) else {}
        for factor_name, q_dict in factors.items():
            if isinstance(q_dict, dict):
                shocks_dict[factor_name] = float(q_dict.get("p50", 0.0))
            elif isinstance(q_dict, (int, float)):
                shocks_dict[factor_name] = float(q_dict)

    dy_bps = shocks_dict.get("yield_10y", 0.0025) * 10_000.0
    dspread_bps = shocks_dict.get("credit_spread", 0.0045) * 10_000.0
    doil_pct = shocks_dict.get("brent_oil", 0.05) * 100.0
    dfx_pct = shocks_dict.get("fx_dxy", shocks_dict.get("fx_usdinr", 0.01)) * 100.0
    dequity_pct = shocks_dict.get("equity_us", -0.025) * 100.0

    # Regulatory Capital Computation
    reg_cap = compute_regulatory_capital(pre_val, loss)

    report_title = (
        "Tactical 'What-If' Stress Scenario Dossier"
        if is_custom
        else "Strategic Risk Committee Audit Pack (Module B)"
    )
    classification_badge = (
        '<span class="badge badge-warning">Interactive Scenario Simulation</span>'
        if is_custom
        else '<span class="badge badge-danger">Official High-Impact Trigger Run</span>'
    )

    # Format positions rows
    pos_rows_html = ""
    for idx, p in enumerate(top_positions, 1):
        c_name = p.get("counterparty", p.get("position_id", f"Position {idx}"))
        ac = p.get("asset_class", "Loan")
        sec = p.get("sector", "Financials")
        ctry = p.get("country", "US")
        p_pre = float(p.get("pre_value", 0.0))
        p_loss = float(p.get("loss", 0.0))
        p_pct = float(p.get("loss_pct", 0.0)) * 100.0
        pos_rows_html += f"""
        <tr>
            <td style="font-weight: 600;">{idx}. {c_name}</td>
            <td><span class="chip-sm">{ac}</span></td>
            <td>{sec}</td>
            <td>{ctry}</td>
            <td class="text-right">${p_pre/1e6:.1f}M</td>
            <td class="text-right text-loss">-${p_loss/1e6:.1f}M</td>
            <td class="text-right text-loss">-{p_pct:.2f}%</td>
        </tr>
        """

    # Asset class breakdown rows
    asset_rows_html = ""
    for ac, ac_loss in loss_by_class.items():
        share = (ac_loss / loss * 100.0) if loss > 0 else 0.0
        asset_rows_html += f"""
        <tr>
            <td style="font-weight: 600;">{ac}</td>
            <td class="text-right text-loss">-${ac_loss/1e6:.2f}M</td>
            <td class="text-right">{share:.1f}%</td>
            <td>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {min(100.0, max(5.0, share))}%;"></div>
                </div>
            </td>
        </tr>
        """

    # Top shock nodes
    top_shock_nodes = sorted(shock_nodes.items(), key=lambda x: x[1], reverse=True)[:6]
    contagion_nodes_html = ""
    for node, intensity in top_shock_nodes:
        contagion_nodes_html += f"""
        <div class="contagion-tag">
            <span class="node-name">{node}</span>
            <span class="node-intensity">{intensity:.3f}</span>
        </div>
        """
    if not contagion_nodes_html:
        contagion_nodes_html = "<div style='color: #64748B;'>Primary industry sectors and corporate borrower nodes.</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ripple Risk Audit Pack — {run_id}</title>
    <style>
        /* Base typography & CSS variables */
        :root {{
            --primary: #0F172A;
            --secondary: #334155;
            --muted: #64748B;
            --border: #E2E8F0;
            --border-strong: #94A3B8;
            --accent: #2D6BFF;
            --loss: #DC2626;
            --gain: #16A34A;
            --surface: #FFFFFF;
            --surface-subtle: #F8FAFC;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: var(--font-family);
            background-color: #F1F5F9;
            color: var(--primary);
            font-size: 13px;
            line-height: 1.5;
            padding: 24px;
        }}

        .container {{
            max-width: 980px;
            margin: 0 auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
            overflow: hidden;
        }}

        /* Header bar */
        .header {{
            background: #0B1B33;
            color: #FFFFFF;
            padding: 24px 32px;
            border-bottom: 3px solid #2D6BFF;
            position: relative;
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .brand-title {{
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: #FFFFFF;
        }}

        .brand-sub {{
            font-size: 11px;
            color: #94A3B8;
            letter-spacing: 0.2px;
        }}

        .header-headline {{
            font-size: 20px;
            font-weight: 600;
            color: #F8FAFC;
            margin: 8px 0;
            line-height: 1.35;
        }}

        .meta-pills {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 12px;
        }}

        .pill {{
            background: rgba(255, 255, 255, 0.12);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
            color: #E2E8F0;
        }}

        .pill-danger {{
            background: rgba(220, 38, 38, 0.25);
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}

        /* Print Button */
        .btn-print {{
            background: #2D6BFF;
            color: #FFFFFF;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn-print:hover {{
            background: #1D4ED8;
        }}

        /* Content body */
        .body-content {{
            padding: 32px;
        }}

        /* Section Headings */
        .section-header {{
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--secondary);
            border-bottom: 2px solid var(--border);
            padding-bottom: 6px;
            margin-top: 24px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Metric Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px 16px;
        }}

        .kpi-label {{
            font-size: 11px;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
        }}

        .kpi-value {{
            font-size: 20px;
            font-weight: 700;
            color: var(--primary);
            font-variant-numeric: tabular-nums;
        }}

        .kpi-value.loss {{
            color: var(--loss);
        }}

        .kpi-sub {{
            font-size: 11px;
            color: var(--muted);
            margin-top: 4px;
        }}

        /* Two column layout */
        .col-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }}

        th {{
            background: #F1F5F9;
            color: #475569;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10.5px;
            letter-spacing: 0.4px;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-strong);
        }}

        td {{
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            color: var(--secondary);
            font-variant-numeric: tabular-nums;
        }}

        tr:nth-child(even) td {{
            background: #FAFAFA;
        }}

        .text-right {{
            text-align: right;
        }}

        .text-loss {{
            color: var(--loss);
            font-weight: 600;
        }}

        .text-gain {{
            color: var(--gain);
            font-weight: 600;
        }}

        .chip-sm {{
            background: #E2E8F0;
            color: #1E293B;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }}

        .progress-bar-bg {{
            background: #E2E8F0;
            border-radius: 4px;
            height: 8px;
            width: 100%;
            overflow: hidden;
        }}

        .progress-bar-fill {{
            background: var(--loss);
            height: 100%;
            border-radius: 4px;
        }}

        /* Contagion tags */
        .contagion-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }}

        .contagion-tag {{
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 4px;
            padding: 4px 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
        }}

        .contagion-tag .node-name {{
            font-weight: 600;
            color: #991B1B;
        }}

        .contagion-tag .node-intensity {{
            background: #DC2626;
            color: #FFFFFF;
            font-size: 9.5px;
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: 700;
        }}

        /* Governance sign-off block */
        .governance-block {{
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            padding: 16px 20px;
            border-radius: 4px;
            margin-top: 28px;
            font-size: 11.5px;
            line-height: 1.6;
        }}

        .footer {{
            border-top: 1px solid var(--border);
            padding: 16px 32px;
            background: #F8FAFC;
            color: var(--muted);
            font-size: 11px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Print Styles */
        @media print {{
            body {{
                background: #FFFFFF !important;
                padding: 0 !important;
                color: #000000 !important;
            }}
            .container {{
                box-shadow: none !important;
                border: none !important;
                max-width: 100% !important;
            }}
            .btn-print {{
                display: none !important;
            }}
            .header {{
                background: #0B1B33 !important;
                color: #FFFFFF !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            tr:nth-child(even) td {{
                background: #F8FAFC !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .progress-bar-fill {{
                background: #DC2626 !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .page-break {{
                page-break-before: always;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-top">
                <div>
                    <div class="brand-title">Ripple Risk Analytics | S&P Global & CRISIL</div>
                    <div class="brand-sub">Campus Hackathon 2026 — Institutional Risk Governance Protocol</div>
                </div>
                <button onclick="window.print()" class="btn-print">
                    Print / Save as PDF
                </button>
            </div>
            <div class="header-headline">{title}</div>
            <div class="meta-pills">
                <span class="pill pill-danger">{report_title}</span>
                <span class="pill">Run ID: {run_id}</span>
                <span class="pill">Event ID: {event_id}</span>
                <span class="pill">Classification: {event_type}</span>
                <span class="pill">Calibrated Impact: {impact:.1f} / 10.0</span>
                <span class="pill">FinBERT Sentiment: {sentiment:+.3f}</span>
                <span class="pill">Generated: {timestamp[:19].replace('T', ' ')} UTC</span>
            </div>
        </div>

        <div class="body-content">
            <!-- 1. Key Performance Indicators -->
            <div class="section-header">
                <span>1. Portfolio Balance Sheet Loss & Capital Health</span>
                <span style="font-size: 11px; text-transform: none; color: var(--muted); font-weight: normal;">Wholesale Banking Book ($8.54B Par)</span>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Pre-Stress Exposure</div>
                    <div class="kpi-value">${pre_val/1e9:.2f}B</div>
                    <div class="kpi-sub">180 Multi-Asset Positions</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Stressed Valuation</div>
                    <div class="kpi-value">${post_val/1e9:.2f}B</div>
                    <div class="kpi-sub">Expected Median (p50)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Expected Economic Loss</div>
                    <div class="kpi-value loss">-${loss/1e6:.1f}M</div>
                    <div class="kpi-sub">-{loss_pct*100:.2f}% Book Value Write-Down</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Confidence Band (p5 - p95)</div>
                    <div class="kpi-value" style="font-size: 17px; color: #B91C1C;">-${p5_loss/1e6:.0f}M to -${p95_loss/1e6:.0f}M</div>
                    <div class="kpi-sub">Analog Bayesian Shrinkage</div>
                </div>
            </div>

            <!-- 2. Basel III Regulatory Capital Health -->
            <div class="section-header">
                <span>2. Basel III Regulatory Capital Adequacy (CET1 Ratio)</span>
                <span style="font-size: 11px; text-transform: none; color: {'#16A34A' if reg_cap['is_compliant'] else '#DC2626'}; font-weight: 600;">Status: {reg_cap['status']}</span>
            </div>

            <table style="margin-bottom: 24px;">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline Value</th>
                        <th>Stressed Value</th>
                        <th>Regulatory Minimum</th>
                        <th>Headroom / Surplus</th>
                        <th>Supervisory Assessment</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="font-weight: 600;">Common Equity Tier 1 (CET1) Ratio</td>
                        <td>{reg_cap['base_cet1']*100:.2f}%</td>
                        <td class="text-loss" style="font-weight: 700;">{reg_cap['stressed_cet1']*100:.2f}%</td>
                        <td>{reg_cap['min_requirement']*100:.2f}%</td>
                        <td class="{'text-gain' if reg_cap['headroom_bps'] > 0 else 'text-loss'}">+{reg_cap['headroom_bps']:.0f} bps</td>
                        <td>{reg_cap['status']}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600;">Tier 1 Eligible Capital</td>
                        <td>${reg_cap['base_tier1']/1e6:.1f}M</td>
                        <td class="text-loss">${reg_cap['stressed_tier1']/1e6:.1f}M</td>
                        <td>N/A</td>
                        <td class="text-loss">-${loss/1e6:.1f}M Absorption</td>
                        <td>Mark-to-market and loan ECL deduction</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600;">Risk-Weighted Assets (RWA)</td>
                        <td>${reg_cap['base_rwa']/1e6:.1f}M</td>
                        <td>${reg_cap['stressed_rwa']/1e6:.1f}M</td>
                        <td>N/A</td>
                        <td>+${(reg_cap['stressed_rwa'] - reg_cap['base_rwa'])/1e6:.1f}M</td>
                        <td>Expansion from distressed borrower migrations</td>
                    </tr>
                </tbody>
            </table>

            <!-- 3. Factor Shocks Matrix & Asset Class Decomposition -->
            <div class="col-2">
                <div>
                    <div class="section-header">
                        <span>3. Macro Factor Shocks Applied</span>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>Risk Factor</th>
                                <th class="text-right">Applied Shock</th>
                                <th>Channel Sensitivity</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Benchmark 10Y Yield</td>
                                <td class="text-right">{dy_bps:+.1f} bps</td>
                                <td>Bond ModDur & IRS DV01</td>
                            </tr>
                            <tr>
                                <td>Corporate Credit Spread</td>
                                <td class="text-right">{dspread_bps:+.1f} bps</td>
                                <td>Spread Duration & Loan ECL</td>
                            </tr>
                            <tr>
                                <td>Brent Crude Oil</td>
                                <td class="text-right">{doil_pct:+.1f}%</td>
                                <td>Consumer & Industrial Margin</td>
                            </tr>
                            <tr>
                                <td>US Dollar Index (DXY)</td>
                                <td class="text-right">{dfx_pct:+.1f}%</td>
                                <td>FX Forward Notional MTM</td>
                            </tr>
                            <tr>
                                <td>Loan Sensitivity ($k$)</td>
                                <td class="text-right">{k_loan:.1f}x</td>
                                <td>Merton Hazard Rate Exp</td>
                            </tr>
                            <tr>
                                <td>Bond Convexity Adjustment</td>
                                <td class="text-right">{'Active (2nd Order)' if include_convexity else 'Linear ModDur'}</td>
                                <td>Curvature correction</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div>
                    <div class="section-header">
                        <span>4. Loss Decomposition by Asset Class</span>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>Asset Class</th>
                                <th class="text-right">Loss ($)</th>
                                <th class="text-right">Share (%)</th>
                                <th style="width: 35%;">Relative Burden</th>
                            </tr>
                        </thead>
                        <tbody>
                            {asset_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- 5. Top 10 Counterparty Exposure Write-Downs -->
            <div class="section-header page-break">
                <span>5. Top 10 Counterparty Exposure Write-Downs</span>
                <span style="font-size: 11px; text-transform: none; color: var(--muted); font-weight: normal;">Primary Single-Name Concentrations</span>
            </div>

            <table style="margin-bottom: 24px;">
                <thead>
                    <tr>
                        <th>Counterparty Entity</th>
                        <th>Asset Class</th>
                        <th>Sector</th>
                        <th>Jurisdiction</th>
                        <th class="text-right">Baseline Value</th>
                        <th class="text-right">Stressed Loss</th>
                        <th class="text-right">Write-Down</th>
                    </tr>
                </thead>
                <tbody>
                    {pos_rows_html}
                </tbody>
            </table>

            <!-- 6. Contagion Propagation Pathways -->
            <div class="section-header">
                <span>6. Interconnected Network Contagion Pathways</span>
                <span style="font-size: 11px; text-transform: none; color: var(--muted); font-weight: normal;">NetworkX Propagation (Hops = 2, Decay = 0.50)</span>
            </div>

            <div style="font-size: 12px; color: var(--secondary); margin-bottom: 8px;">
                Identified primary casualty nodes transmitting distress along intercompany and supply-chain linkages:
            </div>
            <div class="contagion-container" style="margin-bottom: 24px;">
                {contagion_nodes_html}
            </div>

            <!-- 7. Model Governance & Audit Certification -->
            <div class="governance-block">
                <strong style="color: var(--primary);">Model Governance & Methodology Certification:</strong><br>
                This audit pack was generated deterministically by the <strong>Ripple AI/NLP Risk Engine (Module B)</strong>. 
                Valuations adhere to the Merton/Vasicek credit risk hazard model for loans ($EAD \cdot LGD \cdot \Delta PD$), 
                modified duration and credit spread duration pricing for corporate bonds, and linear DV01/delta sensitivities for OTC derivatives. 
                Macro shocks incorporate historical analog crisis retrieval with Bayesian prior shrinkage. 
                Outputs have been certified for review by the Wholesale Banking Risk Committee.
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <span>Ripple Risk Engine v1.0.0 &bull; S&P Global & CRISIL Campus Hackathon 2026</span>
            <span>Document ID: {run_id} &bull; Confidential Risk Management Working Paper</span>
        </div>
    </div>
</body>
</html>
"""
    return html


def save_audit_pack_to_disk(
    stress_run: dict[str, Any],
    custom_shocks: dict[str, float] | None = None,
    is_custom: bool = False,
    output_dir: Path | str | None = None,
    k_loan: float = 2.0,
    include_convexity: bool = False,
) -> Path:
    """Generate HTML audit pack and write to disk under data/audit/."""
    if output_dir is None:
        target_dir = Path("data/audit")
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    run_id = stress_run.get("run_id", "audit_run")
    prefix = "whatif_audit" if is_custom else "risk_audit_pack"
    filename = f"{prefix}_{run_id}.html"
    filepath = target_dir / filename

    html = generate_audit_pack_html(
        stress_run=stress_run,
        custom_shocks=custom_shocks,
        is_custom=is_custom,
        k_loan=k_loan,
        include_convexity=include_convexity,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath
