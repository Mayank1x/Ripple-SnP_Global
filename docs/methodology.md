# Ripple Stress Engine (Module B) — Methodology & Valuation Formulas

This document outlines the theoretical foundations, shock propagation dynamics, and exact asset-level revaluation formulas implemented in the **Ripple AI/NLP Risk Engine**.

---

## 1. Overview of Stress Architecture

The Ripple Stress Engine evaluates the financial vulnerability of a wholesale banking portfolio to emerging external shocks extracted by the AI/NLP pipeline.

The execution workflow proceeds through four interconnected stages:
1. **Signal Intake & Triggering**: Evaluates the calibrated impact score of the incoming event signal against a threshold ($\text{impact\_score} \ge 7.0$).
2. **Analog Event Retrieval & Factor Distribution**:
   - Queries a FAISS index of historical financial crises (filtered by `event_type`) using sentence-transformer embeddings (`all-MiniLM-L6-v2`).
   - Retrieves the empirical distribution of forward factor moves across 1-day and 5-day horizons.
   - If fewer than $N$ analogs are found ($N=3$), applies Bayesian shrinkage towards a curated prior table:
     $$\text{Shock}_{\text{shrunk}} = \lambda \cdot \text{Shock}_{\text{analog}} + (1 - \lambda) \cdot \text{Shock}_{\text{prior}}, \quad \lambda = \frac{k}{N}$$
3. **Network Contagion Propagation**:
   - Injects the shock into the NetworkX Exposure Graph at seed nodes (extracted companies, operating sectors, geographic jurisdictions, and commodity exposures).
   - Propagates shock intensities through supply chain (`supplies`), ownership (`member_of`), and structural sensitivity (`sensitive_to`) edges with distance decay:
     $$\text{Intensity}_{v} = \max_{u \in \text{neighbors}(v)} \left( \text{Intensity}_{u} \times W_{uv} \times \text{decay}^{\text{hop}} \right)$$
4. **Multi-Asset Revaluation & Banding**:
   - Revalues the banking book under median ($\text{p50}$), mild ($\text{p5}$), and severe ($\text{p95}$) factor moves.

---

## 2. Asset Revaluation Formulas

### 2.1 Corporate Loans Book

Credit risk on corporate loans is driven by Exposure at Default ($\text{EAD}$), Loss Given Default ($\text{LGD}$), and the Probability of Default ($\text{PD}$).

Under stress, systemic network contagion and sector distress elevate the borrower's default intensity:

$$\text{PD}_{\text{stressed}} = \min \left( 1.0, \; \text{PD} \times \exp \left( k \times \text{Intensity}_{\text{sector}} \right) \right)$$

Where:
- $\text{PD}$: Baseline annual Probability of Default (from rating agency calibration).
- $k$: Loan sensitivity factor ($k = 2.0$ default in config).
- $\text{Intensity}_{\text{sector}}$: Net contagion intensity transmitted to the company or its operating sector from the exposure graph.

The monetary credit loss is calculated as the incremental expected loss:

$$\text{Loss}_{\text{Loan}} = \text{EAD} \times \text{LGD} \times \left( \text{PD}_{\text{stressed}} - \text{PD} \right)$$

*Property*: When $\text{Intensity}_{\text{sector}} = 0$, $\text{PD}_{\text{stressed}} = \text{PD}$, resulting in zero credit loss ($\text{Loss} = 0$).

---

### 2.2 Bonds Book

Bond positions are revalued using duration-based price sensitivity to changes in benchmark sovereign yields and issuer credit spreads:

$$dP = - \text{ModDur} \times dy - \text{SpreadDur} \times d\text{spread}$$

Where:
- $\text{ModDur}$: Modified Duration of the bond (in years).
- $\text{SpreadDur}$: Credit Spread Duration of the bond (in years).
- $dy$: Benchmark interest rate / 10Y sovereign yield shock (e.g. $+0.0025$ for $+25\text{ bps}$).
- $d\text{spread}$: Credit spread widening (e.g. $+0.0035$ for $+35\text{ bps}$).

The monetary market value loss is:

$$\text{Loss}_{\text{Bond}} = \text{MTM}_{\text{pre}} \times \max(0, -dP)$$

*Property*: An increase in interest rates ($dy > 0$) or widening of credit spreads ($d\text{spread} > 0$) causes $dP < 0$, lowering the bond's value and generating a positive loss. When $dy = 0$ and $d\text{spread} = 0$, $dP = 0$ and $\text{Loss} = 0$.

---

### 2.3 Derivatives Book

Derivatives are revalued according to their primary market risk driver:

1. **Foreign Exchange (FX) Forwards**:
   $$\text{Loss}_{\text{FX}} = \text{Notional} \times |d\text{FX}|$$
   Where $d\text{FX}$ is the relative change in the foreign exchange rate or dollar index ($\text{DXY}$).

2. **Interest Rate Swaps (IRS)**:
   $$\text{Loss}_{\text{IRS}} = \text{DV01} \times |d\text{Rate}_{\text{bp}}|$$
   Where $\text{DV01}$ is the Dollar Value of a Basis Point, and $d\text{Rate}_{\text{bp}} = dy \times 10{,}000$ represents the yield change in basis points.

3. **Commodity Swaps**:
   $$\text{Loss}_{\text{Commodity}} = \text{Notional} \times |d\text{Price}|$$
   Where $d\text{Price}$ is the percentage return of the underlying commodity (e.g. Brent crude oil, natural gas, or copper).

*Property*: When all factor moves are zero, all derivative losses evaluate exactly to zero.

---

## 3. Commodity-to-Sector Transmission Channels

Spikes in commodity prices (e.g., Brent crude oil surges) transmit adverse economic pressure through two distinct channels:
1. **Network Contagion**: Seeded at the commodity node (e.g. "Crude Oil"), propagating along `sensitive_to` edges to Energy producers and downstream supply chain partners.
2. **Margin Compression on Energy-Consuming Sectors**: High energy costs squeeze operating margins in heavy consumer sectors (Industrials, Automotive, Utilities, Materials), driving higher borrower PDs and increased loan losses.

---

## 4. Confidence Banding (p5 / p50 / p95)

Each stress run computes three portfolio outcomes:
- **Median ($\text{p50}$)**: Expected baseline stress trajectory.
- **Low Impact ($\text{p5}$)**: Favorable / mild outcome band.
- **High Impact ($\text{p95}$)**: Tail risk / severe outcome band.

This provides wholesale risk committees and capital planners with a comprehensive picture of both expected losses and extreme tail vulnerabilities.
