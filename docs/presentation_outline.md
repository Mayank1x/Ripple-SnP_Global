# 7-Slide Presentation Deck Outline: Ripple Risk Engine

**S&P Global & CRISIL Campus Hackathon 2026**  
**Module B: Strategic Event-Driven Portfolio Stress-Testing Tool**

This document provides the slide-by-slide structure, talking points, and screenshot assignments for the final presentation deck (`docs/presentation.pdf`).

---

### Slide 1: Title & Introduction
- **Header:** Ripple: AI/NLP Risk Engine & Strategic Stress-Testing Platform
- **Subheader:** Automated Financial Event Extraction and Contagion Revaluation for Wholesale Banking
- **Candidate Details:**
  - Name: Mayank Sharma
  - College: Indian Institute of Technology
  - Track: Module B (Strategic Portfolio Stress Testing)
  - Date: October 2026
- **Visual / Graphic:**
  - Ripple Platform Logo & S&P Global / CRISIL Hackathon banner.
- **Speaker Notes (30s):**
  - "Good morning, members of the jury. I am presenting Ripple, an AI risk platform that connects raw, unstructured financial news to balance-sheet loss estimates. Ripple monitors global news and social feeds, identifies emerging credit and market events, and stresses wholesale banking books across loans, bonds, and derivatives."

---

### Slide 2: Problem Statement & Solution Approach
- **Header:** The Challenge: Unstructured Text to P&L Risk Intelligence
- **Key Pain Points:**
  1. *Information Overload*: Risk analysts scan thousands of headlines daily across news feeds, regulatory wires, and social platforms.
  2. *Disconnect from Balance Sheets*: Standard NLP models output generic sentiment tags (e.g., "Negative 0.82") that fail to inform capital reserves, credit risk, or asset revaluation.
  3. *Unseen Contagion*: Supply chain linkages and cross-sector dependencies cause second-order losses not mentioned in the source article.
- **Our Approach (Ripple):**
  - Convert unstructured articles into structured risk events using FinBERT and taxonomy classification.
  - Calibrate raw impact scores against historical market price moves ($|CAR_{0,1}|$ and volatility shifts).
  - Automatically propagate shocks through an interconnected exposure graph to revalue loans, bonds, and derivatives under empirical crisis analogs.
- **Screenshot / Graphic:**
  - Conceptual diagram comparing the **Traditional Manual Workflow** (Headline $\rightarrow$ Human Discretion $\rightarrow$ Delayed Analysis) vs. **Ripple Automated Workflow** (Raw Text $\rightarrow$ AI Engine $\rightarrow$ Calibrated Signal $\rightarrow$ Portfolio Loss $\&$ Confidence Bands).
- **Speaker Notes (60s):**
  - "Traditional sentiment analysis gives analysts a score from -1 to +1, but it doesn't answer the question risk officers ask: 'How much money could this cost our portfolio?' Ripple bridges that gap by linking NLP event detection directly to multi-asset pricing models and contagion networks."

---

### Slide 3: System Design & End-to-End Architecture
- **Header:** End-to-End System Architecture
- **Key Architectural Components:**
  - **Ingestion & Deduplication:** Multi-source feeds (GDELT 2.0, NewsAPI/RSS, Twitter) with automated disk cache replay and fallback handling.
  - **NLP Risk Engine:** FinBERT domain sentiment, NER entity dictionary lookups, and Gemini 2.0 Flash crisis classification.
  - **Clustering & Calibration:** Rolling 7-day FAISS clustering, narrative divergence tracking, and isotonic regression calibration.
  - **Module B Stress Engine:** NetworkX contagion propagation, historical analog matching, Bayesian shrinkage, and multi-asset position revaluation.
  - **Delivery Layer:** 100% offline Streamlit + Plotly interactive dashboard and FastAPI microservice.
- **Primary Screenshot to Embed:**
  - **File:** `docs/architecture.png`
  - High-resolution architecture flow diagram showing all 6 interconnected layers and dataflows.
- **Speaker Notes (60s):**
  - "Here is our complete system architecture. Data enters from GDELT, NewsAPI, and social channels. It is deduplicated and scored by FinBERT. Related articles are clustered over a 7-day rolling window in FAISS. When an event's calibrated impact reaches 7 or above, it triggers our Module B stress engine to calculate portfolio-level losses."

---

### Slide 4: Implementation Highlights & Technical Choices
- **Header:** Core Technical Innovations & Design Decisions
- **Four Key Engineering Choices:**
  1. *FinBERT vs. Generic LLM for Sentiment*: FinBERT (`ProsusAI/finbert`) runs locally in milliseconds and understands financial context where words like 'liability' or 'interest' are operational rather than negative.
  2. *Isotonic Regression Calibration*: Avoids arbitrary linear weights. Fits raw NLP features against empirical abnormal returns ($|CAR_{0,1}|$) from historical event studies, enforcing monotonicity without assuming normality.
  3. *Bayesian Prior Shrinkage*: When fewer than $N=3$ crisis analogs exist in FAISS, the engine shrinks empirical shocks toward documented crisis priors:
     $$\text{Shock}_{\text{shrunk}} = \lambda \cdot \text{Shock}_{\text{analog}} + (1 - \lambda) \cdot \text{Shock}_{\text{prior}}$$
  4. *Deterministic Offline Replay*: All API requests are cached to disk, enabling complete offline execution in under 30 seconds (`python run.py --mode replay`).
- **Screenshot / Graphic:**
  - Snippet comparison: Table comparing naive keyword search vs. FinBERT sentiment, and the Bayesian shrinkage formula.
- **Speaker Notes (60s):**
  - "We made intentional choices to balance accuracy, speed, and auditability. Instead of sending entire portfolios to proprietary cloud LLMs, our financial formulas—like loan default intensity acceleration and bond spread duration—are executed deterministically in Python. Every run is saved to disk as a timestamped audit log."

---

### Slide 5: Key Results & Quantitative Validation
- **Header:** Quantitative Results & Prototype Demonstration
- **Key Metrics & Outputs:**
  - **Replay Speed:** Processes 102 news documents, clusters 74 signals, and runs 3 full portfolio stress tests in **26.8 seconds** (< 60s benchmark).
  - **Wholesale Portfolio Stress Test ($8.54B Book):**
    - Commercial real estate credit contagion event triggers a **$127.6 Million** median loss ($-1.49\%$).
    - Multi-asset breakdown: Loans account for $68.4\%$ ($-\$87.3\text{M}$), Corporate Bonds $24.1\%$ ($-\$30.8\text{M}$), OTC Derivatives $7.5\%$ ($-\$9.5\text{M}$).
    - Confidence interval: Mild band ($\text{p5}$) at $-\$81.2\text{M}$ to Severe tail ($\text{p95}$) at $-\$189.4\text{M}$.
  - **Empirical Calibration:** Statistically significant rank correlation (Spearman's $\rho = 0.58$) between calibrated impact score and realized market volatility shifts.
- **Screenshots to Embed:**
  1. **Primary (Left):** `docs/calibration.png` (Impact Bucket vs. Mean Realized Move with error bars).
  2. **Secondary (Right):** Dashboard screenshot of **Page 2: Stress Test View** showing the asset class loss waterfall and p5-p95 confidence band.
- **Speaker Notes (60s):**
  - "On Slide 5, you see our empirical validation. Our calibration chart confirms that higher impact scores consistently correspond to larger observed market moves. On the right, our dashboard displays the stress outcome on an $8.54 billion portfolio: loans take the primary hit, followed by credit spread widening in bonds, with confidence bands giving risk committees full visibility of the tail risk."

---

### Slide 6: Domain Impact & Business Value
- **Header:** Business Value for Wholesale Banking & Risk Committees
- **Four Core Applications:**
  1. *Early Warning System (EWS)*: Identifies credit deterioration days before formal rating agency downgrades or quarterly regulatory filings.
  2. *Capital Planning & ICAAP*: Informs Internal Capital Adequacy Assessment Process (ICAAP) by quantifying event-driven loss bands alongside standard regulatory scenarios.
  3. *Second-Order Contagion Discovery*: Uncovers indirect counterparties at risk (e.g., regional suppliers, tenant companies, and derivative counterparties) using the Exposure Graph.
  4. *Sentiment Divergence Early Indicator*: Flags narrative decouplings where public sentiment remains optimistic while institutional news warns of liquidity strain.
- **Screenshot to Embed:**
  - **File:** `docs/exposure_graph.png` or Dashboard screenshot of **Page 3: Contagion Map** showing node shock intensities and contagion spread from a distressed sector to supply chain partners.
- **Speaker Notes (60s):**
  - "From a business perspective, Ripple transforms compliance monitoring into an active risk management tool. Risk officers can immediately see how a supply chain shock or regional bank default ripples into their loan book and derivative margin calls, enabling pre-emptive hedging before markets price in the full disruption."

---

### Slide 7: Honest Limitations & Production Roadmap
- **Header:** Technical Limitations & Future Enhancements
- **Current Limitations:**
  - *Calibration Sample Size*: Trained on 20 historical crisis events; production requires 500+ events across multiple economic cycles.
  - *Bond Pricing*: Uses first-order duration and spread duration; omits second-order bond convexity ($+\frac{1}{2}\text{Conv} \cdot dy^2$).
  - *Static Book*: Assumes the portfolio remains unhedged during the 5-day stress horizon (no dynamic deleveraging).
- **Roadmap for Enterprise Deployment:**
  - Ingest full SEC 10-K/10-Q filings and loan credit agreements using long-context LLMs.
  - Connect live enterprise Bloomberg B-PIPE or Refinitiv data feeds.
  - Implement full Monte Carlo simulation for non-linear derivative portfolios (options gamma/vega).
- **Screenshot / Graphic:**
  - Roadmap timeline graphic showing Phase 1 (Current Prototype) $\rightarrow$ Phase 2 (Enterprise Data Connectors) $\rightarrow$ Phase 3 (Production Trading & Hedging Integration).
- **Speaker Notes (30s):**
  - "We maintain full technical honesty regarding our prototype's boundaries. While our sample size is small and our bond formulas use first-order duration, the mathematical architecture is sound and ready for enterprise scale with full historical market data and real-time news feeds. Thank you, and I look forward to your questions."
