# Ripple Platform — Limitations & Assumptions

This document provides a technical and quantitative assessment of the limitations, simplifications, and operational assumptions present in the **Ripple AI/NLP Risk Engine** and **Module B Stress Engine**.

---

## 1. Data & Sample Size Limitations

1. **Small Calibration Sample Size:**
   - The isotonic calibration model was trained on a curated set of 20 historical crisis events matched against Yahoo Finance market returns.
   - While the calibrated model demonstrates a positive rank correlation (Spearman's $\rho \approx 0.58$), a production-grade risk model would require hundreds of events across multiple market cycles (e.g., 2008 GFC, 2011 European sovereign debt crisis, 2020 COVID shock, 2023 US regional banking crisis).
2. **Public API Rate Limits & Latency:**
   - Public APIs such as GDELT 2.0 frequently enforce aggressive HTTP 429 rate limiting.
   - The platform includes an automated disk cache fallback to address this, but live mode depends on external network uptime and provider response times.
3. **Static Social Media Sample:**
   - The Twitter feed sample consists of 1,000 historical financial tweets from Kaggle.
   - Due to Twitter/X API access restrictions, live streaming social feeds are simulated via disk replay rather than an active firehose connection.
4. **Headline & Snippet Processing:**
   - NLP classification operates primarily on document headlines and summary snippets rather than full-length text.
   - Subtle nuances, footnotes, and covenants in 100-page loan agreements or bond indentures are not captured.

---

## 2. NLP & Machine Learning Limitations

1. **Sentiment Misinterpretation:**
   - While FinBERT is pre-trained on financial text, sarcastic social media posts, double negatives, and legal rhetoric can lead to misclassified sentiment.
2. **Entity Ambiguity:**
   - Entity recognition relies on dictionary and regex lookups against standard corporate tickers and GICS sectors.
   - Multi-conglomerate entities or subsidiaries with naming variations may fail to map to the correct counterparty node in the exposure graph.
3. **LLM Dependency & Rule Fallback:**
   - Zero-shot classification via Gemini 2.0 Flash requires an active API key and internet connectivity.
   - When keys are missing or requests time out, the system falls back to heuristic rule-based classifiers, which provide broader categorization accuracy.

---

## 3. Financial Modeling & Valuation Limitations

1. **First-Order Linear Sensitivity for Bonds:**
   - Bond positions are revalued using modified duration and credit spread duration:
     $$dP = -\text{ModDur} \cdot dy - \text{SpreadDur} \cdot d\text{spread}$$
   - This formula omits the second-order convexity term ($+\frac{1}{2} \text{Convexity} \cdot dy^2$).
   - For yield movements under 100 basis points, the error is negligible ($< 0.15\%$). However, during severe interest rate shocks ($> 250\text{ bps}$), it overestimates bond price declines.
2. **Merton-Type Loan PD Approximation:**
   - The exponential default intensity formula ($PD_{\text{stressed}} = \min(1.0, PD \cdot \exp(k \cdot \text{Intensity}))$) is a simplified hazard rate model.
   - It does not account for collateral liquidation recovery lags, secondary guarantor guarantees, or loan covenant waivers.
3. **Linear Derivative Approximations:**
   - OTC derivatives are limited to linear delta/DV01 sensitivities (FX forwards, interest rate swaps, commodity swaps).
   - Non-linear option Greeks (gamma, vega, theta) and complex exotic structures are not modeled.
4. **Static Portfolio Assumption:**
   - Revaluation assumes the banking book remains static over the 1-day to 5-day stress horizon.
   - In practice, risk managers execute dynamic hedges, margin calls, or portfolio rebalancing to mitigate losses.
5. **Synthetic Portfolio Scope:**
   - The 180 positions ($8.54 Billion exposure) are generated deterministically using synthetic credit profiles.
   - While the rating distributions and asset class weights reflect realistic commercial banking books, they do not represent actual client accounts.

---

## 4. Contagion Graph Limitations

1. **Uniform Network Attenuation:**
   - The exposure graph uses fixed edge weights (`member_of`: 1.0, `located_in`: 0.8, `supplies`: 0.6) and a constant distance decay ($\delta = 0.5$).
   - Real-world financial contagion displays non-linear regime shifts, liquidity freezes, and asymmetric cross-border capital controls that cannot be fully captured by static graph propagation.
2. **Maximum Propagation Depth:**
   - Graph propagation is bounded to 2 hops to prevent artificial dissipation of shocks across the entire network.

---

## 5. Summary of Recommended Enhancements for Production

| Area | Current Implementation | Production Recommendation |
|:---|:---|:---|
| **Data Ingestion** | Cached GDELT + NewsAPI RSS | Bloomberg B-PIPE / Refinitiv Enterprise real-time feed |
| **NLP Context** | Headlines + Snippets (512 tokens) | Long-context LLM processing full regulatory filings |
| **Calibration** | 20 crisis events (Isotonic Regression) | 500+ crisis events with cross-asset factor covariance matrix |
| **Bond Pricing** | Modified Duration + Spread Duration | Full cash flow discounting with yield curve spline interpolation |
| **Derivatives** | Linear Delta & DV01 | Full Monte Carlo simulation with volatility surface revaluation |
| **Network Graph** | NetworkX static graph ($h=2$) | Dynamic interbank credit matrix with clearinghouse netting |
