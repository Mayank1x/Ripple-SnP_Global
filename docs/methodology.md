# Ripple Risk Engine & Stress Testing — Complete Methodology & Formulas

This document details the mathematical models, formulas, and operational assumptions underlying the **Ripple AI/NLP Risk Engine** and the **Module B Strategic Stress Engine**.

---

## 1. System Overview & Data Pipeline

Ripple transforms raw, unstructured financial text into portfolio-level economic loss estimates through five sequential stages:

```
[Raw Feeds] -> [NLP Extraction] -> [Clustering & Novelty] -> [Impact Calibration] -> [Portfolio Stress Engine]
```

---

## 2. Ingestion & Deduplication

### 2.1 Content Normalization
Incoming articles from GDELT 2.0, NewsAPI/RSS, and Twitter feeds are cleaned:
- URLs are stripped of tracking parameters (`utm_*`, `ref`) and converted to lowercase.
- Publication timestamps are parsed into standardized UTC ISO-8601 strings.
- Non-English documents are filtered out using source metadata flags or basic language detection.

### 2.2 Deduplication Rules
A document is classified as a duplicate and excluded if:
1. **Exact Match**: Its normalized URL MD5 hash matches an existing document in the current session:
   $$\text{Hash}(u) = \text{MD5}(\text{clean\_url}(u))$$
2. **Text Jaccard Similarity**: The title Jaccard word similarity with an existing document published within 24 hours exceeds $\theta = 0.85$:
   $$J(A, B) = \frac{|W_A \cap W_B|}{|W_A \cup W_B|} \ge 0.85$$

---

## 3. NLP Feature Extraction

### 3.1 FinBERT Sentiment Scoring
Ripple uses `ProsusAI/finbert`, a BERT model fine-tuned on financial text, to produce class probabilities:
$$P(\text{Positive}), \quad P(\text{Negative}), \quad P(\text{Neutral})$$

The scalar sentiment score is normalized to the $[-1.0, +1.0]$ range:
$$\text{Score}_{\text{sentiment}} = P(\text{Positive}) - P(\text{Negative})$$

Model confidence is defined as the maximum class probability:
$$\text{Confidence} = \max\left(P(\text{Positive}), P(\text{Negative}), P(\text{Neutral})\right)$$

### 3.2 Named Entity Recognition (NER) & Taxonomy Mapping
Entities are identified through dictionary and regex mapping:
- **Companies / Tickers**: Mapped to equity tickers and counterparties (e.g., "JPMorgan" $\rightarrow$ `JPM`, "Credit Suisse" $\rightarrow$ `CS`).
- **Sectors**: Standardized against 11 GICS sectors (Financials, Energy, Technology, Industrials, etc.).
- **Countries**: Mapped to ISO country names and regional trading blocs.
- **Commodities**: Extracted for crude oil, natural gas, copper, and gold.

### 3.3 Event Classification
Articles are categorized into one of 10 financial crisis taxonomies:
- Credit Event / Default
- Regulatory Action / Fine
- Geopolitical Conflict / Sanctions
- Macroeconomic / Inflation / Rate Hike
- Supply Chain Disruption
- Cyber Attack / IT Outage
- Legal Dispute / Lawsuit
- Executive Departure / Governance
- Earnings Miss / Revenue Warning
- Other / General Corporate

Classification runs via Gemini 2.0 Flash with a local keyword/rule heuristic fallback when API keys are absent or rate limits occur.

---

## 4. Clustering, Novelty, and Sentiment Divergence

### 4.1 Rolling FAISS Clustering
Articles are converted into 384-dimensional dense vectors using `all-MiniLM-L6-v2`:
$$\mathbf{e}_i = \text{Embedding}(\text{title}_i), \quad \|\mathbf{e}_i\|_2 = 1.0$$

Clusters represent real-world events. An incoming document $\mathbf{e}_{\text{new}}$ at time $t$ joins an existing cluster $C_k$ with centroid $\mathbf{c}_k$ if:
1. Time distance is within the rolling window: $|t - t_{\text{latest}, k}| \le 7\text{ days}$.
2. Cosine similarity with the cluster centroid meets the threshold:
   $$\text{sim}(\mathbf{e}_{\text{new}}, \mathbf{c}_k) = \mathbf{e}_{\text{new}} \cdot \mathbf{c}_k \ge 0.75$$

When a document joins, the cluster centroid is updated:
$$\mathbf{c}_{k, \text{new}} = \frac{N \mathbf{c}_{k, \text{old}} + \mathbf{e}_{\text{new}}}{N + 1}$$

### 4.2 Event Novelty Metric
Novelty measures how distinct an event is from stories published over the prior 30 days:
$$\text{Novelty}(e) = 1.0 - \max_{p \in \text{Priors}} \left(\mathbf{e} \cdot \mathbf{p}\right)$$
- If the event is identical to prior stories, $\text{Novelty} \approx 0.0$.
- If the event is unprecedented, $\text{Novelty} \approx 1.0$.

### 4.3 News vs. Social Divergence Flag
When both traditional news (GDELT, NewsAPI) and social feeds (Twitter) cover the same event cluster, Ripple computes average sentiments:
$$\bar{s}_{\text{news}} = \frac{1}{|D_{\text{news}}|} \sum_{d \in D_{\text{news}}} s_d, \quad \bar{s}_{\text{social}} = \frac{1}{|D_{\text{social}}|} \sum_{d \in D_{\text{social}}} s_d$$

A divergence flag is raised when narrative sentiment conflicts:
$$\mathbf{1}_{\text{divergence}} = \begin{cases} 1 & \text{if } |\bar{s}_{\text{news}} - \bar{s}_{\text{social}}| \ge 0.40 \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Event Study & Empirical Calibration

### 5.1 Market Model Estimation
For events with a linked company or sector ETF, abnormal returns are estimated against a benchmark index ($R_{m,t} =$ `^GSPC` for US, `^NSEI` for India) over an estimation window of $T = [-120, -21]$ trading days:
$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t}$$

Parameters $\hat{\alpha}_i$ and $\hat{\beta}_i$ are estimated via Ordinary Least Squares (OLS).

### 5.2 Abnormal Returns & Volatility Shift
Abnormal return ($AR$) on day $t \in \{0, +1, +3\}$ is:
$$AR_{i,t} = R_{i,t} - \left(\hat{\alpha}_i + \hat{\beta}_i R_{m,t}\right)$$

Cumulative Abnormal Return ($CAR$):
$$CAR_{0,1} = AR_{i,0} + AR_{i,1}$$
$$CAR_{0,3} = \sum_{t=0}^{3} AR_{i,t}$$

Volatility change compares post-event standard deviation against pre-event:
$$\Delta \sigma = \sigma_{\text{post}} - \sigma_{\text{pre}}$$

### 5.3 Realized Absolute Move
The target variable for calibration combines price move magnitude and volatility shift:
$$Y = |CAR_{0,1}| + 0.35 \cdot \max(0.0, \; \Delta \sigma)$$

### 5.4 Raw Impact Score
Raw uncalibrated impact combines five NLP signals:
$$\text{Raw Impact} = 0.35 |\text{sentiment}| + 0.20 \text{confidence} + 0.20 w_{\text{event}} + 0.15 \text{novelty} + 0.10 \mathbf{1}_{\text{divergence}}$$

Where $w_{\text{event}}$ is the crisis severity weight (Credit Event: $1.0$, Regulatory: $0.85$, Cyber: $0.80$, Other: $0.40$).

### 5.5 Isotonic Regression Calibration
Isotonic regression fits a non-decreasing, non-parametric step function:
$$\min \sum_{j} \left(Y_j - \hat{f}(\text{Raw Impact}_j)\right)^2 \quad \text{subject to } \hat{f}(x_a) \le \hat{f}(x_b) \text{ for } x_a \le x_b$$

The calibrated output is binned into integer deciles from $1.0$ (minimal impact, expected $|CAR| < 0.2\%$) to $10.0$ (systemic crisis, expected $|CAR| > 4.5\%$).

---

## 6. Exposure Graph & Contagion Propagation

### 6.1 Network Topology
The exposure graph $G = (V, E)$ consists of:
- **Nodes ($V$)**: Companies (180 counterparties), GICS Sectors (11), Jurisdictions (15 countries), and Commodities (4).
- **Edges ($E$)**:
  - `member_of` (Company $\rightarrow$ Sector, weight $W = 1.0$)
  - `located_in` (Company $\rightarrow$ Country, weight $W = 0.8$)
  - `supplies` (Company $\rightarrow$ Customer Company, weight $W = 0.6$)
  - `sensitive_to` (Sector $\rightarrow$ Commodity, weight $W = 0.7$)

### 6.2 Propagation Algorithm
A shock initiates at seed nodes identified by the event's entities. Intensity decays with graph distance:
$$\text{Intensity}_v = \max_{u \in \text{neighbors}(v)} \left(\text{Intensity}_u \cdot W_{uv} \cdot \delta^h\right)$$

Where:
- $\delta = 0.5$ is the distance attenuation factor.
- $h \in \{1, 2\}$ is the maximum hop limit.
- $\text{Intensity} \in [0.0, 1.0]$.

---

## 7. Multi-Asset Revaluation Formulas (Module B)

### 7.1 Historical Analog Retrieval & Bayesian Shrinkage
For a triggered event (`impact_score` >= 7.0), the system retrieves $k$ historical crisis analogs from a FAISS index of past events.
If fewer than $N = 3$ analogs are matched, factor shocks are shrunk toward a documented prior table:
$$\text{Shock}_{\text{shrunk}} = \lambda \cdot \text{Shock}_{\text{analog}} + (1 - \lambda) \cdot \text{Shock}_{\text{prior}}, \quad \lambda = \frac{\min(k, N)}{N}$$

Shocks are computed at the 5th percentile (p5), median (p50), and 95th percentile (p95).

### 7.2 Corporate Loans Book
Credit risk on corporate loans uses the Merton/Vasicek structural credit framework:
- Exposure at Default ($EAD$)
- Loss Given Default ($LGD$)
- Baseline 1-year Probability of Default ($PD$)

Network contagion and sector distress elevate the borrower's default intensity:
$$PD_{\text{stressed}} = \min\left(1.0, \; PD \cdot \exp\left(k_{\text{loan}} \cdot \text{Intensity}_{\text{sector}}\right)\right)$$

Where $k_{\text{loan}} = 2.0$ is the loan sensitivity factor.

The incremental expected credit loss is:
$$\text{Loss}_{\text{Loan}} = EAD \cdot LGD \cdot \left(PD_{\text{stressed}} - PD\right)$$

*Property*: When $\text{Intensity} = 0$, $PD_{\text{stressed}} = PD$, and credit loss is zero.

### 7.3 Corporate Bonds Book
Corporate bonds are revalued using duration-based sensitivity to benchmark yields and credit spreads:
$$dP = -\text{ModDur} \cdot dy - \text{SpreadDur} \cdot d\text{spread}$$

Where:
- $\text{ModDur}$: Modified Duration in years.
- $\text{SpreadDur}$: Spread Duration in years.
- $dy$: Benchmark yield shift (e.g., $+25\text{ bps} = 0.0025$).
- $d\text{spread}$: Issuer credit spread shift (e.g., $+45\text{ bps} = 0.0045$).

The mark-to-market loss is:
$$\text{Loss}_{\text{Bond}} = \text{MTM}_{\text{pre}} \cdot \max(0.0, \; -dP)$$

*Property*: When $dy = 0$ and $d\text{spread} = 0$, $dP = 0$, and bond loss is zero.

### 7.4 OTC Derivatives Book
Derivatives are priced based on linear risk sensitivities:
1. **Foreign Exchange (FX) Forwards**:
   $$\text{Loss}_{\text{FX}} = \text{Notional} \cdot |d\text{FX}|$$
2. **Interest Rate Swaps (IRS)**:
   $$\text{Loss}_{\text{IRS}} = \text{DV01} \cdot |dy \cdot 10{,}000|$$
3. **Commodity Swaps**:
   $$\text{Loss}_{\text{Commodity}} = \text{Notional} \cdot |d\text{Price}|$$

*Property*: When all factor changes are zero, derivative losses evaluate to zero.

---

## 8. Summary of Baseline Model Parameters

| Parameter | Default Value | Description |
|:---|:---:|:---|
| Clustering Threshold | $0.75$ | Minimum FAISS cosine similarity to join cluster |
| Clustering Window | 7 days | Maximum lookback for rolling temporal cluster |
| Sentiment Divergence Threshold | $0.40$ | Absolute difference between news and social sentiment |
| Stress Trigger Threshold | $7.0$ | Minimum calibrated impact score to execute Module B |
| Analog Sample Size ($N$) | $3$ | Minimum analogs before Bayesian shrinkage applies |
| Network Contagion Decay ($\delta$) | $0.50$ | Attenuation per hop in exposure graph |
| Contagion Max Hops | $2$ | Maximum propagation depth in network |
| Loan Sensitivity ($k_{\text{loan}}$) | $2.0$ | Multiplier for default intensity acceleration |
| Estimation Window | 120 days | Pre-event window for market model OLS calibration |
| Event Study Window | [0, +3] | Forward days evaluated for cumulative abnormal return |
