# Digital Banking Cybersecurity: Risk, Trust & Threat Intelligence

> A production-grade, end-to-end cybersecurity research pipeline — from survey data and
> statistical inference through ML-based risk modelling and CVSS-inspired threat analysis —
> anchored in a 10 000-respondent synthetic dataset built on realistic inter-feature correlations.

**Author:** Grace Ibeji · [github.com/gracious0102](https://github.com/gracious0102)  
**Context:** Deep cybersecurity research project extending an undergraduate dissertation into a
rigorous, reproducible analysis platform for a Cybersecurity & Public Policy portfolio.

---

## Why this project exists

Digital banking has made financial services instant and borderless — and dramatically widened
the attack surface for cybercriminals. Whether a user in Guwahati or Goa, the same vectors apply:
phishing, account takeover, SIM swap, social engineering, credential stuffing. Regulators and
banks need **data** to prioritise defences and policies.

This project bridges the gap between:

| Domain | Contribution |
|---|---|
| Survey research | 50-feature schema capturing security behaviours, fraud incidents, literacy |
| Inferential statistics | Cronbach's α, Spearman correlations, Mann-Whitney, Chi-square, Kruskal-Wallis |
| Machine learning | K-Means segmentation, XGBoost/RF fraud classifier, trust predictor, Isolation Forest |
| Cybersecurity | CVSS-inspired channel risk matrix, attack surface scoring, threat actor profiles |
| Policy | Evidence-based consumer-protection recommendations with severity ratings |

---

## What's new (vs. original dissertation baseline)

| Dimension | Before | After |
|---|---|---|
| Dataset size | 199 rows | **10 000 rows** (correlated synthetic) |
| Features | 17 | **50+** (incidents, device, literacy, MFA, income, region…) |
| Analysis | Likert summaries | + hypothesis tests, scale reliability, Cronbach's α |
| ML | None | K-Means · XGBoost · RF · Isolation Forest · SHAP |
| Cybersecurity | Survey framing | CVSS risk matrix · attack surface · threat actors · gap priority |
| Visualisations | 8 basic charts | **25+ publication charts** (diverging Likert, ROC, SHAP, heatmaps…) |
| Reports | 1 Markdown summary | Analysis report + policy brief (auto-generated) |

---

## Project structure

```
digital-banking-security-trust/
│
├── data/
│   ├── synthetic/               # 10 000-row generated dataset (git-ignored)
│   └── raw/                     # Your real CSV goes here (git-ignored)
│
├── src/
│   ├── config.py                # Paths, schema, risk weights, threat profiles
│   │
│   ├── data/
│   │   ├── generate.py          # Correlated 10k dataset generator (latent factor model)
│   │   └── load.py              # Loader + schema validator
│   │
│   ├── analysis/
│   │   ├── descriptive.py       # Frequency tables, SPT index, security posture score
│   │   ├── statistical.py       # Cronbach α, Spearman, Mann-Whitney U, Chi-square, K-W
│   │   └── cybersecurity.py     # Channel risk matrix, gap priority, threat mapping
│   │
│   ├── ml/
│   │   ├── preprocessing.py     # Feature engineering, ordinal encoding, scaling
│   │   ├── clustering.py        # K-Means + silhouette analysis + PCA projection
│   │   ├── classification.py    # LR / RF / XGBoost for fraud risk + trust prediction
│   │   ├── anomaly.py           # Isolation Forest anomaly detection
│   │   └── explainability.py    # SHAP TreeExplainer
│   │
│   └── visualization/
│       ├── descriptive_plots.py # Demographics, adoption, SPT, posture, fraud charts
│       ├── ml_plots.py          # Cluster scatter, ROC, SHAP, model comparison
│       └── cybersecurity_plots.py # Heatmaps, bubble chart, gap matrix, cohort chart
│
├── scripts/
│   ├── run_full_pipeline.py     # ★ Master orchestrator (all 5 phases)
│   └── run_analysis.py          # Legacy: descriptive-only pipeline
│
├── outputs/
│   ├── figures/                 # 25+ PNG charts
│   ├── tables/                  # CSV summaries and ML metrics
│   ├── models/                  # Serialised joblib models
│   └── reports/
│       ├── analysis_summary.md
│       └── policy_recommendations.md
│
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/gracious0102/digital-banking-security-trust.git
cd digital-banking-security-trust

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (generates data, analyses, ML, reports)
python scripts/run_full_pipeline.py

# 4. Outputs land in outputs/figures/, outputs/tables/, outputs/reports/, outputs/models/
```

### Options

```bash
# Use your own survey CSV
python scripts/run_full_pipeline.py --data path/to/survey.csv

# Custom dataset size
python scripts/run_full_pipeline.py --n 20000

# Skip ML phase (faster, descriptive + cyber only)
python scripts/run_full_pipeline.py --skip-ml
```

---

## Dataset schema (50+ features)

| Group | Columns | Notes |
|---|---|---|
| Demographics | gender, age_group, education, occupation, income_bracket, region, bank_type, years_digital_banking | 8 columns |
| Service adoption | uses_mobile_banking, uses_internet_banking, uses_atm, uses_upi, uses_neft_rtgs, uses_aadhaar_pay | Binary |
| Usage patterns | usage_frequency, transaction_frequency, avg_transaction_amount_tier | Ordinal |
| Security setup | has_mfa_enabled, uses_biometric_auth, uses_strong_password, regularly_updates_app, shared_device, device_type, os_type | Binary + nominal |
| Incidents | has_experienced_fraud, fraud_type, reported_to_bank, fraud_resolved, num_phishing_attempts_received | Mixed |
| Cybersecurity literacy | q_phishing_awareness, q_password_hygiene, q_data_sharing_awareness, q_regulatory_awareness | Likert 1–5 |
| Security perceptions | q_feel_secure, q_security_features, q_privacy_policy_clarity, q_mfa_effectiveness, q_data_breach_concern, q_incident_response_speed, q_third_party_concern | Likert 1–5 |
| Trust | q_trust_platform, q_grievance_resolution, q_regulatory_trust | Likert 1–5 |
| Convenience | q24_7_availability, q_saves_time, q_easy_access, q_reduced_branch_visits, q_speed_efficiency, q_user_friendly, q_overall_satisfaction, q_improved_experience | Likert 1–5 |

**Likert scale:** 1 = Strongly Agree (positive), 5 = Strongly Disagree (negative)

---

## Data generation methodology

The synthetic dataset uses a **correlated latent factor model**:

1. Four Gaussian latent factors are drawn from a Cholesky-decomposed covariance matrix:
   - `tech_savvy` — drives MFA adoption, biometric use, device choice
   - `risk_awareness` — drives phishing literacy, password hygiene
   - `trust_disposition` — drives all trust/security Likert items
   - `fraud_exposure` — drives fraud probability (negatively correlated with tech_savvy)

2. Each feature is derived from a linear combination of relevant factors plus noise, creating
   realistic co-movement:
   - Fraud victims → lower trust scores, higher breach concern
   - MFA users → higher security adequacy ratings
   - High usage + many channels → higher fraud probability
   - Metro users → higher mobile/UPI adoption, lower ATM usage

This ensures ML models find genuinely informative signal — not random noise.

---

## Machine learning pipeline

### 1. User Segmentation (K-Means)
- Features: security posture score, MFA status, literacy, fraud history, trust scores
- k selection: silhouette analysis over k ∈ {2, …, 7}
- Output: 4 risk clusters (Security-Conscious, Active-but-Vulnerable, Skeptical, High-Risk)
- Visualisation: PCA scatter + cluster profile heatmap

### 2. Fraud Risk Classification
- Target: `has_experienced_fraud` (binary)
- Models: Logistic Regression · Random Forest · XGBoost
- Evaluation: Stratified 5-fold CV · Accuracy · F1 · Precision · Recall · AUC-ROC
- Output: fraud risk score per respondent, ROC curves, feature importance

### 3. Trust Prediction
- Target: `q_trust_platform` (ordinal 1–5, treated as classification)
- Same model set as above
- Output: SHAP-ranked trust drivers, model comparison chart

### 4. Anomaly Detection (Isolation Forest)
- Features: security setup + literacy + exposure proxy
- Contamination: 8%
- Output: anomaly score (0–100), tier (Critical/High/Medium/Normal), anomalous-vs-normal profile

### 5. SHAP Explainability
- TreeExplainer for XGBoost/RF; LinearExplainer for LR
- Global bar summary + top-20 feature table
- Interpretable answer to "what drives fraud risk?"

---

## Cybersecurity analysis

### Channel Risk Matrix
CVSS-inspired risk scores across 5 threat vectors (attack surface, malware, phishing,
account takeover, data exfiltration) for 6 banking channels. Weighted by adoption rate
to produce a **population-level attack surface score**.

### Security Gap Priority Matrix
Cross-product of severity (expert-assigned, 1–5) × prevalence (data-derived) to rank
security gaps by urgency. Top gaps: MFA absence, password hygiene, phishing susceptibility.

### Threat Actor Profiles
Five actor archetypes (Phishing Operator, ATO Actor, Insider Threat, APT, Social Engineer)
mapped to channels, tactics, and skill levels — cross-referenced with the most vulnerable
user cohorts identified by the ML pipeline.

### Vulnerable Cohort Analysis
Cross-tabulates vulnerability flags (low posture + no MFA + fraud history) across
demographic segments to surface priority intervention targets.

---

## Statistical methods

| Method | Purpose |
|---|---|
| Cronbach's alpha | Scale reliability for SPT, literacy, convenience sub-scales |
| Spearman correlation | Non-parametric correlation matrix for ordinal Likert items |
| Mann-Whitney U | Compare fraud victims vs non-victims on each SPT dimension |
| Chi-square + Cramér V | Fraud incidence vs gender, age, income, region, bank type |
| Kruskal-Wallis + η² | Trust variation across income / region segments |

---

## Output artefacts

| Path | Contents |
|---|---|
| `outputs/figures/` | 25+ PNG charts |
| `outputs/tables/` | CSV: service adoption, Likert summary, SPT, cluster profiles, risk matrix, ML metrics, SHAP, anomaly summary |
| `outputs/models/` | `kmeans.joblib`, `fraud_classifier.joblib`, `trust_classifier.joblib` |
| `outputs/reports/analysis_summary.md` | Full narrative with tables and ML results |
| `outputs/reports/policy_recommendations.md` | Consumer-protection policy brief |

---

## Ethics & data

All data under `data/synthetic/` is **fully synthetic** — generated via a latent factor
model to match realistic distributions. No real respondent PII is published or committed.
Real survey CSVs placed in `data/raw/` are `.gitignore`d.

---

## License

MIT — see [LICENSE](LICENSE).
