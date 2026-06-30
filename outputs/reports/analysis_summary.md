# Digital Banking Cybersecurity: Risk, Trust & Threat Intelligence

**Author:** Grace Ibeji  
**Repository:** https://github.com/gracious0102/digital-banking-security-trust  
**Sample:** 10,000 respondents  
**Dataset:** Synthetic, correlated (10k-scale)

## Executive Summary

Population attack surface risk level: **High** (score 65.7/100).  
Fraud incidence: **31.9%** of respondents.  
MFA coverage: **62.6%** — the most critical gap.

## Key Insights

- Top channel: Mobile Banking App (87.1% adoption) — highest attack-surface priority.
- 31.9% of respondents have experienced digital banking fraud — exceeding industry benchmarks.
- Only 62.6% have MFA enabled — a critical security gap across all risk segments.
- SPT index mean: 3.396 — 3.9% of users are positive on security/trust.
- Weakest SPT dimension: 'Timely grievance resolution' (15.5% positive, mean 3.84).
- Only 41.1% of respondents score ≥70/100 on the composite Security Posture Index.
- Platform trust: 30.3% positive — driven by convenience but undermined by incident resolution failures.

## Security–Privacy–Trust (SPT) Index

- Mean SPT score: **3.396** (1 = most positive, 5 = most negative)
- Positive responses (≤ 2.0): **3.9%**
- Negative responses (≥ 4.0): **27.2%**
- Weakest dimension: **Timely grievance resolution** (15.5% positive, mean 3.84)

## Service Adoption

| channel                |   adopters |   adoption_pct |
|:-----------------------|-----------:|---------------:|
| Mobile Banking App     |       8710 |           87.1 |
| UPI / Digital Payments |       8522 |           85.2 |
| Internet Banking       |       7215 |           72.2 |
| ATM Services           |       5452 |           54.5 |
| NEFT / RTGS            |       5246 |           52.5 |
| Aadhaar Pay            |       2910 |           29.1 |

## SPT Dimension Summary

| column                    | question                              |   mean_score |   std |   strongly_agree_pct |   positive_pct |   neutral_pct |   negative_pct |
|:--------------------------|:--------------------------------------|-------------:|------:|---------------------:|---------------:|--------------:|---------------:|
| q_feel_secure             | Overall feeling of security           |        3.256 | 1.307 |                 12   |           30   |          24.4 |           45.6 |
| q_security_features       | Adequacy of security features         |        3.349 | 1.31  |                 10.9 |           27.6 |          23.9 |           48.5 |
| q_privacy_policy_clarity  | Privacy policy is clear/transparent   |        3.556 | 1.268 |                  8   |           21.9 |          22.7 |           55.4 |
| q_mfa_effectiveness       | MFA / two-step login is effective     |        2.789 | 1.296 |                 20.8 |           43.1 |          26.1 |           30.8 |
| q_data_breach_concern     | Concern about data breaches           |        3.448 | 1.266 |                  8.6 |           24.2 |          24.8 |           51   |
| q_incident_response_speed | Bank responds quickly to incidents    |        3.736 | 1.217 |                  5.6 |           17.2 |          22.2 |           60.6 |
| q_third_party_concern     | Concern over third-party data sharing |        3.295 | 1.278 |                 10.4 |           28.3 |          25.6 |           46.1 |
| q_trust_platform          | Trust in digital banking platform     |        3.262 | 1.317 |                 11.9 |           30.3 |          24.2 |           45.4 |
| q_grievance_resolution    | Timely grievance resolution           |        3.84  | 1.197 |                  4.8 |           15.5 |          19.9 |           64.6 |
| q_regulatory_trust        | Trust in regulators to protect data   |        3.427 | 1.249 |                  8.3 |           24.3 |          25.2 |           50.5 |

## Channel Risk Matrix

| channel          |   adoption_pct |   composite_risk_score |   weighted_risk_score |
|:-----------------|---------------:|-----------------------:|----------------------:|
| Mobile Banking   |           87.1 |                     80 |                  69.7 |
| UPI / Payments   |           85.2 |                     76 |                  64.8 |
| Internet Banking |           72.2 |                     80 |                  57.7 |
| NEFT/RTGS        |           52.5 |                     59 |                  31   |
| ATM Services     |           54.5 |                     55 |                  30   |
| Aadhaar Pay      |           29.1 |                     64 |                  18.6 |

## Security Gap Priority

| gap                              |   severity |   prevalence_pct |   priority_score | mitigation                                                        |
|:---------------------------------|-----------:|-----------------:|-----------------:|:------------------------------------------------------------------|
| No MFA enabled                   |          5 |             37.4 |             1.87 | Mandate MFA for all high-value transactions                       |
| Low phishing awareness           |          4 |             34.7 |             1.39 | Mandatory in-app phishing-simulation training                     |
| No fraud reported after incident |          5 |             27.7 |             1.38 | Streamline one-tap fraud reporting; proactive outreach to victims |
| App not regularly updated        |          3 |             39.1 |             1.17 | Enforce minimum app version before login                          |
| No biometric authentication      |          3 |             34.6 |             1.04 | Default-enable biometrics on supported devices                    |
| Shared device usage              |          4 |             17.1 |             0.68 | Enforce session expiry + device-binding alerts                    |
| Weak password hygiene            |          4 |             11.3 |             0.45 | Enforce password complexity + breach-check on login               |

## ML Results

### Fraud Risk Classification

| model               |   accuracy |     f1 |   auc_roc |   cv_accuracy_mean |
|:--------------------|-----------:|-------:|----------:|-------------------:|
| Logistic Regression |     0.984  | 0.984  |    0.999  |             0.9779 |
| Random Forest       |     0.9075 | 0.9079 |    0.9616 |             0.8945 |
| XGBoost             |     0.995  | 0.995  |    1      |             0.9948 |

### Trust Prediction

| model               |   accuracy |     f1 |   cv_accuracy_mean |
|:--------------------|-----------:|-------:|-------------------:|
| Logistic Regression |     0.6105 | 0.606  |             0.6038 |
| Random Forest       |     0.5255 | 0.5235 |             0.5215 |
| XGBoost             |     0.5275 | 0.5294 |             0.5159 |

### Anomaly Detection

- Anomalies flagged: **800** (8.0%)

| tier     |   count |   pct |
|:---------|--------:|------:|
| Critical |      88 |   0.9 |
| High     |     994 |   9.9 |
| Medium   |    3473 |  34.7 |
| Normal   |    5445 |  54.4 |