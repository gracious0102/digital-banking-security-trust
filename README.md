# Digital Banking: Security, Privacy & Customer Trust

A reproducible research and data-analysis project examining how **perceived security,
data privacy, and trust** shape customer adoption and satisfaction with digital banking
(mobile apps, UPI, internet banking, and ATMs), based on a primary survey of
**199 respondents** in Guwahati, India.

The project frames its findings as evidence for **consumer-protection and
digital-governance** recommendations — the intersection of cybersecurity and public policy.

> Author: **Grace Ibeji** ([github.com/gracious0102](https://github.com/gracious0102))
> Context: Undergraduate dissertation extended into a reproducible analysis pipeline for
> an MS in Cybersecurity & Public Policy application portfolio.

---

## Why this project

Digital banking has moved money, identity, and daily financial decisions onto phones and
apps. Whether people adopt these services depends not only on convenience but on whether
they **trust** them to be **secure** and to **protect their personal data**. Those trust
and security perceptions are exactly what cybersecurity policy and consumer-protection
regulation try to safeguard.

This project asks:

1. How widely are digital-banking channels adopted, and how often are they used?
2. How do customers perceive the **security** and **data privacy** of these services?
3. Where are the **trust gaps** (e.g. fraud concern, grievance resolution) that policy
   and institutions should address?
4. What **consumer-protection and digital-governance** actions do the findings support?

## What it does

- Loads survey data (demographics, channel usage, and Likert-scale items covering
  security, privacy, trust, and satisfaction)
- Computes a **Security–Privacy–Trust (SPT) index** and per-item summaries
- Identifies the weakest trust/security dimensions (policy-relevant gaps)
- Produces publication-ready charts
- Writes a markdown analysis summary and a **policy recommendations** brief

## Project structure

```
digital-banking-security-trust/
├── data/
│   ├── raw/                 # Put your real survey CSV here (git-ignored)
│   └── sample/              # 199-row synthetic dataset (thesis-aligned)
├── src/
│   ├── config.py            # Column names, Likert scale, paths
│   ├── generate_sample_data.py
│   ├── load_data.py
│   ├── analyze.py           # Security / privacy / trust analysis
│   └── visualize.py
├── scripts/run_analysis.py  # End-to-end pipeline entry point
├── reports/                 # Policy recommendations brief
└── outputs/
    ├── figures/             # PNG charts
    └── tables/              # CSV summary tables
```

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/run_analysis.py
```

Outputs are written to `outputs/figures/`, `outputs/tables/`, and
`outputs/reports/analysis_summary.md`.

## Using your own survey data

1. Export your survey responses as CSV.
2. Save as `data/raw/survey_responses.csv` (column names must match `src/config.py`).
3. Run: `python scripts/run_analysis.py --data data/raw/survey_responses.csv`

## Likert scale

| Code | Label             |
|------|-------------------|
| 1    | Strongly Agree    |
| 2    | Agree             |
| 3    | Neutral           |
| 4    | Disagree          |
| 5    | Strongly Disagree |

## Ethics & data

The included dataset under `data/sample/` is **synthetic** — generated to match the
aggregate distributions reported in the original study — so no real respondent data is
published. Real survey responses (`data/raw/`) are git-ignored.

## License

Released under the MIT License. See [LICENSE](LICENSE).
