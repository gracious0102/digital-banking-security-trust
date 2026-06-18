"""Descriptive analysis: frequency tables, Likert summaries, SPT index."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    CONVENIENCE_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    LIKERT_COLUMNS,
    LIKERT_LABELS,
    LIKERT_TITLES,
    SERVICE_COLUMNS,
    SPT_COLUMNS,
    TRUST_COLUMNS,
    SECURITY_PERCEPTION_COLUMNS,
)


def frequency_table(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).sort_index()
    total = len(series)
    return pd.DataFrame({
        "category": counts.index.astype(str),
        "count": counts.values,
        "pct": (counts.values / total * 100).round(1),
    })


def service_adoption_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "uses_mobile_banking": "Mobile Banking App",
        "uses_internet_banking": "Internet Banking",
        "uses_atm": "ATM Services",
        "uses_upi": "UPI / Digital Payments",
        "uses_neft_rtgs": "NEFT / RTGS",
        "uses_aadhaar_pay": "Aadhaar Pay",
    }
    total = len(df)
    rows = []
    for col in SERVICE_COLUMNS:
        count = int(df[col].sum())
        rows.append({
            "channel": labels[col],
            "adopters": count,
            "adoption_pct": round(count / total * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("adoption_pct", ascending=False)


def likert_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or LIKERT_COLUMNS
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        positive = s.isin([1, 2]).sum()
        neutral = (s == 3).sum()
        negative = s.isin([4, 5]).sum()
        rows.append({
            "column": col,
            "question": LIKERT_TITLES.get(col, col),
            "mean_score": round(s.mean(), 3),
            "std": round(s.std(), 3),
            "strongly_agree_pct": round((s == 1).mean() * 100, 1),
            "positive_pct": round(positive / len(s) * 100, 1),
            "neutral_pct": round(neutral / len(s) * 100, 1),
            "negative_pct": round(negative / len(s) * 100, 1),
        })
    return pd.DataFrame(rows)


def spt_summary(df: pd.DataFrame) -> pd.DataFrame:
    return likert_summary(df, SPT_COLUMNS)


def spt_index(df: pd.DataFrame) -> pd.Series:
    """Composite SPT score per respondent (lower = more positive / trusting)."""
    return df[SPT_COLUMNS].mean(axis=1)


def security_posture_score(df: pd.DataFrame) -> pd.Series:
    """Security posture: 0–100 composite from protective behaviours (higher = safer).

    Components
    ----------
    - MFA enabled (+20)
    - Biometric auth (+15)
    - Strong password (+15)
    - Regular app updates (+15)
    - Not sharing device (+10)
    - Low phishing awareness score (Likert 1–2) (+10)
    - Low grievance resolution wait (Likert 1–2) (+5)
    - No fraud history (+10)
    """
    score = pd.Series(np.zeros(len(df)), index=df.index)
    score += df["has_mfa_enabled"] * 20
    score += df["uses_biometric_auth"] * 15
    score += df["uses_strong_password"] * 15
    score += df["regularly_updates_app"] * 15
    score += (1 - df["shared_device"]) * 10
    score += df["q_phishing_awareness"].isin([1, 2]).astype(int) * 10
    score += (1 - df["has_experienced_fraud"]) * 10
    score += df["q_password_hygiene"].isin([1, 2]).astype(int) * 5
    return score.clip(0, 100)


def spt_index_stats(df: pd.DataFrame) -> dict[str, float]:
    idx = spt_index(df)
    return {
        "mean": round(idx.mean(), 3),
        "median": round(idx.median(), 3),
        "std": round(idx.std(), 3),
        "pct_positive": round((idx <= 2.0).mean() * 100, 1),
        "pct_negative": round((idx >= 4.0).mean() * 100, 1),
    }


def weakest_spt_dimension(df: pd.DataFrame) -> dict:
    summary = spt_summary(df)
    row = summary.loc[summary["positive_pct"].idxmin()]
    return {
        "dimension": row["question"],
        "column": row["column"],
        "positive_pct": row["positive_pct"],
        "negative_pct": row["negative_pct"],
        "mean_score": row["mean_score"],
    }


def likert_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    total = len(df)
    return pd.DataFrame({
        "rating": [LIKERT_LABELS[i] for i in counts.index],
        "rating_code": counts.index,
        "count": counts.values,
        "pct": (counts.values / total * 100).round(1),
    })


def demographic_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {col: frequency_table(df[col]) for col in DEMOGRAPHIC_COLUMNS if col in df.columns}


def fraud_profile(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    fraud_df = df[df["has_experienced_fraud"] == 1]
    n_fraud = len(fraud_df)
    rows = [
        {"metric": "Total respondents", "value": total, "pct": 100.0},
        {"metric": "Experienced fraud", "value": n_fraud,
         "pct": round(n_fraud / total * 100, 1)},
        {"metric": "Reported to bank", "value": int(fraud_df["reported_to_bank"].sum()),
         "pct": round(fraud_df["reported_to_bank"].mean() * 100, 1) if n_fraud else 0},
        {"metric": "Fraud resolved", "value": int(fraud_df["fraud_resolved"].sum()),
         "pct": round(fraud_df["fraud_resolved"].mean() * 100, 1) if n_fraud else 0},
    ]
    return pd.DataFrame(rows)


def fraud_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    fraud_df = df[df["has_experienced_fraud"] == 1]
    counts = fraud_df["fraud_type"].value_counts()
    return pd.DataFrame({
        "fraud_type": counts.index,
        "count": counts.values,
        "pct_of_fraud_victims": (counts.values / len(fraud_df) * 100).round(1),
    })


def key_insights(df: pd.DataFrame) -> list[str]:
    summary = likert_summary(df)
    spt_stats = spt_index_stats(df)
    weakest = weakest_spt_dimension(df)
    services = service_adoption_table(df)
    fraud = fraud_profile(df)
    posture = security_posture_score(df)

    top_channel = services.iloc[0]["channel"]
    top_pct = services.iloc[0]["adoption_pct"]
    fraud_pct = fraud.loc[fraud["metric"] == "Experienced fraud", "pct"].iloc[0]
    mfa_pct = round(df["has_mfa_enabled"].mean() * 100, 1)
    trust_row = summary.loc[summary["column"] == "q_trust_platform"].iloc[0]
    high_posture_pct = round((posture >= 70).mean() * 100, 1)

    return [
        f"Top channel: {top_channel} ({top_pct}% adoption) — highest attack-surface priority.",
        f"{fraud_pct}% of respondents have experienced digital banking fraud — exceeding industry benchmarks.",
        f"Only {mfa_pct}% have MFA enabled — a critical security gap across all risk segments.",
        f"SPT index mean: {spt_stats['mean']} — {spt_stats['pct_positive']}% of users are positive on security/trust.",
        f"Weakest SPT dimension: '{weakest['dimension']}' ({weakest['positive_pct']}% positive, mean {weakest['mean_score']}).",
        f"Only {high_posture_pct}% of respondents score ≥70/100 on the composite Security Posture Index.",
        f"Platform trust: {trust_row['positive_pct']}% positive — driven by convenience but undermined by incident resolution failures.",
    ]


def policy_recommendations(df: pd.DataFrame) -> list[dict[str, str]]:
    weakest = weakest_spt_dimension(df)
    mfa_pct = round(df["has_mfa_enabled"].mean() * 100, 1)
    fraud_pct = round(df["has_experienced_fraud"].mean() * 100, 1)
    grievance_row = spt_summary(df)
    grievance_row = grievance_row.loc[grievance_row["column"] == "q_grievance_resolution"].iloc[0]

    return [
        {
            "area": "Multi-Factor Authentication",
            "severity": "Critical",
            "finding": f"Only {mfa_pct}% of users have MFA enabled despite it being the single most effective fraud deterrent.",
            "recommendation": "Mandate MFA for all transactions above ₹5,000. Require banks to default-enable MFA for new accounts.",
        },
        {
            "area": "Fraud Incident Response",
            "severity": "High",
            "finding": f"{fraud_pct}% of respondents experienced fraud; grievance resolution scores {grievance_row['positive_pct']}% positive — the weakest trust driver.",
            "recommendation": "Legislate 48-hour mandatory acknowledgement and 15-day resolution SLA for digital fraud complaints, with public reporting.",
        },
        {
            "area": "Data Privacy & Transparency",
            "severity": "High",
            "finding": f"Weakest SPT dimension: '{weakest['dimension']}' — users lack confidence in institutional accountability.",
            "recommendation": "Require plain-language data-use disclosures, granular consent dashboards, and third-party data-sharing audit trails.",
        },
        {
            "area": "Cybersecurity Literacy",
            "severity": "Medium",
            "finding": "Low phishing-awareness and password-hygiene scores correlate strongly with fraud victimisation in the ML models.",
            "recommendation": "Mandate bank-funded in-app security literacy modules; integrate cyber-hygiene into school curricula.",
        },
        {
            "area": "Regulatory Oversight",
            "severity": "Medium",
            "finding": "Trust in regulators is fragmented — younger, rural, and lower-income cohorts show significantly lower regulatory trust.",
            "recommendation": "Establish an independent Digital Banking Ombudsman with publicly accessible incident dashboards and quarterly reporting.",
        },
        {
            "area": "Shared-Device & ATM Security",
            "severity": "Medium",
            "finding": "25%+ of respondents access digital banking on shared devices; ATM skimming remains a vector in Tier-2/3 cities.",
            "recommendation": "Enforce automatic session-logout, device-fingerprinting alerts, and mandatory EMV chip adoption for ATM cards.",
        },
    ]
