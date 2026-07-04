"""Security, privacy, and trust analysis for digital banking survey data."""

from __future__ import annotations

import pandas as pd

from config import (
    DEMOGRAPHIC_COLUMNS,
    LIKERT_COLUMNS,
    LIKERT_LABELS,
    LIKERT_TITLES,
    PRIVACY_RELATED_COLUMNS,
    SECURITY_COLUMNS,
    SERVICE_COLUMNS,
    SPT_COLUMNS,
    SPT_TITLES,
    TRUST_COLUMNS,
)


def frequency_table(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).sort_index()
    total = len(series)
    return pd.DataFrame({
        "category": counts.index.astype(str),
        "count": counts.values,
        "percentage": (counts.values / total * 100).round(1),
    })


def service_adoption_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "uses_mobile_banking": "Mobile Banking App",
        "uses_internet_banking": "Internet Banking",
        "uses_atm": "ATM Services",
        "uses_upi": "Digital Payment Apps (UPI)",
    }
    total = len(df)
    rows = []
    for col in SERVICE_COLUMNS:
        count = int(df[col].sum())
        rows.append({
            "service": labels[col],
            "count": count,
            "percentage": round(count / total * 100, 1),
        })
    return pd.DataFrame(rows)


def likert_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or LIKERT_COLUMNS
    rows = []
    for col in columns:
        series = df[col]
        positive = series.isin([1, 2]).sum()
        neutral = (series == 3).sum()
        negative = series.isin([4, 5]).sum()
        rows.append({
            "question": LIKERT_TITLES[col],
            "column": col,
            "mean_score": round(series.mean(), 2),
            "strongly_agree_pct": round((series == 1).mean() * 100, 1),
            "positive_pct": round(positive / len(series) * 100, 1),
            "neutral_pct": round(neutral / len(series) * 100, 1),
            "negative_pct": round(negative / len(series) * 100, 1),
        })
    return pd.DataFrame(rows)


def security_privacy_trust_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise only Security–Privacy–Trust (SPT) Likert items."""
    return likert_summary(df, SPT_COLUMNS)


def spt_index(df: pd.DataFrame) -> pd.Series:
    """Composite Security–Privacy–Trust score per respondent (lower = more positive)."""
    return df[SPT_COLUMNS].mean(axis=1)


def spt_index_stats(df: pd.DataFrame) -> dict[str, float]:
    index = spt_index(df)
    return {
        "mean": round(index.mean(), 2),
        "median": round(index.median(), 2),
        "std": round(index.std(), 2),
        "pct_positive": round((index <= 2.0).mean() * 100, 1),
    }


def weakest_spt_dimension(df: pd.DataFrame) -> dict[str, str | float]:
    """Return the SPT dimension with the lowest positive response rate."""
    summary = security_privacy_trust_summary(df)
    row = summary.loc[summary["positive_pct"].idxmin()]
    return {
        "dimension": row["question"],
        "positive_pct": row["positive_pct"],
        "negative_pct": row["negative_pct"],
    }


def likert_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    total = len(df)
    return pd.DataFrame({
        "rating": [LIKERT_LABELS[i] for i in counts.index],
        "rating_code": counts.index,
        "count": counts.values,
        "percentage": (counts.values / total * 100).round(1),
    })


def demographic_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {col: frequency_table(df[col]) for col in DEMOGRAPHIC_COLUMNS}


def key_insights(df: pd.DataFrame) -> list[str]:
    """Policy-relevant insights emphasising security, privacy, and trust."""
    summary = likert_summary(df)
    spt = security_privacy_trust_summary(df)
    services = service_adoption_table(df)
    usage = frequency_table(df["usage_frequency"])
    spt_stats = spt_index_stats(df)
    weakest = weakest_spt_dimension(df)

    daily_pct = usage.loc[usage["category"] == "Daily", "percentage"].iloc[0]
    mobile_pct = services.loc[services["service"] == "Mobile Banking App", "percentage"].iloc[0]
    upi_pct = services.loc[services["service"] == "Digital Payment Apps (UPI)", "percentage"].iloc[0]
    satisfaction_row = summary.loc[summary["question"] == "Overall satisfaction"].iloc[0]
    security_row = spt.loc[spt["question"] == "Feeling of security"].iloc[0]
    trust_row = spt.loc[spt["question"] == "Trust in digital platform"].iloc[0]

    return [
        f"{daily_pct}% of respondents use digital banking daily — high adoption increases the policy stakes for security and privacy.",
        f"Mobile banking ({mobile_pct}%) and UPI ({upi_pct}%) are the dominant channels; payments-security policy must cover both.",
        f"{security_row['positive_pct']}% report positive perceptions of security; {trust_row['positive_pct']}% trust the digital platform.",
        f"Weakest SPT dimension: {weakest['dimension']} ({weakest['positive_pct']}% positive) — a priority for consumer-protection policy.",
        f"Composite SPT index: {spt_stats['pct_positive']}% of respondents score positively across security/trust items.",
        f"{satisfaction_row['positive_pct']}% report positive overall satisfaction despite identifiable trust gaps.",
    ]


def policy_recommendations(df: pd.DataFrame) -> list[dict[str, str]]:
    """Generate consumer-protection and digital-governance recommendations from findings."""
    weakest = weakest_spt_dimension(df)
    spt = security_privacy_trust_summary(df)
    grievance = spt.loc[spt["question"] == "Timely grievance resolution"].iloc[0]

    recs = [
        {
            "area": "Consumer Protection",
            "finding": f"Grievance resolution scores {grievance['positive_pct']}% positive — the weakest operational trust driver.",
            "recommendation": "Mandate time-bound digital grievance resolution SLAs and publish bank-wise resolution metrics.",
        },
        {
            "area": "Data Privacy",
            "finding": f"Security perceptions ({spt.loc[spt['question']=='Feeling of security','positive_pct'].iloc[0]}% positive) lag behind convenience drivers.",
            "recommendation": "Require plain-language privacy notices and consent flows before enabling high-risk payment features.",
        },
        {
            "area": "Payments Security",
            "finding": "UPI and mobile banking are the most-used channels, concentrating fraud and phishing risk.",
            "recommendation": "Expand real-time fraud alerts, transaction limits for new devices, and mandatory multi-factor authentication.",
        },
        {
            "area": "Digital Governance",
            "finding": f"Overall trust is high but uneven — weakest dimension: {weakest['dimension']}.",
            "recommendation": "Establish independent digital-banking ombudsman reporting with public dashboards on security incidents.",
        },
        {
            "area": "Financial Inclusion & Literacy",
            "finding": "Majority of respondents are under 25 and students — digitally native but potentially less risk-aware.",
            "recommendation": "Integrate cybersecurity and scam-awareness modules into school and university financial-literacy programmes.",
        },
    ]
    return recs
