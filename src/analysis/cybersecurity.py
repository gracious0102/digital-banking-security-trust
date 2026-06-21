"""Cybersecurity-specific analysis: threat modelling, risk scoring, attack surface.

Methodology
-----------
- Channel Risk Matrix: CVSS-inspired composite risk score per banking channel,
  weighted by adoption rate from the survey.
- Security Posture Distribution: distribution of composite posture scores by segment.
- Threat Actor Mapping: match threat actors to the most vulnerable user cohorts.
- Vulnerability Heatmap: cross-tabulate security gap severity across demographics.
- Attack Surface Score: aggregate exposure score for the respondent population.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import CHANNEL_RISK_WEIGHTS, SERVICE_COLUMNS, THREAT_ACTOR_PROFILES  # noqa: E402
from analysis.descriptive import security_posture_score  # noqa: E402


# ---------------------------------------------------------------------------
# Channel risk matrix
# ---------------------------------------------------------------------------

_SERVICE_TO_CHANNEL = {
    "uses_mobile_banking": "Mobile Banking",
    "uses_internet_banking": "Internet Banking",
    "uses_atm": "ATM Services",
    "uses_upi": "UPI / Payments",
    "uses_neft_rtgs": "NEFT/RTGS",
    "uses_aadhaar_pay": "Aadhaar Pay",
}


def channel_risk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Composite risk score per channel, weighted by survey adoption rate.

    Returns a DataFrame with columns: channel, adoption_pct, threat_scores...,
    composite_risk, weighted_risk (composite × adoption fraction).
    """
    total = len(df)
    rows = []
    for svc_col, channel in _SERVICE_TO_CHANNEL.items():
        if svc_col not in df.columns or channel not in CHANNEL_RISK_WEIGHTS:
            continue
        adoption = df[svc_col].sum() / total
        weights = CHANNEL_RISK_WEIGHTS[channel]
        composite = np.mean(list(weights.values()))
        row = {"channel": channel, "adoption_pct": round(adoption * 100, 1)}
        row.update({k: round(v * 100, 1) for k, v in weights.items()})
        row["composite_risk_score"] = round(composite * 100, 1)
        row["weighted_risk_score"] = round(composite * adoption * 100, 1)
        rows.append(row)
    df_risk = pd.DataFrame(rows)
    return df_risk.sort_values("weighted_risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Security posture by segment
# ---------------------------------------------------------------------------

def posture_by_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Mean security posture score and % high-posture (≥70) per segment value."""
    posture = security_posture_score(df)
    df = df.copy()
    df["_posture"] = posture
    df["_high_posture"] = (posture >= 70).astype(int)
    grouped = df.groupby(segment).agg(
        mean_posture=("_posture", "mean"),
        high_posture_pct=("_high_posture", lambda x: round(x.mean() * 100, 1)),
        n=("_posture", "count"),
    ).reset_index()
    grouped["mean_posture"] = grouped["mean_posture"].round(1)
    return grouped.sort_values("mean_posture", ascending=False)


# ---------------------------------------------------------------------------
# Vulnerability cohort identification
# ---------------------------------------------------------------------------

def vulnerable_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Identify highest-risk user cohorts by cross-cutting characteristics.

    Risk factors
    ------------
    - No MFA (high weight)
    - Has experienced fraud
    - Low phishing awareness (q_phishing_awareness ≥ 4)
    - Shared device
    - Low cybersecurity literacy
    """
    posture = security_posture_score(df)
    df = df.copy()
    df["_posture"] = posture

    # Vulnerability flag: posture < 40 AND (no MFA OR fraud victim)
    df["_vulnerable"] = (
        (posture < 40) &
        ((df["has_mfa_enabled"] == 0) | (df["has_experienced_fraud"] == 1))
    ).astype(int)

    rows = []
    for col in ["age_group", "income_bracket", "region", "education", "occupation"]:
        if col not in df.columns:
            continue
        grp = df.groupby(col).agg(
            vulnerable_pct=("_vulnerable", lambda x: round(x.mean() * 100, 1)),
            mean_posture=("_posture", lambda x: round(x.mean(), 1)),
            n=("_vulnerable", "count"),
        ).reset_index()
        grp["segment"] = col
        grp = grp.rename(columns={col: "segment_value"})
        rows.append(grp)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values("vulnerable_pct", ascending=False)


# ---------------------------------------------------------------------------
# Threat actor mapping
# ---------------------------------------------------------------------------

def threat_actor_table() -> pd.DataFrame:
    rows = []
    for actor, info in THREAT_ACTOR_PROFILES.items():
        rows.append({
            "threat_actor": actor,
            "primary_target": info["primary_target"],
            "tactics": " | ".join(info["tactics"]),
            "skill_level": info["skill_level"],
            "motivation": info["motivation"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attack surface score
# ---------------------------------------------------------------------------

def population_attack_surface(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate attack-surface score for the surveyed population.

    Computed as the sum of (channel adoption × channel composite risk)
    across all channels, normalised to 0–100.
    """
    risk_matrix = channel_risk_matrix(df)
    raw_score = risk_matrix["weighted_risk_score"].sum()
    max_possible = sum(
        np.mean(list(w.values())) * 100
        for w in CHANNEL_RISK_WEIGHTS.values()
    )
    normalised = min(raw_score / max_possible * 100, 100)

    mfa_pct = df["has_mfa_enabled"].mean() * 100
    fraud_pct = df["has_experienced_fraud"].mean() * 100
    shared_pct = df["shared_device"].mean() * 100

    return {
        "raw_exposure_score": round(raw_score, 2),
        "normalised_attack_surface": round(normalised, 1),
        "mfa_coverage_pct": round(mfa_pct, 1),
        "fraud_incidence_pct": round(fraud_pct, 1),
        "shared_device_pct": round(shared_pct, 1),
        "risk_level": "Critical" if normalised > 70 else "High" if normalised > 50 else "Medium",
    }


# ---------------------------------------------------------------------------
# Security gap priority matrix
# ---------------------------------------------------------------------------

def security_gap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Gap matrix: (severity × prevalence) for each identified security gap.

    Severity (1–5) is expert-assigned. Prevalence is derived from data.
    """
    gaps = [
        {
            "gap": "No MFA enabled",
            "severity": 5,
            "prevalence_pct": round((1 - df["has_mfa_enabled"].mean()) * 100, 1),
            "mitigation": "Mandate MFA for all high-value transactions",
        },
        {
            "gap": "Shared device usage",
            "severity": 4,
            "prevalence_pct": round(df["shared_device"].mean() * 100, 1),
            "mitigation": "Enforce session expiry + device-binding alerts",
        },
        {
            "gap": "No biometric authentication",
            "severity": 3,
            "prevalence_pct": round((1 - df["uses_biometric_auth"].mean()) * 100, 1),
            "mitigation": "Default-enable biometrics on supported devices",
        },
        {
            "gap": "Weak password hygiene",
            "severity": 4,
            "prevalence_pct": round((df["q_password_hygiene"] >= 4).mean() * 100, 1),
            "mitigation": "Enforce password complexity + breach-check on login",
        },
        {
            "gap": "Low phishing awareness",
            "severity": 4,
            "prevalence_pct": round((df["q_phishing_awareness"] >= 4).mean() * 100, 1),
            "mitigation": "Mandatory in-app phishing-simulation training",
        },
        {
            "gap": "App not regularly updated",
            "severity": 3,
            "prevalence_pct": round((1 - df["regularly_updates_app"].mean()) * 100, 1),
            "mitigation": "Enforce minimum app version before login",
        },
        {
            "gap": "No fraud reported after incident",
            "severity": 5,
            "prevalence_pct": round(
                df[df["has_experienced_fraud"] == 1]["reported_to_bank"].eq(0).mean() * 100, 1
            ) if df["has_experienced_fraud"].sum() > 0 else 0,
            "mitigation": "Streamline one-tap fraud reporting; proactive outreach to victims",
        },
    ]
    gap_df = pd.DataFrame(gaps)
    gap_df["priority_score"] = (gap_df["severity"] * gap_df["prevalence_pct"] / 100).round(2)
    return gap_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
