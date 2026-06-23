"""Feature engineering and preprocessing for the ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FRAUD_TARGET,
    LIKERT_COLUMNS,
    SERVICE_COLUMNS,
    TRUST_TARGET,
)
from analysis.descriptive import security_posture_score  # noqa: E402


_ORDINAL_MAPS: dict[str, dict] = {
    "age_group": {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4},
    "education": {"High School": 0, "Graduate": 1, "Postgraduate": 2, "Doctoral": 3},
    "income_bracket": {"<25k": 0, "25k-50k": 1, "50k-100k": 2, "100k-200k": 3, ">200k": 4},
    "usage_frequency": {"Rarely": 0, "Monthly": 1, "Weekly": 2, "Daily": 3},
    "transaction_frequency": {"Monthly": 0, "Weekly": 1, "Once daily": 2, "Multiple times daily": 3},
    "avg_transaction_amount_tier": {"<1k INR": 0, "1k-10k INR": 1, "10k-50k INR": 2, ">50k INR": 3},
    "years_digital_banking": None,  # numeric, keep as-is
}

_NOMINAL_COLS = ["gender", "region", "bank_type", "device_type", "os_type", "occupation"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features useful for ML models."""
    df = df.copy()

    # Security posture composite (0–100)
    df["security_posture_score"] = security_posture_score(df)

    # Number of services adopted
    svc_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
    df["num_services_used"] = df[svc_cols].sum(axis=1)

    # Average literacy score (1–5, lower = better)
    literacy_cols = ["q_phishing_awareness", "q_password_hygiene",
                     "q_data_sharing_awareness", "q_regulatory_awareness"]
    lit_present = [c for c in literacy_cols if c in df.columns]
    if lit_present:
        df["avg_literacy_score"] = df[lit_present].mean(axis=1).round(3)

    # Average SPT score (1–5, lower = more positive)
    spt_present = [c for c in LIKERT_COLUMNS if c in df.columns]
    if spt_present:
        df["avg_spt_score"] = df[spt_present].mean(axis=1).round(3)

    # Security behaviour count (how many protective measures taken)
    behaviour_cols = ["has_mfa_enabled", "uses_biometric_auth",
                      "uses_strong_password", "regularly_updates_app"]
    beh_present = [c for c in behaviour_cols if c in df.columns]
    df["security_behaviour_count"] = df[beh_present].sum(axis=1)

    # Fraud risk indicator (binary, derived from high fraud_exposure proxy)
    df["high_usage_intensity"] = (
        df["usage_frequency"].isin(["Daily"]) &
        df["num_services_used"].ge(4)
    ).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame,
                          target: str,
                          drop_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with all numeric features ready for sklearn."""
    df = engineer_features(df)

    exclude = {
        "respondent_id", target, "fraud_type",
        *(drop_cols or []),
    }
    # Don't leak the target's close relatives
    if target == FRAUD_TARGET:
        exclude.update(["reported_to_bank", "fraud_resolved"])
    if target == TRUST_TARGET:
        exclude.update(["q_overall_satisfaction"])  # near-identical

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Ordinal encode known ordinals
    for col, mapping in _ORDINAL_MAPS.items():
        if col not in X.columns:
            continue
        if mapping is not None:
            X[col] = X[col].map(mapping).fillna(-1).astype(int)

    # One-hot encode nominal categoricals
    nominal = [c for c in _NOMINAL_COLS if c in X.columns]
    X = pd.get_dummies(X, columns=nominal, drop_first=False, dtype=int)

    # Drop any remaining object columns
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    X = X.drop(columns=obj_cols)

    # Fill NaN
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def cluster_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature set optimised for clustering (security / trust dimensions)."""
    df = engineer_features(df)
    cluster_cols = [
        "security_posture_score",
        "security_behaviour_count",
        "avg_literacy_score",
        "has_experienced_fraud",
        "num_services_used",
        "high_usage_intensity",
        "q_feel_secure",
        "q_trust_platform",
        "q_grievance_resolution",
        "q_phishing_awareness",
        "q_data_breach_concern",
    ]
    valid = [c for c in cluster_cols if c in df.columns]
    X = df[valid].fillna(df[valid].median(numeric_only=True))
    return X
