"""Anomaly detection for unusual security-behaviour patterns.

Uses Isolation Forest to surface respondents whose combination of
high-exposure behaviour and poor security posture marks them as outliers
(i.e., high-risk but undetected by conventional rules).

Output
------
- anomaly_score: continuous score (lower = more anomalous)
- is_anomaly: bool flag (True = flagged as outlier by Isolation Forest)
- anomaly_tier: Critical / High / Medium / Normal
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.preprocessing import cluster_feature_matrix  # noqa: E402


_CONTAMINATION = 0.08  # expect ~8% anomalous


def run_anomaly_detection(df: pd.DataFrame,
                           contamination: float = _CONTAMINATION,
                           seed: int = 42) -> dict:
    """Fit Isolation Forest and return enriched DataFrame + summary stats."""
    X = cluster_feature_matrix(df)
    feature_names = list(X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    raw_scores = iso.decision_function(X_scaled)   # higher = more normal
    predictions = iso.predict(X_scaled)             # -1 = anomaly, 1 = normal

    # Normalise score to 0–100 (0 = most anomalous, 100 = most normal)
    score_min, score_max = raw_scores.min(), raw_scores.max()
    normalised = (raw_scores - score_min) / (score_max - score_min + 1e-9) * 100

    # Attach engineered features to a copy for profiling
    df_out = df.copy()
    for col in feature_names:
        if col not in df_out.columns:
            df_out[col] = X[col].values

    df_out["anomaly_score_raw"] = raw_scores.round(4)
    df_out["anomaly_score"] = normalised.round(1)
    df_out["is_anomaly"] = predictions == -1

    df_out["anomaly_tier"] = pd.cut(
        df_out["anomaly_score"],
        bins=[-1, 20, 40, 60, 101],
        labels=["Critical", "High", "Medium", "Normal"],
        right=True,
    )

    summary = df_out["anomaly_tier"].value_counts().reset_index()
    summary.columns = ["tier", "count"]
    summary["pct"] = (summary["count"] / len(df_out) * 100).round(1)
    summary = summary.sort_values("tier")

    # Profile of anomalous respondents vs normal
    anomaly_profile = df_out.groupby("is_anomaly")[feature_names].mean().round(3)
    anomaly_profile.index = anomaly_profile.index.map({True: "Anomalous", False: "Normal"})
    df = df_out

    return {
        "df_anomaly": df_out,
        "model": iso,
        "scaler": scaler,
        "feature_names": feature_names,
        "anomaly_summary": summary,
        "anomaly_profile": anomaly_profile,
        "n_anomalies": int((predictions == -1).sum()),
        "anomaly_pct": round((predictions == -1).mean() * 100, 1),
    }
