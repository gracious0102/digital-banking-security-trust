"""User segmentation via K-Means and hierarchical clustering.

Produces risk-labeled clusters:
  0 → Security-Conscious & Trusting
  1 → Digitally Active but Vulnerable
  2 → Skeptical / Low-Adopter
  3 → High-Risk / Fraud-Exposed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.preprocessing import cluster_feature_matrix  # noqa: E402

# Stable cluster labels ordered by mean security posture (descending)
CLUSTER_LABELS = {
    0: "Security-Conscious & Trusting",
    1: "Digitally Active but Vulnerable",
    2: "Skeptical / Low-Adopter",
    3: "High-Risk / Fraud-Exposed",
}

N_CLUSTERS = 4


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS,
               seed: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20, max_iter=500)
    km.fit(X_scaled)
    return km


def silhouette_analysis(X_scaled: np.ndarray,
                         k_range: range = range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=300)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels, sample_size=min(3000, len(X_scaled)))
        inertia = km.inertia_
        rows.append({"k": k, "silhouette_score": round(score, 4), "inertia": round(inertia, 1)})
    return pd.DataFrame(rows)


def run_clustering(df: pd.DataFrame) -> dict:
    """Full clustering pipeline. Returns enriched df + cluster profiles."""
    X = cluster_feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for 2D visualisation (store components)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Silhouette analysis (quick check)
    sil_df = silhouette_analysis(X_scaled, range(2, 8))
    best_k = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"])
    # Use configured N_CLUSTERS for stable labels
    km = fit_kmeans(X_scaled, N_CLUSTERS)

    df = df.copy()
    df["cluster_id"] = km.labels_
    df["cluster_label"] = df["cluster_id"].map(CLUSTER_LABELS)
    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    # Cluster profiles
    profile_cols = list(X.columns) + ["has_experienced_fraud"]
    profile_cols = list(dict.fromkeys(profile_cols))  # dedup
    valid = [c for c in profile_cols if c in df.columns]
    profiles = df.groupby("cluster_id")[valid].mean().round(3)
    profiles["cluster_label"] = profiles.index.map(CLUSTER_LABELS)
    profiles["n_members"] = df["cluster_id"].value_counts().sort_index()
    profiles["fraud_rate_pct"] = (profiles["has_experienced_fraud"] * 100).round(1)

    return {
        "df_clustered": df,
        "cluster_profiles": profiles,
        "silhouette_analysis": sil_df,
        "best_k_silhouette": best_k,
        "kmeans_model": km,
        "scaler": scaler,
        "pca_model": pca,
        "feature_names": list(X.columns),
        "explained_variance_pct": (pca.explained_variance_ratio_ * 100).round(1).tolist(),
    }
