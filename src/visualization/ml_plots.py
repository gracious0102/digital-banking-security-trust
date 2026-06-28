"""Visualisations for ML model outputs: clusters, ROC curves, SHAP, feature importance."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import FIGURES_DIR  # noqa: E402

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_CLUSTER_COLORS = ["#1d4ed8", "#d97706", "#dc2626", "#16a34a"]
_CLUSTER_MARKERS = ["o", "s", "^", "D"]


def _save(fig: plt.Figure, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def plot_cluster_scatter(df_clustered: pd.DataFrame,
                          explained_variance: list[float],
                          out_dir: Path = FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    clusters = sorted(df_clustered["cluster_id"].unique())

    for cid, color, marker in zip(clusters, _CLUSTER_COLORS, _CLUSTER_MARKERS):
        mask = df_clustered["cluster_id"] == cid
        label = df_clustered.loc[mask, "cluster_label"].iloc[0]
        ax.scatter(df_clustered.loc[mask, "pca_1"],
                   df_clustered.loc[mask, "pca_2"],
                   c=color, marker=marker, alpha=0.35, s=15, label=label)

    ax.set_title("User Risk Segments (K-Means, PCA projection)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}% var.)")
    ax.legend(markerscale=2, framealpha=0.85, fontsize=9)
    return _save(fig, "cluster_scatter.png", out_dir)


def plot_cluster_profiles(profiles: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    # Radar / heatmap of cluster means for key dimensions
    key_cols = [c for c in [
        "security_posture_score", "security_behaviour_count",
        "avg_literacy_score", "fraud_rate_pct",
        "q_feel_secure", "q_trust_platform",
        "q_data_breach_concern", "q_phishing_awareness",
    ] if c in profiles.columns]

    heat_data = profiles[key_cols].copy()
    heat_data.index = [f"Cluster {i}: {profiles.loc[i, 'cluster_label']}"
                       if "cluster_label" in profiles.columns else f"Cluster {i}"
                       for i in heat_data.index]
    heat_data.columns = [c.replace("_", " ").title() for c in heat_data.columns]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat_data.astype(float), annot=True, fmt=".1f",
                cmap="RdYlGn_r", ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.7})
    ax.set_title("Cluster Profiles — Key Security & Trust Dimensions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "cluster_profiles.png", out_dir)


def plot_silhouette_analysis(sil_df: pd.DataFrame, best_k: int,
                              out_dir: Path = FIGURES_DIR) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sil_df["k"], sil_df["silhouette_score"], "o-", color="#1d4ed8", linewidth=2)
    axes[0].axvline(best_k, color="#dc2626", linestyle="--", linewidth=1.2,
                    label=f"Best k={best_k}")
    axes[0].set_title("Silhouette Score vs k", fontweight="bold")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette score")
    axes[0].legend()

    axes[1].plot(sil_df["k"], sil_df["inertia"], "s-", color="#7c3aed", linewidth=2)
    axes[1].set_title("Elbow Curve (Inertia)", fontweight="bold")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Inertia (WCSS)")

    fig.suptitle("Optimal Cluster Selection", fontsize=13, fontweight="bold")
    return _save(fig, "cluster_selection.png", out_dir)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(importance: pd.Series,
                              title: str = "Feature Importance",
                              filename: str = "feature_importance.png",
                              out_dir: Path = FIGURES_DIR) -> Path:
    top = importance.head(20)
    colors = ["#dc2626" if i < 5 else "#1d4ed8" if i < 10 else "#6b7280"
              for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], height=0.65)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean importance score")
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + max(top.values) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    return _save(fig, filename, out_dir)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(fraud_result: dict, out_dir: Path = FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.50)")

    y_test = fraud_result["y_test"]
    metrics_df = fraud_result["metrics"]
    fitted_models = fraud_result["fitted_models"]
    X_test_s = fraud_result["X_test_scaled"]

    colors_line = ["#1d4ed8", "#dc2626", "#16a34a"]
    for (name, model), color in zip(fitted_models.items(), colors_line):
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_title("ROC Curves — Fraud Risk Classifier", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", framealpha=0.85)
    return _save(fig, "roc_fraud_classifier.png", out_dir)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(metrics_df: pd.DataFrame,
                            title: str = "Model Performance Comparison",
                            filename: str = "model_comparison.png",
                            out_dir: Path = FIGURES_DIR) -> Path:
    plot_cols = [c for c in ["accuracy", "f1", "precision", "recall", "auc_roc"]
                 if c in metrics_df.columns]
    melt = metrics_df.melt(id_vars="model", value_vars=plot_cols,
                            var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melt, x="metric", y="score", hue="model",
                palette=["#1d4ed8", "#dc2626", "#16a34a"], ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend(title="Model", framealpha=0.85)
    return _save(fig, filename, out_dir)


# ---------------------------------------------------------------------------
# SHAP visualisations
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_result: dict,
                       title: str = "SHAP Feature Importance",
                       filename: str = "shap_summary.png",
                       out_dir: Path = FIGURES_DIR) -> Path:
    try:
        import shap
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X_sample"],
        feature_names=shap_result["feature_names"],
        show=False,
        plot_type="bar",
        max_display=20,
        color="#1d4ed8",
    )
    plt.title(title, fontsize=13, fontweight="bold")
    return _save(plt.gcf(), filename, out_dir)


# ---------------------------------------------------------------------------
# Anomaly visualisation
# ---------------------------------------------------------------------------

def plot_anomaly_distribution(df_anomaly: pd.DataFrame,
                               out_dir: Path = FIGURES_DIR) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Score distribution
    sns.histplot(df_anomaly["anomaly_score"], bins=30, kde=True,
                 ax=axes[0], color="#1d4ed8", edgecolor="white")
    axes[0].axvline(40, color="#dc2626", linestyle="--", linewidth=1.2,
                    label="Anomaly threshold")
    axes[0].set_title("Anomaly Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Anomaly Score (0=critical, 100=normal)")
    axes[0].legend()

    # Tier breakdown
    if "anomaly_tier" in df_anomaly.columns:
        tier_counts = df_anomaly["anomaly_tier"].value_counts()
        tier_colors = {"Critical": "#dc2626", "High": "#d97706",
                       "Medium": "#fbbf24", "Normal": "#16a34a"}
        bars = axes[1].bar(tier_counts.index,
                           tier_counts.values,
                           color=[tier_colors.get(t, "#6b7280") for t in tier_counts.index])
        axes[1].set_title("Anomaly Tier Breakdown", fontweight="bold")
        axes[1].set_xlabel("Risk Tier")
        axes[1].set_ylabel("Respondents")
        total = len(df_anomaly)
        for bar, val in zip(bars, tier_counts.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.003,
                         f"{val/total*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Isolation Forest — Anomaly Detection Results",
                 fontsize=13, fontweight="bold")
    return _save(fig, "anomaly_distribution.png", out_dir)
