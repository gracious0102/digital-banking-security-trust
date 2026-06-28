"""Cybersecurity-specific visualisations: risk matrix, threat map, gap matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import FIGURES_DIR  # noqa: E402
from analysis.cybersecurity import (  # noqa: E402
    channel_risk_matrix,
    population_attack_surface,
    security_gap_matrix,
    threat_actor_table,
    vulnerable_cohorts,
)
from analysis.descriptive import security_posture_score  # noqa: E402
from analysis.cybersecurity import posture_by_segment  # noqa: E402

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

_RED_PALETTE = ["#fef2f2", "#fecaca", "#f87171", "#ef4444", "#dc2626", "#b91c1c"]


def _save(fig: plt.Figure, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Channel risk matrix heatmap
# ---------------------------------------------------------------------------

def plot_channel_risk_matrix(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    risk_df = channel_risk_matrix(df)
    threat_cols = [c for c in risk_df.columns if c.endswith("_risk")]
    heat = risk_df.set_index("channel")[threat_cols].copy()
    heat.columns = [c.replace("_risk", "").replace("_", " ").title() for c in heat.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat.astype(float), annot=True, fmt=".0f", cmap="Reds",
                ax=ax, linewidths=0.4, vmin=40, vmax=100,
                cbar_kws={"label": "Risk score (0–100)", "shrink": 0.8})
    ax.set_title("Channel Threat Vector Heatmap (CVSS-inspired)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Threat Vector")
    ax.set_ylabel("Banking Channel")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, "channel_risk_heatmap.png", out_dir)


# ---------------------------------------------------------------------------
# Weighted channel risk bubble chart
# ---------------------------------------------------------------------------

def plot_channel_risk_bubble(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    risk_df = channel_risk_matrix(df)
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        risk_df["adoption_pct"],
        risk_df["composite_risk_score"],
        s=risk_df["weighted_risk_score"] * 40,
        c=risk_df["weighted_risk_score"],
        cmap="Reds",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in risk_df.iterrows():
        ax.annotate(row["channel"],
                    (row["adoption_pct"], row["composite_risk_score"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    plt.colorbar(scatter, ax=ax, label="Weighted Risk Score")
    ax.set_xlabel("Channel Adoption (%)", fontsize=11)
    ax.set_ylabel("Composite Risk Score", fontsize=11)
    ax.set_title("Attack Surface — Adoption vs Risk (bubble size = weighted risk)",
                 fontsize=13, fontweight="bold", pad=12)
    return _save(fig, "channel_risk_bubble.png", out_dir)


# ---------------------------------------------------------------------------
# Security gap priority matrix
# ---------------------------------------------------------------------------

def plot_security_gap_matrix(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    gap_df = security_gap_matrix(df)
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = ["#dc2626" if s >= 3.5 else "#d97706" if s >= 2.5 else "#16a34a"
              for s in gap_df["priority_score"]]
    bars = ax.barh(gap_df["gap"][::-1], gap_df["priority_score"][::-1],
                   color=colors[::-1], height=0.6)
    ax.set_title("Security Gap Priority Matrix\n(Severity × Prevalence)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Priority Score (higher = more urgent)")

    for bar, row in zip(bars, gap_df[::-1].itertuples()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"Sev:{row.severity} | Prev:{row.prevalence_pct}%",
                va="center", fontsize=8.5)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(color="#dc2626", label="Critical priority"),
        Patch(color="#d97706", label="High priority"),
        Patch(color="#16a34a", label="Medium priority"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.85)
    return _save(fig, "security_gap_matrix.png", out_dir)


# ---------------------------------------------------------------------------
# Vulnerable cohorts
# ---------------------------------------------------------------------------

def plot_vulnerable_cohorts(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    cohorts = vulnerable_cohorts(df)
    if cohorts.empty:
        return None

    top = cohorts.head(15)
    top["label"] = top["segment"] + ": " + top["segment_value"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#dc2626" if v > 40 else "#d97706" if v > 25 else "#16a34a"
              for v in top["vulnerable_pct"]]
    bars = ax.barh(top["label"][::-1], top["vulnerable_pct"][::-1], color=colors[::-1], height=0.6)
    ax.set_title("Vulnerable User Cohorts (Low Posture + High Risk)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("% Vulnerable in segment")
    for bar, pct in zip(bars, top["vulnerable_pct"][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=9)
    return _save(fig, "vulnerable_cohorts.png", out_dir)


# ---------------------------------------------------------------------------
# Population attack surface summary card (text/bar)
# ---------------------------------------------------------------------------

def plot_attack_surface_summary(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    surface = population_attack_surface(df)
    metrics = {
        "Normalised Attack\nSurface Score": surface["normalised_attack_surface"],
        "Fraud Incidence\n(%)": surface["fraud_incidence_pct"],
        "Shared Device\nUsage (%)": surface["shared_device_pct"],
        "MFA Coverage\n(%)": surface["mfa_coverage_pct"],
    }
    colors = ["#dc2626", "#d97706", "#fbbf24", "#16a34a"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=colors, width=0.55)
    ax.set_title(
        f"Population Attack Surface — Risk Level: {surface['risk_level']}",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_ylabel("Score / Percentage")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
    return _save(fig, "attack_surface_summary.png", out_dir)


# ---------------------------------------------------------------------------
# Spearman correlation heatmap (SPT items)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr_df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.3, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.7, "label": "Spearman ρ"},
                annot_kws={"size": 7})
    ax.set_title("Spearman Correlation Matrix — SPT Dimensions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    return _save(fig, "spt_correlation_heatmap.png", out_dir)
