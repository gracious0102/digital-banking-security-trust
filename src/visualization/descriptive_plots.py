"""Publication-quality descriptive charts for demographics, services, and SPT."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    DEMOGRAPHIC_COLUMNS,
    FIGURES_DIR,
    LIKERT_COLUMNS,
    LIKERT_TITLES,
    SPT_COLUMNS,
)
from analysis.descriptive import (  # noqa: E402
    fraud_type_breakdown,
    likert_distribution,
    likert_summary,
    security_posture_score,
    service_adoption_table,
    spt_index,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 140,
    "figure.figsize": (11, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

PALETTE = {
    "primary": "#1d4ed8",
    "security": "#0891b2",
    "trust": "#7c3aed",
    "danger": "#dc2626",
    "warning": "#d97706",
    "success": "#16a34a",
    "neutral": "#6b7280",
}

_DEMO_TITLES = {
    "gender": "Gender Distribution",
    "age_group": "Age Group",
    "education": "Education Level",
    "occupation": "Occupation",
    "income_bracket": "Monthly Income Bracket (INR)",
    "region": "Region",
    "bank_type": "Primary Bank Type",
    "years_digital_banking": "Years Using Digital Banking",
}


def _save(fig: plt.Figure, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def plot_demographics(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    paths = []
    for col in DEMOGRAPHIC_COLUMNS:
        if col not in df.columns:
            continue
        if col == "years_digital_banking":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[col], bins=15, ax=ax, color=PALETTE["primary"], edgecolor="white")
            ax.set_title(_DEMO_TITLES[col], fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel("Years")
            ax.set_ylabel("Respondents")
            paths.append(_save(fig, f"demo_{col}.png", out_dir))
            continue

        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = sns.barplot(x=counts.index, y=counts.values, ax=ax,
                           palette="Blues_d", hue=counts.index, legend=False)
        ax.set_title(_DEMO_TITLES.get(col, col), fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Respondents")
        ax.tick_params(axis="x", rotation=20)
        total = len(df)
        for p in ax.patches:
            h = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, h + total * 0.005,
                    f"{h/total*100:.1f}%", ha="center", va="bottom", fontsize=8.5)
        paths.append(_save(fig, f"demo_{col}.png", out_dir))
    return paths


# ---------------------------------------------------------------------------
# Service adoption
# ---------------------------------------------------------------------------

def plot_service_adoption(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    table = service_adoption_table(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE["primary"] if p >= 60 else PALETTE["warning"]
               for p in table["adoption_pct"]]
    bars = ax.barh(table["channel"], table["adoption_pct"], color=colors, height=0.6)
    ax.set_xlabel("Adoption (%)", fontsize=11)
    ax.set_title("Digital Banking Channel Adoption", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    for bar, pct in zip(bars, table["adoption_pct"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=10, fontweight="bold")
    return _save(fig, "service_adoption.png", out_dir)


# ---------------------------------------------------------------------------
# SPT summary
# ---------------------------------------------------------------------------

def plot_spt_summary(summary: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    plot_df = summary.sort_values("positive_pct").copy()
    colors = [PALETTE["security"] if "secure" in q.lower() or "security" in q.lower() or "mfa" in q.lower() or "privacy" in q.lower()
              else PALETTE["trust"]
              for q in plot_df["question"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(plot_df["question"], plot_df["positive_pct"], color=colors, height=0.65)
    ax.set_title("Security, Privacy & Trust — Positive Response Rates",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Respondents with positive perception (%)", fontsize=11)
    ax.set_xlim(0, 110)
    ax.axvline(70, color=PALETTE["danger"], linewidth=1.2, linestyle="--", alpha=0.7,
               label="70% threshold")
    for bar, row in zip(bars, plot_df.itertuples()):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{row.positive_pct}%", va="center", fontsize=9)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(color=PALETTE["security"], label="Security / Privacy"),
        Patch(color=PALETTE["trust"], label="Trust"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.8)
    return _save(fig, "spt_summary.png", out_dir)


# ---------------------------------------------------------------------------
# SPT index distribution
# ---------------------------------------------------------------------------

def plot_spt_index_distribution(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    idx = spt_index(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(idx, bins=30, kde=True, ax=ax, color=PALETTE["primary"], edgecolor="white")
    ax.axvline(idx.mean(), color=PALETTE["danger"], linestyle="--", linewidth=1.5,
               label=f"Mean = {idx.mean():.2f}")
    ax.axvline(2.0, color=PALETTE["success"], linestyle=":", linewidth=1.2,
               label="Positive threshold (≤ 2.0)")
    ax.set_title("Security–Privacy–Trust (SPT) Index Distribution",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean SPT score (1 = most positive, 5 = most negative)", fontsize=11)
    ax.set_ylabel("Respondents")
    ax.legend()
    return _save(fig, "spt_index_distribution.png", out_dir)


# ---------------------------------------------------------------------------
# Security posture distribution
# ---------------------------------------------------------------------------

def plot_security_posture(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    posture = security_posture_score(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribution
    sns.histplot(posture, bins=20, kde=True, ax=axes[0],
                 color=PALETTE["security"], edgecolor="white")
    axes[0].axvline(70, color=PALETTE["success"], linestyle="--", linewidth=1.5,
                    label="High posture threshold (70)")
    axes[0].set_title("Security Posture Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Score (0–100, higher = safer)")
    axes[0].legend()

    # By region
    df2 = df.copy()
    df2["_posture"] = posture
    if "region" in df.columns:
        region_means = df2.groupby("region")["_posture"].mean().sort_values(ascending=False)
        bars = axes[1].bar(region_means.index, region_means.values,
                           color=PALETTE["primary"], alpha=0.85)
        axes[1].set_title("Mean Security Posture by Region", fontweight="bold")
        axes[1].set_xlabel("Region")
        axes[1].set_ylabel("Mean Posture Score")
        axes[1].tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, region_means.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}", ha="center", fontsize=9)

    fig.suptitle("Security Posture Index", fontsize=14, fontweight="bold", y=1.02)
    return _save(fig, "security_posture.png", out_dir)


# ---------------------------------------------------------------------------
# Fraud profile
# ---------------------------------------------------------------------------

def plot_fraud_profile(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    ft = fraud_type_breakdown(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Fraud type breakdown
    colors_ft = sns.color_palette("Reds_d", len(ft))
    axes[0].barh(ft["fraud_type"], ft["pct_of_fraud_victims"], color=colors_ft)
    axes[0].set_title("Fraud Types Among Victims", fontweight="bold")
    axes[0].set_xlabel("% of fraud victims")
    axes[0].set_xlim(0, 60)
    for i, (_, row) in enumerate(ft.iterrows()):
        axes[0].text(row["pct_of_fraud_victims"] + 0.5, i,
                     f"{row['pct_of_fraud_victims']}%", va="center", fontsize=9)

    # Fraud incidence by region
    if "region" in df.columns:
        fraud_region = df.groupby("region")["has_experienced_fraud"].mean() * 100
        fraud_region = fraud_region.sort_values(ascending=False)
        axes[1].bar(fraud_region.index, fraud_region.values,
                    color=[PALETTE["danger"] if v > 30 else PALETTE["warning"]
                           for v in fraud_region.values])
        axes[1].set_title("Fraud Incidence by Region", fontweight="bold")
        axes[1].set_ylabel("% experienced fraud")
        axes[1].tick_params(axis="x", rotation=20)
        for i, v in enumerate(fraud_region.values):
            axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Fraud Incident Analysis", fontsize=14, fontweight="bold", y=1.02)
    return _save(fig, "fraud_profile.png", out_dir)


# ---------------------------------------------------------------------------
# Likert diverging stacked bar
# ---------------------------------------------------------------------------

def plot_diverging_likert(df: pd.DataFrame,
                           columns: list[str] | None = None,
                           out_dir: Path = FIGURES_DIR,
                           filename: str = "likert_diverging.png") -> Path:
    cols = [c for c in (columns or SPT_COLUMNS) if c in df.columns]
    labels_order = ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
    color_map = {
        "Strongly Agree": "#16a34a",
        "Agree": "#86efac",
        "Neutral": "#d1d5db",
        "Disagree": "#fca5a5",
        "Strongly Disagree": "#dc2626",
    }

    rows = []
    for col in cols:
        dist = df[col].value_counts(normalize=True) * 100
        row = {"question": LIKERT_TITLES.get(col, col)}
        for rating, label in [(1, "Strongly Agree"), (2, "Agree"), (3, "Neutral"),
                               (4, "Disagree"), (5, "Strongly Disagree")]:
            row[label] = round(dist.get(rating, 0), 1)
        rows.append(row)
    plot_df = pd.DataFrame(rows).set_index("question")

    fig, ax = plt.subplots(figsize=(13, max(6, len(cols) * 0.7)))
    lefts_neg = np.zeros(len(plot_df))
    lefts_pos = np.zeros(len(plot_df))

    for label in ["Strongly Disagree", "Disagree"]:
        vals = plot_df[label].values
        ax.barh(plot_df.index, -vals, left=-lefts_neg, color=color_map[label],
                label=label, height=0.6)
        lefts_neg += vals

    ax.barh(plot_df.index, plot_df["Neutral"].values, left=0,
            color=color_map["Neutral"], label="Neutral", height=0.6)

    for label in ["Agree", "Strongly Agree"]:
        vals = plot_df[label].values
        ax.barh(plot_df.index, vals, left=lefts_pos, color=color_map[label],
                label=label, height=0.6)
        lefts_pos += vals

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("← Negative  |  Positive →", fontsize=10)
    ax.set_title("Likert Response Distribution — SPT Dimensions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.8, ncol=3, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(x):.0f}%"))
    return _save(fig, filename, out_dir)
