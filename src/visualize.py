"""Charts for digital banking security, privacy, and trust analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analyze import likert_distribution, service_adoption_table, spt_index
from config import (
    DEMOGRAPHIC_COLUMNS,
    FIGURES_DIR,
    LIKERT_COLUMNS,
    LIKERT_TITLES,
    SPT_COLUMNS,
)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 120, "font.size": 10})

ACCENT = "#0a4d78"
SECURITY_COLOR = "#1d4ed8"
TRUST_COLOR = "#7c3aed"


def _save(fig, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_demographics(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    paths = []
    titles = {
        "gender": "Gender Distribution",
        "age_group": "Age Group",
        "education": "Educational Qualification",
        "occupation": "Occupational Profile",
        "usage_frequency": "Frequency of Digital Banking Usage",
    }
    for col in DEMOGRAPHIC_COLUMNS:
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False)
        ax.set_title(titles[col])
        ax.set_xlabel("")
        ax.set_ylabel("Respondents")
        ax.tick_params(axis="x", rotation=20)
        paths.append(_save(fig, f"demo_{col}.png", out_dir))
    return paths


def plot_service_adoption(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    table = service_adoption_table(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=table, x="service", y="percentage", ax=ax, hue="service", legend=False, color=ACCENT)
    ax.set_title("Digital Banking Channel Adoption")
    ax.set_xlabel("")
    ax.set_ylabel("Adoption (%)")
    ax.tick_params(axis="x", rotation=15)
    for i, pct in enumerate(table["percentage"]):
        ax.text(i, pct + 1, f"{pct}%", ha="center", fontsize=9)
    return _save(fig, "service_adoption.png", out_dir)


def plot_likert_summary(summary: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    plot_df = summary.sort_values("positive_pct")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=plot_df, y="question", x="positive_pct", ax=ax, color=ACCENT)
    ax.set_title("Positive Perception by Dimension (Agree + Strongly Agree)")
    ax.set_xlabel("Positive responses (%)")
    ax.set_ylabel("")
    return _save(fig, "likert_positive_summary.png", out_dir)


def plot_security_privacy_trust_summary(summary: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Bar chart of SPT dimensions only — core policy-relevant chart."""
    plot_df = summary.sort_values("positive_pct")
    colors = [SECURITY_COLOR if "security" in q.lower() or "secure" in q.lower() else TRUST_COLOR
              for q in plot_df["question"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(plot_df["question"], plot_df["positive_pct"], color=colors)
    ax.set_title("Security, Privacy & Trust — Positive Response Rates")
    ax.set_xlabel("Positive responses (%)")
    ax.set_xlim(0, 100)
    for bar, pct in zip(bars, plot_df["positive_pct"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{pct}%", va="center", fontsize=9)
    return _save(fig, "spt_summary.png", out_dir)


def plot_spt_index_distribution(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    index = spt_index(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(index, bins=20, kde=True, ax=ax, color=ACCENT)
    ax.axvline(index.mean(), color="crimson", linestyle="--", label=f"Mean = {index.mean():.2f}")
    ax.set_title("Security–Privacy–Trust (SPT) Index Distribution")
    ax.set_xlabel("Mean SPT score per respondent (lower = more positive)")
    ax.set_ylabel("Count")
    ax.legend()
    return _save(fig, "spt_index_distribution.png", out_dir)


def plot_likert_item(df: pd.DataFrame, column: str, out_dir: Path = FIGURES_DIR) -> Path:
    dist = likert_distribution(df, column)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dist, x="rating", y="percentage", ax=ax, hue="rating", legend=False, palette="Blues_r")
    ax.set_title(LIKERT_TITLES[column])
    ax.set_xlabel("")
    ax.set_ylabel("Responses (%)")
    for i, pct in enumerate(dist["percentage"]):
        ax.text(i, pct + 1, f"{pct}%", ha="center", fontsize=9)
    return _save(fig, f"likert_{column}.png", out_dir)


def plot_all_likert_items(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    return [plot_likert_item(df, col, out_dir) for col in LIKERT_COLUMNS]


def plot_spt_items(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    return [plot_likert_item(df, col, out_dir) for col in SPT_COLUMNS]


def plot_satisfaction_by_usage(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    plot_df = df.groupby("usage_frequency")["q_overall_satisfaction"].mean().reset_index()
    order = ["Daily", "Weekly", "Monthly", "Rarely"]
    plot_df["usage_frequency"] = pd.Categorical(plot_df["usage_frequency"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("usage_frequency")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="usage_frequency", y="q_overall_satisfaction", ax=ax, palette="viridis")
    ax.set_title("Average Satisfaction Score by Usage Frequency")
    ax.set_xlabel("Usage frequency")
    ax.set_ylabel("Mean score (1=Strongly Agree … 5=Strongly Disagree)")
    ax.invert_yaxis()
    return _save(fig, "satisfaction_by_usage.png", out_dir)
