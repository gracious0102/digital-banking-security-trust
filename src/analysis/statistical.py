"""Inferential statistics: correlations, hypothesis tests, reliability.

Tests
-----
- Pearson / Spearman correlation matrices for Likert blocks
- Cronbach's alpha for scale reliability (SPT, literacy, convenience)
- Mann–Whitney U: fraud victims vs non-victims on trust/security scores
- Chi-square: fraud incidence vs demographic groups
- Kruskal–Wallis: SPT scores across income / region segments
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    CONVENIENCE_COLUMNS,
    LIKERT_COLUMNS,
    LIKERT_TITLES,
    LITERACY_COLUMNS,
    SECURITY_PERCEPTION_COLUMNS,
    SPT_COLUMNS,
    TRUST_COLUMNS,
)


# ---------------------------------------------------------------------------
# Cronbach's alpha
# ---------------------------------------------------------------------------

def cronbach_alpha(df: pd.DataFrame, columns: list[str]) -> float:
    """Compute Cronbach's alpha for a set of Likert items."""
    sub = df[columns].dropna()
    k = len(columns)
    if k < 2:
        return float("nan")
    item_vars = sub.var(axis=0, ddof=1)
    total_var = sub.sum(axis=1).var(ddof=1)
    return round(k / (k - 1) * (1 - item_vars.sum() / total_var), 4)


def scale_reliability(df: pd.DataFrame) -> pd.DataFrame:
    scales = {
        "Security Perceptions": SECURITY_PERCEPTION_COLUMNS,
        "Trust Dimensions": TRUST_COLUMNS,
        "Cybersecurity Literacy": LITERACY_COLUMNS,
        "Convenience / Satisfaction": CONVENIENCE_COLUMNS,
        "Full SPT Scale": SPT_COLUMNS,
        "All Likert Items": LIKERT_COLUMNS,
    }
    rows = []
    for scale_name, cols in scales.items():
        valid_cols = [c for c in cols if c in df.columns]
        alpha = cronbach_alpha(df, valid_cols)
        rows.append({
            "scale": scale_name,
            "items": len(valid_cols),
            "cronbach_alpha": alpha,
            "reliability": _alpha_label(alpha),
        })
    return pd.DataFrame(rows)


def _alpha_label(alpha: float) -> str:
    if np.isnan(alpha):
        return "N/A"
    if alpha >= 0.90:
        return "Excellent"
    if alpha >= 0.80:
        return "Good"
    if alpha >= 0.70:
        return "Acceptable"
    if alpha >= 0.60:
        return "Questionable"
    return "Poor"


# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------

def likert_correlation_matrix(df: pd.DataFrame,
                               columns: list[str] | None = None,
                               method: str = "spearman") -> pd.DataFrame:
    cols = [c for c in (columns or SPT_COLUMNS) if c in df.columns]
    corr = df[cols].corr(method=method)
    corr.index = [LIKERT_TITLES.get(c, c) for c in corr.index]
    corr.columns = [LIKERT_TITLES.get(c, c) for c in corr.columns]
    return corr.round(3)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

class TestResult(NamedTuple):
    test: str
    statistic: float
    p_value: float
    effect_size: float | None
    significant: bool
    interpretation: str


def mann_whitney_fraud_vs_trust(df: pd.DataFrame) -> list[TestResult]:
    """Mann–Whitney U comparing fraud victims vs non-victims on SPT scores."""
    fraud = df[df["has_experienced_fraud"] == 1]
    no_fraud = df[df["has_experienced_fraud"] == 0]
    results = []
    for col in SPT_COLUMNS:
        if col not in df.columns:
            continue
        u_stat, p_val = stats.mannwhitneyu(
            fraud[col].dropna(), no_fraud[col].dropna(), alternative="two-sided"
        )
        # rank-biserial correlation as effect size
        n1, n2 = len(fraud[col].dropna()), len(no_fraud[col].dropna())
        r = 1 - (2 * u_stat) / (n1 * n2)
        results.append(TestResult(
            test=f"Mann-Whitney U: {LIKERT_TITLES.get(col, col)}",
            statistic=round(u_stat, 2),
            p_value=round(p_val, 4),
            effect_size=round(abs(r), 3),
            significant=p_val < 0.05,
            interpretation=(
                f"Fraud victims score {'higher' if r < 0 else 'lower'} trust "
                f"(r={r:.3f}, {'significant' if p_val < 0.05 else 'not significant'})"
            ),
        ))
    return results


def chi_square_fraud_demographics(df: pd.DataFrame) -> list[TestResult]:
    """Chi-square tests: fraud incidence vs demographic groups."""
    results = []
    for col in ["gender", "age_group", "income_bracket", "region", "bank_type"]:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col], df["has_experienced_fraud"])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        n = len(df)
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        results.append(TestResult(
            test=f"Chi-square: Fraud vs {col}",
            statistic=round(chi2, 3),
            p_value=round(p_val, 4),
            effect_size=round(v, 3),
            significant=p_val < 0.05,
            interpretation=(
                f"Fraud incidence {'differs' if p_val < 0.05 else 'does not differ'} "
                f"significantly by {col} (Cramér V={v:.3f})"
            ),
        ))
    return results


def kruskal_spt_by_segment(df: pd.DataFrame,
                            segment_col: str = "income_bracket") -> list[TestResult]:
    """Kruskal–Wallis: SPT scores across a categorical segment."""
    results = []
    if segment_col not in df.columns:
        return results
    groups = [grp[1]["q_trust_platform"].dropna().values
              for grp in df.groupby(segment_col)]
    if len(groups) < 2:
        return results
    h_stat, p_val = stats.kruskal(*groups)
    n = len(df)
    eta2 = (h_stat - len(groups) + 1) / (n - len(groups))
    results.append(TestResult(
        test=f"Kruskal-Wallis: Trust by {segment_col}",
        statistic=round(h_stat, 3),
        p_value=round(p_val, 4),
        effect_size=round(max(eta2, 0), 3),
        significant=p_val < 0.05,
        interpretation=(
            f"Platform trust {'varies' if p_val < 0.05 else 'does not vary'} "
            f"significantly by {segment_col} (η²={eta2:.3f})"
        ),
    ))
    return results


def full_statistical_report(df: pd.DataFrame) -> dict:
    return {
        "scale_reliability": scale_reliability(df),
        "spearman_correlation": likert_correlation_matrix(df, SPT_COLUMNS, "spearman"),
        "fraud_vs_trust_tests": mann_whitney_fraud_vs_trust(df),
        "fraud_vs_demographics_tests": chi_square_fraud_demographics(df),
        "trust_by_income": kruskal_spt_by_segment(df, "income_bracket"),
        "trust_by_region": kruskal_spt_by_segment(df, "region"),
    }
