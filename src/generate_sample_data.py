"""Generate a 199-row synthetic dataset aligned with dissertation frequency tables.

The dataset is synthetic — no real respondent PII is published.
Distributions are calibrated to match the aggregate results reported in the
original undergraduate dissertation (Guwahati, 199 respondents).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import SAMPLE_CSV, SAMPLE_SIZE


def _weighted_series(labels: list[str], counts: list[int], size: int, rng: np.random.Generator) -> pd.Series:
    probs = np.array(counts, dtype=float)
    probs /= probs.sum()
    return pd.Series(rng.choice(labels, size=size, p=probs))


def _weighted_likert(counts_by_rating: dict[int, int], size: int, rng: np.random.Generator) -> pd.Series:
    ratings = list(counts_by_rating.keys())
    weights = np.array(list(counts_by_rating.values()), dtype=float)
    weights /= weights.sum()
    return pd.Series(rng.choice(ratings, size=size, p=weights))


def build_sample_dataframe(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = SAMPLE_SIZE

    df = pd.DataFrame({"respondent_id": range(1, n + 1)})

    df["gender"] = _weighted_series(["Male", "Female", "Prefer not to say"], [94, 100, 5], n, rng)
    df["age_group"] = _weighted_series(
        ["Below 25", "25-35", "36-45", "46-55", "Above 55"], [125, 62, 10, 2, 0], n, rng
    )
    df["education"] = _weighted_series(
        ["Higher Secondary", "Graduate", "Postgraduate", "Others"], [74, 81, 32, 12], n, rng
    )
    df["occupation"] = _weighted_series(
        ["Student", "Salaried Employee", "Business Owner", "Self-Employed", "Others"],
        [122, 46, 23, 6, 2],
        n,
        rng,
    )
    df["usage_frequency"] = _weighted_series(["Daily", "Weekly", "Monthly", "Rarely"], [118, 60, 17, 4], n, rng)

    # Service adoption (thesis-aligned percentages)
    df["uses_mobile_banking"] = rng.random(n) < 0.824
    df["uses_internet_banking"] = rng.random(n) < 0.668
    df["uses_atm"] = rng.random(n) < 0.688
    df["uses_upi"] = rng.random(n) < 0.739

    likert_specs = {
        "q24_7_availability": {1: 165, 2: 14, 3: 6, 4: 6, 5: 8},
        "q_saves_time": {1: 174, 2: 8, 3: 5, 4: 4, 5: 8},
        "q_easy_access": {1: 158, 2: 22, 3: 8, 4: 5, 5: 6},
        "q_reduced_branch_visits": {1: 162, 2: 18, 3: 7, 4: 6, 5: 6},
        "q_feel_secure": {1: 156, 2: 20, 3: 10, 4: 7, 5: 6},
        "q_security_features": {1: 152, 2: 24, 3: 10, 4: 7, 5: 6},
        "q_trust_platform": {1: 158, 2: 20, 3: 9, 4: 6, 5: 6},
        "q_speed_efficiency": {1: 173, 2: 10, 3: 6, 4: 5, 5: 5},
        "q_user_friendly": {1: 160, 2: 22, 3: 3, 4: 6, 5: 8},
        "q_grievance_resolution": {1: 158, 2: 15, 3: 8, 4: 9, 5: 9},
        "q_overall_satisfaction": {1: 164, 2: 14, 3: 8, 4: 5, 5: 8},
        "q_improved_experience": {1: 164, 2: 14, 3: 8, 4: 5, 5: 8},
    }

    for column, spec in likert_specs.items():
        df[column] = _weighted_likert(spec, n, rng)

    return df


def save_sample_csv(path=SAMPLE_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_sample_dataframe()
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = save_sample_csv()
    print(f"Wrote {out} ({SAMPLE_SIZE} rows)")
