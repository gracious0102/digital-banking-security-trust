"""Load and validate the survey CSV for the cybersecurity pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    BINARY_FEATURES,
    LIKERT_COLUMNS,
    RAW_CSV,
    SERVICE_COLUMNS,
    SYNTHETIC_CSV,
)

_REQUIRED_COLUMNS = (
    ["respondent_id", "gender", "age_group", "education", "occupation",
     "income_bracket", "region", "bank_type", "years_digital_banking"]
    + SERVICE_COLUMNS
    + ["usage_frequency", "transaction_frequency", "avg_transaction_amount_tier",
       "has_mfa_enabled", "uses_biometric_auth", "uses_strong_password",
       "regularly_updates_app", "shared_device", "device_type", "os_type",
       "has_experienced_fraud", "fraud_type", "reported_to_bank",
       "fraud_resolved", "num_phishing_attempts_received"]
    + LIKERT_COLUMNS
)


def load_survey(path: Path | None = None) -> pd.DataFrame:
    """Load survey CSV — prefers explicit path, then raw, then synthetic."""
    if path is not None:
        csv_path = Path(path)
    elif RAW_CSV.exists():
        csv_path = RAW_CSV
    else:
        csv_path = SYNTHETIC_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No data found at {csv_path}.\n"
            "Run:  python src/data/generate.py\n"
            "or place your CSV at data/raw/survey_responses.csv"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    _validate(df)
    df = _coerce_types(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    for col in LIKERT_COLUMNS:
        bad = ~df[col].between(1, 5)
        if bad.any():
            raise ValueError(
                f"Column '{col}' contains values outside 1–5 "
                f"(first bad index: {df.index[bad][0]})"
            )

    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
    for col in binary_cols:
        if not df[col].isin([0, 1, True, False]).all():
            raise ValueError(f"Column '{col}' must be binary (0/1)")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
    df[binary_cols] = df[binary_cols].astype(int)
    for col in LIKERT_COLUMNS:
        df[col] = df[col].astype(int)
    return df
