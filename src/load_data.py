"""Load and validate survey CSV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DEMOGRAPHIC_COLUMNS, LIKERT_COLUMNS, RAW_CSV, SAMPLE_CSV, SERVICE_COLUMNS


def load_survey(path: Path | None = None) -> pd.DataFrame:
    """Load survey CSV. Prefers explicit path, then raw, then sample."""
    if path is not None:
        csv_path = Path(path)
    elif RAW_CSV.exists():
        csv_path = RAW_CSV
    else:
        csv_path = SAMPLE_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No survey data found at {csv_path}. "
            "Run `python src/generate_sample_data.py` or place CSV in data/raw/."
        )

    df = pd.read_csv(csv_path)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    required = set(DEMOGRAPHIC_COLUMNS + SERVICE_COLUMNS + LIKERT_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    for col in SERVICE_COLUMNS:
        if not df[col].isin([True, False, 0, 1]).all():
            raise ValueError(f"Column {col} must be boolean (0/1 or True/False)")

    for col in LIKERT_COLUMNS:
        if not df[col].between(1, 5).all():
            raise ValueError(f"Column {col} must contain Likert ratings 1–5")
