"""SHAP-based model explainability for trust and fraud classifiers.

Produces:
- Global SHAP summary (beeswarm / bar) for the best model
- Local SHAP waterfall for the top anomalous / highest-risk respondents
- Feature importance DataFrame ranked by mean |SHAP value|
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def compute_shap_values(model, X_scaled: np.ndarray,
                         feature_names: list[str],
                         model_name: str = "model",
                         max_samples: int = 2000) -> dict:
    """Compute SHAP values for tree-based or linear models.

    Returns a dict with keys:
        shap_values, expected_value, X_sample, feature_names
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    # Subsample for speed
    idx = np.random.default_rng(42).choice(
        len(X_scaled), size=min(max_samples, len(X_scaled)), replace=False
    )
    X_sample = X_scaled[idx]

    if "XGBoost" in model_name or "Random Forest" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value
    else:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value

    # For multi-class, pick class index 1 (fraud=1) or most negative trust (class 0)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        exp_val = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
    else:
        shap_vals = shap_values
        exp_val = expected_value

    # Global importance: mean |SHAP|
    importance = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    return {
        "shap_values": shap_vals,
        "expected_value": exp_val,
        "X_sample": X_sample,
        "feature_names": feature_names,
        "global_importance": importance,
        "explainer": explainer,
    }


def shap_importance_table(shap_result: dict, top_n: int = 20) -> pd.DataFrame:
    imp = shap_result["global_importance"].head(top_n).reset_index()
    imp.columns = ["feature", "mean_abs_shap"]
    imp["rank"] = range(1, len(imp) + 1)
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(4)
    return imp[["rank", "feature", "mean_abs_shap"]]
