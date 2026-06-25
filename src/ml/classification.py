"""Trust prediction (regression) and fraud risk classification.

Models
------
Trust prediction (q_trust_platform, ordinal 1–5):
  - Baseline: Logistic Regression (ordinal proxy via classification)
  - Random Forest Classifier
  - XGBoost Classifier

Fraud risk (has_experienced_fraud, binary):
  - Logistic Regression baseline
  - Random Forest Classifier
  - XGBoost Classifier

All models use stratified 5-fold cross-validation and report:
  Accuracy, F1, AUC-ROC, Precision, Recall
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import FRAUD_TARGET, TRUST_TARGET  # noqa: E402
from ml.preprocessing import build_feature_matrix, scale_features  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _models(seed: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced", solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_binary(model, X_train, X_test, y_train, y_test,
                     model_name: str) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "fitted_model": model,
    }


def _cv_score(model, X, y) -> dict[str, float]:
    accs = cross_val_score(model, X, y, cv=_CV, scoring="accuracy", n_jobs=-1)
    f1s = cross_val_score(model, X, y, cv=_CV, scoring="f1_weighted", n_jobs=-1)
    return {
        "cv_accuracy_mean": round(accs.mean(), 4),
        "cv_accuracy_std": round(accs.std(), 4),
        "cv_f1_mean": round(f1s.mean(), 4),
        "cv_f1_std": round(f1s.std(), 4),
    }


# ---------------------------------------------------------------------------
# Fraud risk classification
# ---------------------------------------------------------------------------

def train_fraud_classifier(df: pd.DataFrame,
                             test_size: float = 0.2,
                             seed: int = 42) -> dict:
    from sklearn.model_selection import train_test_split

    X, y = build_feature_matrix(df, FRAUD_TARGET)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    results = []
    fitted = {}
    for name, model in _models(seed).items():
        r = _evaluate_binary(model, X_train_s, X_test_s, y_train, y_test, name)
        cv = _cv_score(model, X_train_s, y_train)
        r.update(cv)
        fitted[name] = r.pop("fitted_model")
        results.append(r)

    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != "fitted_model"}
                                for r in results])

    best_name = metrics_df.loc[metrics_df["auc_roc"].idxmax(), "model"]
    best_model = fitted[best_name]

    # Feature importances from best tree model
    feature_importance = None
    for nm in ["XGBoost", "Random Forest"]:
        if nm in fitted and hasattr(fitted[nm], "feature_importances_"):
            fi = pd.Series(fitted[nm].feature_importances_, index=X.columns)
            feature_importance = fi.sort_values(ascending=False).head(20)
            break

    return {
        "task": "Fraud Risk Classification",
        "metrics": metrics_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_importance": feature_importance,
        "feature_names": list(X.columns),
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_scaled": X_test_s,
        "fitted_models": fitted,
    }


# ---------------------------------------------------------------------------
# Trust prediction (treated as classification over 1–5 classes)
# ---------------------------------------------------------------------------

def train_trust_classifier(df: pd.DataFrame,
                             test_size: float = 0.2,
                             seed: int = 42) -> dict:
    from sklearn.model_selection import train_test_split

    X, y = build_feature_matrix(df, TRUST_TARGET)
    # Remap Likert 1–5 → 0–4 for XGBoost compatibility
    y_shifted = y - y.min()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shifted, test_size=test_size, random_state=seed, stratify=y_shifted
    )
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    results = []
    fitted = {}
    for name, model in _models(seed).items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        classes = sorted(y.unique())
        y_bin = label_binarize(y_test, classes=classes)
        y_prob = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted") if y_prob is not None else None
        cv = _cv_score(model, X_train_s, y_train)
        r = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "auc_roc": round(auc, 4) if auc else None,
        }
        r.update(cv)
        fitted[name] = model
        results.append(r)

    metrics_df = pd.DataFrame(results)
    best_name = metrics_df.loc[metrics_df["f1"].idxmax(), "model"]
    best_model = fitted[best_name]

    feature_importance = None
    for nm in ["XGBoost", "Random Forest"]:
        if nm in fitted and hasattr(fitted[nm], "feature_importances_"):
            fi = pd.Series(fitted[nm].feature_importances_, index=X.columns)
            feature_importance = fi.sort_values(ascending=False).head(20)
            break

    return {
        "task": "Trust Prediction",
        "metrics": metrics_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_importance": feature_importance,
        "feature_names": list(X.columns),
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_scaled": X_test_s,
        "fitted_models": fitted,
    }
