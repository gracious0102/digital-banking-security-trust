"""Master pipeline: generate data → analyse → ML → cybersecurity → reports.

Usage
-----
  python scripts/run_full_pipeline.py                    # full run
  python scripts/run_full_pipeline.py --skip-ml         # descriptive + cyber only
  python scripts/run_full_pipeline.py --data path.csv   # your own CSV
  python scripts/run_full_pipeline.py --n 5000          # custom dataset size
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    _RICH = True
except ImportError:
    _RICH = False

from config import (
    AUTHOR, FIGURES_DIR, GITHUB, MODELS_DIR, PROJECT_TITLE,
    REPORTS_DIR, TABLES_DIR,
)
from data.generate import save_synthetic_csv
from data.load import load_survey
from analysis.descriptive import (
    demographic_summary, fraud_profile, key_insights,
    likert_summary, policy_recommendations, service_adoption_table,
    spt_index_stats, spt_summary, weakest_spt_dimension,
)
from analysis.statistical import full_statistical_report
from analysis.cybersecurity import (
    channel_risk_matrix, population_attack_surface,
    security_gap_matrix, threat_actor_table, vulnerable_cohorts,
)
from visualization.descriptive_plots import (
    plot_demographics, plot_diverging_likert, plot_fraud_profile,
    plot_security_posture, plot_service_adoption, plot_spt_index_distribution,
    plot_spt_summary,
)
from visualization.cybersecurity_plots import (
    plot_attack_surface_summary, plot_channel_risk_bubble,
    plot_channel_risk_matrix, plot_correlation_heatmap,
    plot_security_gap_matrix, plot_vulnerable_cohorts,
)

con = Console() if _RICH else None


def _log(msg: str, style: str = "") -> None:
    if _RICH and con:
        con.print(f"[{style}]{msg}[/{style}]" if style else msg)
    else:
        print(msg)


def _banner() -> None:
    if _RICH and con:
        con.print(Panel.fit(
            f"[bold cyan]{PROJECT_TITLE}[/bold cyan]\n"
            f"[dim]{AUTHOR}  ·  {GITHUB}[/dim]",
            border_style="blue",
        ))
    else:
        print("=" * 70)
        print(PROJECT_TITLE)
        print("=" * 70)


# ---------------------------------------------------------------------------
# Phase 1: Data
# ---------------------------------------------------------------------------

def phase_data(args) -> "pd.DataFrame":
    import pandas as pd
    from config import DATASET_SIZE, SYNTHETIC_CSV

    _log("\n[Phase 1] Data Generation & Loading", "bold yellow")

    n = args.n if hasattr(args, "n") and args.n else DATASET_SIZE
    if args.data:
        _log(f"  Using provided CSV: {args.data}")
    else:
        _log(f"  Generating {n:,}-row synthetic dataset …")
        save_synthetic_csv(n=n)
        _log(f"  Saved → {SYNTHETIC_CSV}", "green")

    df = load_survey(args.data if args.data else None)
    _log(f"  Loaded {len(df):,} respondents  ·  {len(df.columns)} features", "green")
    return df


# ---------------------------------------------------------------------------
# Phase 2: Descriptive analysis
# ---------------------------------------------------------------------------

def phase_descriptive(df) -> dict:
    _log("\n[Phase 2] Descriptive & Statistical Analysis", "bold yellow")
    t = time.time()

    summary = likert_summary(df)
    spt_sum = spt_summary(df)
    spt_stats = spt_index_stats(df)
    weakest = weakest_spt_dimension(df)
    services = service_adoption_table(df)
    insights = key_insights(df)
    fraud = fraud_profile(df)
    recs = policy_recommendations(df)
    stat_report = full_statistical_report(df)

    # Save tables
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    services.to_csv(TABLES_DIR / "service_adoption.csv", index=False)
    summary.to_csv(TABLES_DIR / "likert_summary.csv", index=False)
    spt_sum.to_csv(TABLES_DIR / "spt_summary.csv", index=False)
    fraud.to_csv(TABLES_DIR / "fraud_profile.csv", index=False)
    stat_report["scale_reliability"].to_csv(TABLES_DIR / "scale_reliability.csv", index=False)
    stat_report["spearman_correlation"].to_csv(TABLES_DIR / "spt_correlation.csv")

    # Plots
    _log("  Generating descriptive charts …")
    plot_demographics(df)
    plot_service_adoption(df)
    plot_spt_summary(spt_sum)
    plot_spt_index_distribution(df)
    plot_diverging_likert(df)
    plot_security_posture(df)
    plot_fraud_profile(df)
    plot_correlation_heatmap(stat_report["spearman_correlation"])

    _log(f"  Done in {time.time()-t:.1f}s  ·  {len(insights)} insights extracted", "green")
    return {
        "summary": summary, "spt_sum": spt_sum, "spt_stats": spt_stats,
        "weakest": weakest, "services": services, "insights": insights,
        "fraud": fraud, "recs": recs, "stat_report": stat_report,
    }


# ---------------------------------------------------------------------------
# Phase 3: Cybersecurity analysis
# ---------------------------------------------------------------------------

def phase_cybersecurity(df) -> dict:
    _log("\n[Phase 3] Cybersecurity Threat & Risk Analysis", "bold yellow")
    t = time.time()

    risk_matrix = channel_risk_matrix(df)
    attack_surface = population_attack_surface(df)
    gap_matrix = security_gap_matrix(df)
    cohorts = vulnerable_cohorts(df)
    threats = threat_actor_table()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    risk_matrix.to_csv(TABLES_DIR / "channel_risk_matrix.csv", index=False)
    gap_matrix.to_csv(TABLES_DIR / "security_gap_matrix.csv", index=False)
    threats.to_csv(TABLES_DIR / "threat_actors.csv", index=False)
    cohorts.to_csv(TABLES_DIR / "vulnerable_cohorts.csv", index=False)

    _log("  Generating cybersecurity charts …")
    plot_channel_risk_matrix(df)
    plot_channel_risk_bubble(df)
    plot_security_gap_matrix(df)
    plot_vulnerable_cohorts(df)
    plot_attack_surface_summary(df)

    _log(f"  Risk level: [bold red]{attack_surface['risk_level']}[/bold red]"
         if _RICH else f"  Risk level: {attack_surface['risk_level']}")
    _log(f"  Attack surface score: {attack_surface['normalised_attack_surface']}/100")
    _log(f"  Done in {time.time()-t:.1f}s", "green")
    return {"risk_matrix": risk_matrix, "attack_surface": attack_surface,
            "gap_matrix": gap_matrix, "cohorts": cohorts}


# ---------------------------------------------------------------------------
# Phase 4: ML
# ---------------------------------------------------------------------------

def phase_ml(df) -> dict:
    import joblib
    from ml.clustering import run_clustering
    from ml.classification import train_fraud_classifier, train_trust_classifier
    from ml.anomaly import run_anomaly_detection
    from ml.explainability import compute_shap_values, shap_importance_table
    from visualization.ml_plots import (
        plot_cluster_scatter, plot_cluster_profiles, plot_silhouette_analysis,
        plot_feature_importance, plot_roc_curves, plot_model_comparison,
        plot_shap_summary, plot_anomaly_distribution,
    )

    _log("\n[Phase 4] Machine Learning Pipeline", "bold yellow")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 4a. Clustering
    _log("  [4a] K-Means clustering …")
    t = time.time()
    clust = run_clustering(df)
    clust["cluster_profiles"].to_csv(TABLES_DIR / "cluster_profiles.csv")
    clust["silhouette_analysis"].to_csv(TABLES_DIR / "silhouette_analysis.csv", index=False)
    plot_cluster_scatter(clust["df_clustered"], clust["explained_variance_pct"])
    plot_cluster_profiles(clust["cluster_profiles"])
    plot_silhouette_analysis(clust["silhouette_analysis"], clust["best_k_silhouette"])
    joblib.dump(clust["kmeans_model"], MODELS_DIR / "kmeans.joblib")
    _log(f"     Best k={clust['best_k_silhouette']}  ·  {time.time()-t:.1f}s", "green")

    # 4b. Fraud risk classifier
    _log("  [4b] Fraud risk classification (LR / RF / XGBoost) …")
    t = time.time()
    fraud_res = train_fraud_classifier(df)
    fraud_res["metrics"].to_csv(TABLES_DIR / "fraud_classifier_metrics.csv", index=False)
    plot_model_comparison(fraud_res["metrics"], "Fraud Classifier — Model Comparison",
                          "fraud_model_comparison.png")
    plot_roc_curves(fraud_res)
    if fraud_res["feature_importance"] is not None:
        plot_feature_importance(fraud_res["feature_importance"],
                                "Fraud Classifier — Feature Importance",
                                "fraud_feature_importance.png")
    joblib.dump(fraud_res["best_model"], MODELS_DIR / "fraud_classifier.joblib")
    best = fraud_res["best_model_name"]
    best_row = fraud_res["metrics"].loc[fraud_res["metrics"]["model"] == best].iloc[0]
    _log(f"     Best model: {best}  AUC={best_row['auc_roc']:.3f}  F1={best_row['f1']:.3f}  ·  {time.time()-t:.1f}s", "green")

    # 4c. Trust prediction
    _log("  [4c] Trust prediction (LR / RF / XGBoost) …")
    t = time.time()
    trust_res = train_trust_classifier(df)
    trust_res["metrics"].to_csv(TABLES_DIR / "trust_classifier_metrics.csv", index=False)
    plot_model_comparison(trust_res["metrics"], "Trust Classifier — Model Comparison",
                          "trust_model_comparison.png")
    if trust_res["feature_importance"] is not None:
        plot_feature_importance(trust_res["feature_importance"],
                                "Trust Predictor — Feature Importance",
                                "trust_feature_importance.png")
    joblib.dump(trust_res["best_model"], MODELS_DIR / "trust_classifier.joblib")
    best_t = trust_res["best_model_name"]
    best_row_t = trust_res["metrics"].loc[trust_res["metrics"]["model"] == best_t].iloc[0]
    _log(f"     Best model: {best_t}  F1={best_row_t['f1']:.3f}  ·  {time.time()-t:.1f}s", "green")

    # 4d. SHAP explainability (best fraud model)
    _log("  [4d] SHAP explainability …")
    t = time.time()
    try:
        shap_res = compute_shap_values(
            fraud_res["best_model"],
            fraud_res["X_test_scaled"],
            fraud_res["feature_names"],
            fraud_res["best_model_name"],
        )
        shap_table = shap_importance_table(shap_res)
        shap_table.to_csv(TABLES_DIR / "shap_importance.csv", index=False)
        plot_shap_summary(shap_res, "SHAP — Fraud Risk Driver Importance")
        _log(f"     Top SHAP driver: {shap_table.iloc[0]['feature']}  ·  {time.time()-t:.1f}s", "green")
    except Exception as e:
        _log(f"     SHAP skipped: {e}", "dim")
        shap_res = None

    # 4e. Anomaly detection
    _log("  [4e] Isolation Forest anomaly detection …")
    t = time.time()
    anom = run_anomaly_detection(df)
    anom["anomaly_summary"].to_csv(TABLES_DIR / "anomaly_summary.csv", index=False)
    plot_anomaly_distribution(anom["df_anomaly"])
    _log(f"     Flagged {anom['n_anomalies']} anomalies ({anom['anomaly_pct']}%)  ·  {time.time()-t:.1f}s", "green")

    return {
        "clustering": clust,
        "fraud": fraud_res,
        "trust": trust_res,
        "shap": shap_res,
        "anomaly": anom,
    }


# ---------------------------------------------------------------------------
# Phase 5: Reports
# ---------------------------------------------------------------------------

def write_reports(df, descriptive: dict, cybersecurity: dict,
                  ml: dict | None = None) -> list[Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # --- Analysis summary ---
    report_path = REPORTS_DIR / "analysis_summary.md"
    spt_stats = descriptive["spt_stats"]
    weakest = descriptive["weakest"]
    surface = cybersecurity["attack_surface"]

    lines = [
        f"# {PROJECT_TITLE}",
        "",
        f"**Author:** {AUTHOR}  ",
        f"**Repository:** {GITHUB}  ",
        f"**Sample:** {len(df):,} respondents  ",
        f"**Dataset:** Synthetic, correlated (10k-scale)",
        "",
        "## Executive Summary",
        "",
        f"Population attack surface risk level: **{surface['risk_level']}** "
        f"(score {surface['normalised_attack_surface']}/100).  ",
        f"Fraud incidence: **{surface['fraud_incidence_pct']}%** of respondents.  ",
        f"MFA coverage: **{surface['mfa_coverage_pct']}%** — the most critical gap.",
        "",
        "## Key Insights",
        "",
    ]
    lines.extend(f"- {item}" for item in descriptive["insights"])

    lines.extend([
        "",
        "## Security–Privacy–Trust (SPT) Index",
        "",
        f"- Mean SPT score: **{spt_stats['mean']}** (1 = most positive, 5 = most negative)",
        f"- Positive responses (≤ 2.0): **{spt_stats['pct_positive']}%**",
        f"- Negative responses (≥ 4.0): **{spt_stats['pct_negative']}%**",
        f"- Weakest dimension: **{weakest['dimension']}** "
        f"({weakest['positive_pct']}% positive, mean {weakest['mean_score']})",
        "",
        "## Service Adoption",
        "",
    ])
    lines.append(descriptive["services"].to_markdown(index=False))

    lines.extend(["", "## SPT Dimension Summary", ""])
    lines.append(descriptive["spt_sum"].to_markdown(index=False))

    lines.extend(["", "## Channel Risk Matrix", ""])
    lines.append(cybersecurity["risk_matrix"][
        ["channel", "adoption_pct", "composite_risk_score", "weighted_risk_score"]
    ].to_markdown(index=False))

    lines.extend(["", "## Security Gap Priority", ""])
    lines.append(cybersecurity["gap_matrix"][
        ["gap", "severity", "prevalence_pct", "priority_score", "mitigation"]
    ].to_markdown(index=False))

    if ml:
        fraud_m = ml["fraud"]["metrics"]
        trust_m = ml["trust"]["metrics"]
        lines.extend([
            "",
            "## ML Results",
            "",
            "### Fraud Risk Classification",
            "",
        ])
        lines.append(fraud_m[["model", "accuracy", "f1", "auc_roc",
                               "cv_accuracy_mean"]].to_markdown(index=False))
        lines.extend(["", "### Trust Prediction", ""])
        lines.append(trust_m[["model", "accuracy", "f1",
                               "cv_accuracy_mean"]].to_markdown(index=False))

        anom = ml["anomaly"]
        lines.extend([
            "",
            "### Anomaly Detection",
            "",
            f"- Anomalies flagged: **{anom['n_anomalies']}** ({anom['anomaly_pct']}%)",
            "",
        ])
        lines.append(anom["anomaly_summary"].to_markdown(index=False))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    paths.append(report_path)

    # --- Policy brief ---
    policy_path = REPORTS_DIR / "policy_recommendations.md"
    recs = descriptive["recs"]
    p_lines = [
        "# Consumer Protection & Digital Governance — Policy Brief",
        "",
        f"**Project:** {PROJECT_TITLE}  ",
        f"**Author:** {AUTHOR}  ",
        f"**Dataset:** {len(df):,} respondents (synthetic, thesis-aligned)",
        "",
        "## Context",
        "",
        "This brief translates data-driven cybersecurity findings into actionable",
        "consumer-protection and digital-governance recommendations.",
        f"Population attack surface risk level: **{surface['risk_level']}**.",
        "",
        "## Recommendations",
        "",
    ]
    for i, rec in enumerate(recs, 1):
        p_lines.extend([
            f"### {i}. {rec['area']}",
            f"**Severity:** {rec['severity']}  ",
            f"**Finding:** {rec['finding']}  ",
            f"**Recommendation:** {rec['recommendation']}",
            "",
        ])
    policy_path.write_text("\n".join(p_lines), encoding="utf-8")
    paths.append(policy_path)

    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Banking Cybersecurity Pipeline")
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to your own survey CSV (skips generation)")
    parser.add_argument("--n", type=int, default=10_000,
                        help="Synthetic dataset size (default 10 000)")
    parser.add_argument("--skip-ml", action="store_true",
                        help="Skip ML phase (faster run)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_total = time.time()
    _banner()

    df = phase_data(args)
    descriptive = phase_descriptive(df)
    cyber = phase_cybersecurity(df)
    ml_results = None if args.skip_ml else phase_ml(df)

    _log("\n[Phase 5] Writing reports …", "bold yellow")
    report_paths = write_reports(df, descriptive, cyber, ml_results)

    elapsed = time.time() - t_total
    _log(f"\n{'='*60}", "bold")
    _log(f"  Pipeline complete in {elapsed:.1f}s", "bold green")
    _log(f"  Figures  → {FIGURES_DIR}")
    _log(f"  Tables   → {TABLES_DIR}")
    _log(f"  Reports  → {REPORTS_DIR}")
    if ml_results:
        _log(f"  Models   → {MODELS_DIR}")

    _log("\nKey findings:", "bold")
    for item in descriptive["insights"]:
        _log(f"  • {item}")
    _log(f"{'='*60}", "bold")


if __name__ == "__main__":
    main()
