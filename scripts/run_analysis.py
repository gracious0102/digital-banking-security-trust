"""End-to-end pipeline: load data → analyse → visualise → write reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from analyze import (  # noqa: E402
    demographic_summary,
    key_insights,
    likert_summary,
    policy_recommendations,
    security_privacy_trust_summary,
    service_adoption_table,
    spt_index_stats,
    weakest_spt_dimension,
)
from config import AUTHOR, GITHUB, PROJECT_TITLE, REPORTS_DIR, SAMPLE_CSV, TABLES_DIR  # noqa: E402
from generate_sample_data import save_sample_csv  # noqa: E402
from load_data import load_survey  # noqa: E402
from visualize import (  # noqa: E402
    plot_all_likert_items,
    plot_demographics,
    plot_likert_summary,
    plot_security_privacy_trust_summary,
    plot_service_adoption,
    plot_satisfaction_by_usage,
    plot_spt_index_distribution,
    plot_spt_items,
)


def write_analysis_report(df, summary, spt_summary, insights, figure_count: int, data_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "analysis_summary.md"
    weakest = weakest_spt_dimension(df)
    spt_stats = spt_index_stats(df)

    lines = [
        f"# {PROJECT_TITLE}",
        "",
        f"**Author:** {AUTHOR}  ",
        f"**Repository:** {GITHUB}  ",
        f"**Sample size:** {len(df)} respondents  ",
        f"**Data source:** `{data_path.as_posix()}`",
        "",
        "## Research focus",
        "",
        "This analysis examines **perceived security, data privacy, and trust** in digital banking",
        "and frames findings for **consumer-protection and digital-governance** policy.",
        "",
        "## Key insights",
        "",
    ]
    lines.extend(f"- {item}" for item in insights)
    lines.extend([
        "",
        "## Security–Privacy–Trust (SPT) index",
        "",
        f"- Mean SPT score: **{spt_stats['mean']}** (1 = most positive, 5 = most negative)",
        f"- Respondents with positive SPT scores (≤ 2.0): **{spt_stats['pct_positive']}%**",
        f"- Weakest dimension: **{weakest['dimension']}** ({weakest['positive_pct']}% positive)",
        "",
        "## Service adoption",
        "",
    ])
    lines.append(service_adoption_table(df).to_markdown(index=False))
    lines.extend(["", "## SPT dimension summary", ""])
    lines.append(spt_summary.to_markdown(index=False))
    lines.extend(["", "## Full Likert summary", ""])
    lines.append(summary.to_markdown(index=False))
    lines.extend(["", "## Outputs", "", f"- {figure_count} charts saved to `outputs/figures/`", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_policy_report(df) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "policy_recommendations.md"
    recs = policy_recommendations(df)

    lines = [
        "# Consumer Protection & Digital Governance — Policy Brief",
        "",
        f"**Based on:** {PROJECT_TITLE}  ",
        f"**Author:** {AUTHOR}  ",
        f"**Sample:** {len(df)} respondents, Guwahati, India",
        "",
        "## Context",
        "",
        "High digital-banking adoption concentrates financial and personal data on mobile",
        "and payment platforms. Policy must balance innovation with **security, privacy,",
        "and consumer trust** — especially where grievance resolution and fraud prevention lag.",
        "",
        "## Recommendations",
        "",
    ]
    for i, rec in enumerate(recs, 1):
        lines.extend([
            f"### {i}. {rec['area']}",
            "",
            f"**Finding:** {rec['finding']}  ",
            f"**Recommendation:** {rec['recommendation']}",
            "",
        ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_tables(df, summary, spt_summary) -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in [
        ("service_adoption.csv", service_adoption_table(df)),
        ("likert_summary.csv", summary),
        ("spt_summary.csv", spt_summary),
    ]:
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital banking security & trust analysis")
    parser.add_argument("--data", type=Path, help="Path to survey CSV (defaults to raw or sample)")
    parser.add_argument("--regenerate-sample", action="store_true", help="Rebuild sample dataset")
    args = parser.parse_args()

    if args.regenerate_sample or not SAMPLE_CSV.exists():
        save_sample_csv()
        print(f"Sample data ready: {SAMPLE_CSV}")

    df = load_survey(args.data)
    summary = likert_summary(df)
    spt_summary = security_privacy_trust_summary(df)
    insights = key_insights(df)

    figure_paths = []
    figure_paths.extend(plot_demographics(df))
    figure_paths.append(plot_service_adoption(df))
    figure_paths.append(plot_likert_summary(summary))
    figure_paths.append(plot_security_privacy_trust_summary(spt_summary))
    figure_paths.append(plot_spt_index_distribution(df))
    figure_paths.extend(plot_spt_items(df))
    figure_paths.extend(plot_all_likert_items(df))
    figure_paths.append(plot_satisfaction_by_usage(df))

    table_paths = save_tables(df, summary, spt_summary)
    analysis_report = write_analysis_report(df, summary, spt_summary, insights, len(figure_paths), args.data or SAMPLE_CSV)
    policy_report = write_policy_report(df)

    print(f"Loaded {len(df)} responses")
    print("Key insights:")
    for item in insights:
        print(f"  • {item}")
    print(f"Figures: {len(figure_paths)} saved to outputs/figures/")
    print(f"Tables: {len(table_paths)} saved to outputs/tables/")
    print(f"Analysis report: {analysis_report}")
    print(f"Policy brief: {policy_report}")


if __name__ == "__main__":
    main()
