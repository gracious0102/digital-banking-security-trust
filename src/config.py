"""Project paths, survey schema, and security/privacy/trust groupings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_CSV = DATA_DIR / "sample" / "survey_responses.csv"
RAW_CSV = DATA_DIR / "raw" / "survey_responses.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
REPORTS_DIR = ROOT / "outputs" / "reports"

LIKERT_LABELS = {
    1: "Strongly Agree",
    2: "Agree",
    3: "Neutral",
    4: "Disagree",
    5: "Strongly Disagree",
}

DEMOGRAPHIC_COLUMNS = ["gender", "age_group", "education", "occupation", "usage_frequency"]

SERVICE_COLUMNS = [
    "uses_mobile_banking",
    "uses_internet_banking",
    "uses_atm",
    "uses_upi",
]

LIKERT_COLUMNS = [
    "q24_7_availability",
    "q_saves_time",
    "q_easy_access",
    "q_reduced_branch_visits",
    "q_feel_secure",
    "q_security_features",
    "q_trust_platform",
    "q_speed_efficiency",
    "q_user_friendly",
    "q_grievance_resolution",
    "q_overall_satisfaction",
    "q_improved_experience",
]

LIKERT_TITLES = {
    "q24_7_availability": "24/7 transaction availability",
    "q_saves_time": "Saves time vs branch visits",
    "q_easy_access": "Easy to access and navigate",
    "q_reduced_branch_visits": "Reduced need to visit branch",
    "q_feel_secure": "Feeling of security",
    "q_security_features": "Adequacy of security features",
    "q_trust_platform": "Trust in digital platform",
    "q_speed_efficiency": "Speed and efficiency",
    "q_user_friendly": "User-friendly interface",
    "q_grievance_resolution": "Timely grievance resolution",
    "q_overall_satisfaction": "Overall satisfaction",
    "q_improved_experience": "Improved banking experience",
}

# Security–Privacy–Trust (SPT) dimensions for policy framing
SECURITY_COLUMNS = ["q_feel_secure", "q_security_features"]
TRUST_COLUMNS = ["q_trust_platform", "q_grievance_resolution"]
PRIVACY_RELATED_COLUMNS = ["q_feel_secure", "q_security_features", "q_trust_platform"]
SPT_COLUMNS = SECURITY_COLUMNS + TRUST_COLUMNS

SPT_TITLES = {col: LIKERT_TITLES[col] for col in SPT_COLUMNS}

PROJECT_TITLE = "Digital Banking: Security, Privacy & Customer Trust — Guwahati, India"
AUTHOR = "Grace Ibeji"
GITHUB = "https://github.com/gracious0102"
SAMPLE_SIZE = 199
