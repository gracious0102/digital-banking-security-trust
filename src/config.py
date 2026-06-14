"""Project paths, full survey schema, and dimension groupings.

Extended for the deep-cybersecurity revamp: 10 000-respondent dataset,
ML-ready feature groups, and threat-modelling constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SYNTHETIC_CSV = DATA_DIR / "synthetic" / "survey_10k.csv"
RAW_CSV = DATA_DIR / "raw" / "survey_responses.csv"
# Legacy sample kept for backward compat
SAMPLE_CSV = DATA_DIR / "sample" / "survey_responses.csv"

FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
PROJECT_TITLE = "Digital Banking Cybersecurity: Risk, Trust & Threat Intelligence"
AUTHOR = "Grace Ibeji"
GITHUB = "https://github.com/gracious0102/digital-banking-security-trust"
DATASET_SIZE = 10_000

# ---------------------------------------------------------------------------
# Likert scale
# ---------------------------------------------------------------------------
LIKERT_LABELS = {
    1: "Strongly Agree",
    2: "Agree",
    3: "Neutral",
    4: "Disagree",
    5: "Strongly Disagree",
}

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

DEMOGRAPHIC_COLUMNS = [
    "gender",
    "age_group",
    "education",
    "occupation",
    "income_bracket",
    "region",
    "bank_type",
    "years_digital_banking",
]

SERVICE_COLUMNS = [
    "uses_mobile_banking",
    "uses_internet_banking",
    "uses_atm",
    "uses_upi",
    "uses_neft_rtgs",
    "uses_aadhaar_pay",
]

USAGE_COLUMNS = [
    "usage_frequency",
    "transaction_frequency",
    "avg_transaction_amount_tier",
]

SECURITY_SETUP_COLUMNS = [
    "has_mfa_enabled",
    "uses_biometric_auth",
    "uses_strong_password",
    "regularly_updates_app",
    "shared_device",
    "device_type",
    "os_type",
]

INCIDENT_COLUMNS = [
    "has_experienced_fraud",
    "fraud_type",
    "reported_to_bank",
    "fraud_resolved",
    "num_phishing_attempts_received",
]

# Likert: cybersecurity literacy (1 = Strongly Agree = positive)
LITERACY_COLUMNS = [
    "q_phishing_awareness",
    "q_password_hygiene",
    "q_data_sharing_awareness",
    "q_regulatory_awareness",
]

# Likert: security & privacy perceptions
SECURITY_PERCEPTION_COLUMNS = [
    "q_feel_secure",
    "q_security_features",
    "q_privacy_policy_clarity",
    "q_mfa_effectiveness",
    "q_data_breach_concern",
    "q_incident_response_speed",
    "q_third_party_concern",
]

# Likert: trust
TRUST_COLUMNS = [
    "q_trust_platform",
    "q_grievance_resolution",
    "q_regulatory_trust",
]

# Likert: convenience / satisfaction (kept from original)
CONVENIENCE_COLUMNS = [
    "q24_7_availability",
    "q_saves_time",
    "q_easy_access",
    "q_reduced_branch_visits",
    "q_speed_efficiency",
    "q_user_friendly",
    "q_overall_satisfaction",
    "q_improved_experience",
]

# Combined SPT (Security–Privacy–Trust) — core policy dimensions
SPT_COLUMNS = SECURITY_PERCEPTION_COLUMNS + TRUST_COLUMNS

# All Likert items
LIKERT_COLUMNS = LITERACY_COLUMNS + SPT_COLUMNS + CONVENIENCE_COLUMNS

# Human-readable titles for every Likert column
LIKERT_TITLES: dict[str, str] = {
    # Literacy
    "q_phishing_awareness": "Can identify phishing messages",
    "q_password_hygiene": "Uses unique strong passwords",
    "q_data_sharing_awareness": "Understands bank data collection",
    "q_regulatory_awareness": "Knows data-protection rights",
    # Security perceptions
    "q_feel_secure": "Overall feeling of security",
    "q_security_features": "Adequacy of security features",
    "q_privacy_policy_clarity": "Privacy policy is clear/transparent",
    "q_mfa_effectiveness": "MFA / two-step login is effective",
    "q_data_breach_concern": "Concern about data breaches",
    "q_incident_response_speed": "Bank responds quickly to incidents",
    "q_third_party_concern": "Concern over third-party data sharing",
    # Trust
    "q_trust_platform": "Trust in digital banking platform",
    "q_grievance_resolution": "Timely grievance resolution",
    "q_regulatory_trust": "Trust in regulators to protect data",
    # Convenience
    "q24_7_availability": "24/7 transaction availability",
    "q_saves_time": "Saves time vs branch visits",
    "q_easy_access": "Easy to access and navigate",
    "q_reduced_branch_visits": "Reduced need to visit branch",
    "q_speed_efficiency": "Speed and efficiency",
    "q_user_friendly": "User-friendly interface",
    "q_overall_satisfaction": "Overall satisfaction",
    "q_improved_experience": "Improved banking experience",
}

SPT_TITLES = {col: LIKERT_TITLES[col] for col in SPT_COLUMNS}

# ---------------------------------------------------------------------------
# ML feature groups
# ---------------------------------------------------------------------------

# Boolean features → numeric (0/1)
BINARY_FEATURES = [
    "has_mfa_enabled",
    "uses_biometric_auth",
    "uses_strong_password",
    "regularly_updates_app",
    "shared_device",
    "uses_mobile_banking",
    "uses_internet_banking",
    "uses_atm",
    "uses_upi",
    "uses_neft_rtgs",
    "uses_aadhaar_pay",
    "has_experienced_fraud",
    "reported_to_bank",
    "fraud_resolved",
]

# Ordinal / one-hot categorical features
CATEGORICAL_FEATURES = [
    "gender",
    "age_group",
    "education",
    "occupation",
    "income_bracket",
    "region",
    "bank_type",
    "usage_frequency",
    "transaction_frequency",
    "avg_transaction_amount_tier",
    "device_type",
    "os_type",
]

# Target variable for trust prediction (regression)
TRUST_TARGET = "q_trust_platform"

# Target variable for fraud risk (classification)
FRAUD_TARGET = "has_experienced_fraud"

# ---------------------------------------------------------------------------
# Cybersecurity risk weights (CVSS-inspired, 0–1)
# ---------------------------------------------------------------------------
CHANNEL_RISK_WEIGHTS: dict[str, dict[str, float]] = {
    "Mobile Banking": {
        "attack_surface": 0.85,
        "malware_risk": 0.75,
        "phishing_risk": 0.90,
        "account_takeover_risk": 0.80,
        "data_exfiltration_risk": 0.70,
    },
    "UPI / Payments": {
        "attack_surface": 0.90,
        "malware_risk": 0.65,
        "phishing_risk": 0.95,
        "account_takeover_risk": 0.70,
        "data_exfiltration_risk": 0.60,
    },
    "Internet Banking": {
        "attack_surface": 0.70,
        "malware_risk": 0.80,
        "phishing_risk": 0.85,
        "account_takeover_risk": 0.85,
        "data_exfiltration_risk": 0.80,
    },
    "ATM Services": {
        "attack_surface": 0.60,
        "malware_risk": 0.55,
        "phishing_risk": 0.50,
        "account_takeover_risk": 0.65,
        "data_exfiltration_risk": 0.45,
    },
    "NEFT/RTGS": {
        "attack_surface": 0.55,
        "malware_risk": 0.45,
        "phishing_risk": 0.60,
        "account_takeover_risk": 0.70,
        "data_exfiltration_risk": 0.65,
    },
    "Aadhaar Pay": {
        "attack_surface": 0.65,
        "malware_risk": 0.50,
        "phishing_risk": 0.70,
        "account_takeover_risk": 0.60,
        "data_exfiltration_risk": 0.75,
    },
}

THREAT_ACTOR_PROFILES: dict[str, dict] = {
    "Phishing Operator": {
        "primary_target": "UPI / Mobile Banking users",
        "tactics": ["Smishing", "Vishing", "Fake UPI apps", "OTP fraud"],
        "skill_level": "Low–Medium",
        "motivation": "Financial",
    },
    "Account Takeover Actor": {
        "primary_target": "Internet Banking / High-income users",
        "tactics": ["Credential stuffing", "SIM swap", "Session hijack"],
        "skill_level": "Medium–High",
        "motivation": "Financial",
    },
    "Insider Threat": {
        "primary_target": "Customer PII / Transaction records",
        "tactics": ["Data exfiltration", "Privilege abuse"],
        "skill_level": "Medium",
        "motivation": "Financial / Espionage",
    },
    "APT / Nation-State": {
        "primary_target": "Banking infrastructure / SWIFT",
        "tactics": ["Spear phishing", "Supply-chain attack", "Zero-day exploits"],
        "skill_level": "Very High",
        "motivation": "Geopolitical / Financial",
    },
    "Social Engineer": {
        "primary_target": "Low-literacy / Elderly users",
        "tactics": ["Impersonation", "Fake customer care", "Gift-card scams"],
        "skill_level": "Low",
        "motivation": "Financial",
    },
}
