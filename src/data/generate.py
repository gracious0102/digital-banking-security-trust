"""Generate a 10 000-row synthetic dataset with realistic inter-feature correlations.

Latent factors
--------------
Four latent Gaussian factors drive correlated feature clusters:

  tech_savvy        — MFA adoption, biometrics, app updates, device type
  risk_awareness    — phishing literacy, password hygiene, regulatory awareness
  trust_disposition — Likert trust / security-perception scores
  fraud_exposure    — usage intensity × channel count → fraud probability

Correlations are embedded via a Cholesky-decomposed covariance matrix so the
final dataset looks realistic: e.g. fraud victims have lower trust scores, MFA
users report higher security adequacy, rural users favour ATM over UPI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import DATASET_SIZE, SYNTHETIC_CSV  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _choice(rng: np.random.Generator, options: list, probs: list[float], n: int) -> np.ndarray:
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    return rng.choice(options, size=n, p=probs)


def _likert(rng: np.random.Generator, latent: np.ndarray, noise: float = 0.6) -> np.ndarray:
    """Map a latent Gaussian value to a 1–5 Likert rating.

    latent ≈ N(0,1); negative latent → Strongly Agree (1), positive → Strongly Disagree (5).
    """
    noisy = latent + rng.normal(0, noise, len(latent))
    thresholds = [-1.2, -0.4, 0.4, 1.2]
    ratings = np.ones(len(noisy), dtype=int)
    for i, t in enumerate(thresholds, 1):
        ratings = np.where(noisy > t, i + 1, ratings)
    return np.clip(ratings, 1, 5)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def build_dataframe(seed: int = 42, n: int = DATASET_SIZE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # 1. Latent factors (4-vector per respondent, correlated)
    # -----------------------------------------------------------------------
    #   tech_savvy | risk_awareness | trust_disposition | fraud_exposure
    cov = np.array([
        # tech   risk   trust  fraud
        [1.00,  0.55,  0.40, -0.30],   # tech_savvy
        [0.55,  1.00,  0.35, -0.25],   # risk_awareness
        [0.40,  0.35,  1.00, -0.45],   # trust_disposition
        [-0.30, -0.25, -0.45,  1.00],  # fraud_exposure
    ])
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, 4)) @ L.T
    tech      = Z[:, 0]   # high → tech-savvy
    risk_aw   = Z[:, 1]   # high → risk-aware
    trust_fac = Z[:, 2]   # high → trusting
    fraud_exp = Z[:, 3]   # high → more fraud exposure

    # -----------------------------------------------------------------------
    # 2. Demographics
    # -----------------------------------------------------------------------
    gender = _choice(rng, ["Male", "Female", "Non-binary", "Prefer not to say"],
                     [0.47, 0.50, 0.02, 0.01], n)

    # Younger skew for tech-savvy respondents
    age_probs_base = np.array([0.28, 0.32, 0.22, 0.12, 0.06])
    # skew age toward younger for higher tech_savvy
    tech_norm = (tech - tech.min()) / (tech.max() - tech.min())
    age_group = []
    for t in tech_norm:
        probs = age_probs_base.copy()
        probs[0] += 0.15 * t
        probs[-1] -= 0.08 * t
        probs = np.clip(probs, 0.01, 1)
        probs /= probs.sum()
        age_group.append(rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], p=probs))
    age_group = np.array(age_group)

    education = _choice(rng,
        ["High School", "Graduate", "Postgraduate", "Doctoral"],
        [0.18, 0.40, 0.34, 0.08], n)

    occupation = _choice(rng,
        ["Student", "Salaried Employee", "Business Owner", "Self-Employed", "Retired", "Other"],
        [0.24, 0.38, 0.16, 0.12, 0.07, 0.03], n)

    income_bracket = _choice(rng,
        ["<25k", "25k-50k", "50k-100k", "100k-200k", ">200k"],
        [0.14, 0.28, 0.30, 0.20, 0.08], n)

    region = _choice(rng,
        ["Metro", "Tier-1", "Tier-2", "Tier-3", "Rural"],
        [0.30, 0.25, 0.22, 0.14, 0.09], n)

    bank_type = _choice(rng,
        ["Public Sector", "Private Sector", "Cooperative", "Payment Bank", "NBFC"],
        [0.38, 0.42, 0.09, 0.08, 0.03], n)

    # Years using digital banking (more for tech-savvy)
    yrs_base = np.clip(2 + 3 * tech_norm + rng.normal(0, 1, n), 0, 15)
    years_digital_banking = np.round(yrs_base).astype(int)

    # -----------------------------------------------------------------------
    # 3. Service adoption (correlated with tech_savvy + region)
    # -----------------------------------------------------------------------
    urban_mask = np.isin(region, ["Metro", "Tier-1"])

    def _service_prob(base: float, tech_coeff: float, urban_bonus: float) -> np.ndarray:
        p = base + tech_coeff * tech_norm + urban_bonus * urban_mask.astype(float)
        return np.clip(p, 0.02, 0.98)

    uses_mobile_banking    = rng.random(n) < _service_prob(0.72, 0.20, 0.08)
    uses_internet_banking  = rng.random(n) < _service_prob(0.55, 0.22, 0.10)
    uses_atm               = rng.random(n) < _service_prob(0.60, -0.05, -0.05)  # ATM less popular in metro
    uses_upi               = rng.random(n) < _service_prob(0.75, 0.15, 0.05)
    uses_neft_rtgs         = rng.random(n) < _service_prob(0.40, 0.18, 0.06)
    uses_aadhaar_pay       = rng.random(n) < _service_prob(0.30, 0.05, -0.08)  # more rural

    # -----------------------------------------------------------------------
    # 4. Usage patterns
    # -----------------------------------------------------------------------
    usage_frequency = _choice(rng,
        ["Daily", "Weekly", "Monthly", "Rarely"],
        [0.42, 0.33, 0.17, 0.08], n)

    transaction_frequency = _choice(rng,
        ["Multiple times daily", "Once daily", "Weekly", "Monthly"],
        [0.18, 0.30, 0.35, 0.17], n)

    avg_transaction_amount_tier = _choice(rng,
        ["<1k INR", "1k-10k INR", "10k-50k INR", ">50k INR"],
        [0.22, 0.38, 0.28, 0.12], n)

    # -----------------------------------------------------------------------
    # 5. Security setup (correlated with tech_savvy)
    # -----------------------------------------------------------------------
    risk_aw_norm = (risk_aw - risk_aw.min()) / (risk_aw.max() - risk_aw.min())
    has_mfa_enabled      = rng.random(n) < np.clip(0.45 + 0.35 * tech_norm, 0.05, 0.98)
    uses_biometric_auth  = rng.random(n) < np.clip(0.50 + 0.30 * tech_norm, 0.05, 0.98)
    uses_strong_password = rng.random(n) < np.clip(0.40 + 0.40 * (tech_norm + risk_aw_norm) / 2, 0.05, 0.98)
    regularly_updates_app = rng.random(n) < np.clip(0.45 + 0.30 * tech_norm, 0.05, 0.98)
    shared_device        = rng.random(n) < np.clip(0.25 - 0.15 * tech_norm, 0.02, 0.80)

    device_type = _choice(rng,
        ["Smartphone", "Tablet", "Laptop", "Desktop"],
        [0.68, 0.08, 0.16, 0.08], n)

    os_type = _choice(rng,
        ["Android", "iOS", "Windows", "macOS", "Other"],
        [0.52, 0.22, 0.18, 0.06, 0.02], n)

    # -----------------------------------------------------------------------
    # 6. Security incidents (driven by fraud_exposure + security setup)
    # -----------------------------------------------------------------------
    fraud_exp_norm = (fraud_exp - fraud_exp.min()) / (fraud_exp.max() - fraud_exp.min())
    security_protection = 0.4 * (has_mfa_enabled.astype(float) +
                                  uses_biometric_auth.astype(float) +
                                  uses_strong_password.astype(float)) / 3
    fraud_prob = np.clip(0.18 + 0.35 * fraud_exp_norm - 0.20 * security_protection, 0.02, 0.75)
    has_experienced_fraud = rng.random(n) < fraud_prob

    fraud_types = np.where(
        has_experienced_fraud,
        _choice(rng,
            ["Phishing", "Account Takeover", "Unauthorized Transaction",
             "SIM Swap", "Social Engineering"],
            [0.38, 0.20, 0.25, 0.10, 0.07], n),
        "None"
    )

    # Fraud victims are more likely to have reported it
    reported_to_bank = np.where(
        has_experienced_fraud,
        rng.random(n) < 0.72,
        False
    )

    fraud_resolved = np.where(
        reported_to_bank,
        rng.random(n) < 0.58,
        False
    )

    num_phishing_attempts_received = np.where(
        fraud_exp_norm > 0.5,
        rng.integers(2, 15, n),
        rng.integers(0, 5, n)
    )

    # -----------------------------------------------------------------------
    # 7. Cybersecurity literacy Likert items
    #    (1 = Strongly Agree = positive; high risk_awareness → lower score)
    # -----------------------------------------------------------------------
    q_phishing_awareness    = _likert(rng, -risk_aw + 0.2 * tech)
    q_password_hygiene      = _likert(rng, -(risk_aw_norm * 2 - 1) - tech_norm)
    q_data_sharing_awareness = _likert(rng, -risk_aw + 0.3)
    q_regulatory_awareness  = _likert(rng, -risk_aw + 0.5)

    # -----------------------------------------------------------------------
    # 8. Security & privacy perception Likert items
    #    (trust_disposition drives these; fraud victims are more negative)
    # -----------------------------------------------------------------------
    fraud_penalty = has_experienced_fraud.astype(float) * 0.8

    q_feel_secure           = _likert(rng, -trust_fac + fraud_penalty)
    q_security_features     = _likert(rng, -trust_fac + fraud_penalty + 0.1)
    q_privacy_policy_clarity = _likert(rng, -trust_fac + fraud_penalty + 0.3)
    q_mfa_effectiveness     = _likert(rng, -trust_fac + 0.1 - 0.5 * has_mfa_enabled.astype(float))
    q_data_breach_concern   = _likert(rng, trust_fac + fraud_penalty + 0.2)  # concern = DISAGREE trust
    q_incident_response_speed = _likert(rng, -trust_fac + fraud_penalty + 0.5)
    q_third_party_concern   = _likert(rng, trust_fac + 0.3)

    # -----------------------------------------------------------------------
    # 9. Trust Likert items
    # -----------------------------------------------------------------------
    q_trust_platform        = _likert(rng, -trust_fac + fraud_penalty)
    q_grievance_resolution  = _likert(rng, -trust_fac + fraud_penalty + 0.6)
    q_regulatory_trust      = _likert(rng, -trust_fac + 0.4)

    # -----------------------------------------------------------------------
    # 10. Convenience / satisfaction Likert items
    # -----------------------------------------------------------------------
    q24_7_availability  = _likert(rng, -trust_fac - tech + 0.2)
    q_saves_time        = _likert(rng, -trust_fac - tech + 0.1)
    q_easy_access       = _likert(rng, -trust_fac - tech + 0.3)
    q_reduced_branch_visits = _likert(rng, -trust_fac - tech + 0.1)
    q_speed_efficiency  = _likert(rng, -trust_fac - tech + 0.15)
    q_user_friendly     = _likert(rng, -trust_fac - tech + 0.25)
    q_overall_satisfaction = _likert(rng, -trust_fac - 0.5 * tech + fraud_penalty * 0.5)
    q_improved_experience  = _likert(rng, -trust_fac - 0.4 * tech + 0.1)

    # -----------------------------------------------------------------------
    # 11. Assemble DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        # Demographics
        "gender": gender,
        "age_group": age_group,
        "education": education,
        "occupation": occupation,
        "income_bracket": income_bracket,
        "region": region,
        "bank_type": bank_type,
        "years_digital_banking": years_digital_banking,
        # Services
        "uses_mobile_banking": uses_mobile_banking.astype(int),
        "uses_internet_banking": uses_internet_banking.astype(int),
        "uses_atm": uses_atm.astype(int),
        "uses_upi": uses_upi.astype(int),
        "uses_neft_rtgs": uses_neft_rtgs.astype(int),
        "uses_aadhaar_pay": uses_aadhaar_pay.astype(int),
        # Usage patterns
        "usage_frequency": usage_frequency,
        "transaction_frequency": transaction_frequency,
        "avg_transaction_amount_tier": avg_transaction_amount_tier,
        # Security setup
        "has_mfa_enabled": has_mfa_enabled.astype(int),
        "uses_biometric_auth": uses_biometric_auth.astype(int),
        "uses_strong_password": uses_strong_password.astype(int),
        "regularly_updates_app": regularly_updates_app.astype(int),
        "shared_device": shared_device.astype(int),
        "device_type": device_type,
        "os_type": os_type,
        # Incidents
        "has_experienced_fraud": has_experienced_fraud.astype(int),
        "fraud_type": fraud_types,
        "reported_to_bank": reported_to_bank.astype(int),
        "fraud_resolved": fraud_resolved.astype(int),
        "num_phishing_attempts_received": num_phishing_attempts_received,
        # Literacy Likert
        "q_phishing_awareness": q_phishing_awareness,
        "q_password_hygiene": q_password_hygiene,
        "q_data_sharing_awareness": q_data_sharing_awareness,
        "q_regulatory_awareness": q_regulatory_awareness,
        # Security perception Likert
        "q_feel_secure": q_feel_secure,
        "q_security_features": q_security_features,
        "q_privacy_policy_clarity": q_privacy_policy_clarity,
        "q_mfa_effectiveness": q_mfa_effectiveness,
        "q_data_breach_concern": q_data_breach_concern,
        "q_incident_response_speed": q_incident_response_speed,
        "q_third_party_concern": q_third_party_concern,
        # Trust Likert
        "q_trust_platform": q_trust_platform,
        "q_grievance_resolution": q_grievance_resolution,
        "q_regulatory_trust": q_regulatory_trust,
        # Convenience Likert
        "q24_7_availability": q24_7_availability,
        "q_saves_time": q_saves_time,
        "q_easy_access": q_easy_access,
        "q_reduced_branch_visits": q_reduced_branch_visits,
        "q_speed_efficiency": q_speed_efficiency,
        "q_user_friendly": q_user_friendly,
        "q_overall_satisfaction": q_overall_satisfaction,
        "q_improved_experience": q_improved_experience,
    })

    return df


def save_synthetic_csv(path: Path | None = None, seed: int = 42, n: int = DATASET_SIZE) -> Path:
    out = Path(path) if path else SYNTHETIC_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataframe(seed=seed, n=n)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = save_synthetic_csv()
    print(f"Wrote {out} ({DATASET_SIZE:,} rows)")
