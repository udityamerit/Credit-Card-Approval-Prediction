"""
Global Risk Analytics (GRA) - Wholesale Credit Risk (WCR)
Model Risk Evaluation & Credit Decisioning Engine Web Application
Champion Algorithm: Tuned Soft-Voting Super-Ensemble (Random Forest + Extra Trees + LightGBM benchmark)
Standards: US Fed SR 11-7 / OCC 2011-12 / Basel III IRB / IFRS 9
Candidate: Uditya Narayan Tiwari
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model_risk_evaluation import CreditRiskEvaluator, get_benchmark_metrics

app = Flask(__name__)

# =============================================================================
# Load Model and Metadata
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Random_Forest_best_model.pkl")
CHAMPION_MODEL_PATH = os.path.join(BASE_DIR, "models", "Champion_Credit_Model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "best_threshold.txt")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "train_columns.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# If model artifact does not exist (e.g., fresh clone), train it automatically
if not os.path.exists(MODEL_PATH) and not os.path.exists(CHAMPION_MODEL_PATH):
    print("Notice: Model weights not found in models/ directory. Generating champion model...")
    try:
        from train_champion_model import train_and_export_champion_pipeline
        train_and_export_champion_pipeline()
    except Exception as e:
        print(f"Warning: Could not auto-generate models: {e}. Run 'python train_champion_model.py'.")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
elif os.path.exists(CHAMPION_MODEL_PATH):
    model = joblib.load(CHAMPION_MODEL_PATH)
else:
    model = None


try:
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())
except Exception:
    threshold = 0.39

train_columns = joblib.load(COLUMNS_PATH)

# Load metadata medians if present
edu_medians = {
    'Academic degree': 247500.0,
    'Higher education': 202500.0,
    'Incomplete higher': 180000.0,
    'Lower secondary': 135000.0,
    'Secondary / secondary special': 144000.0
}
occ_medians = {
    'Accountants': 4.5,
    'Cleaning staff': 3.2,
    'Cooking staff': 3.0,
    'Core staff': 4.8,
    'Drivers': 4.0,
    'HR staff': 4.2,
    'High skill tech staff': 4.5,
    'IT staff': 3.5,
    'Laborers': 3.8,
    'Low-skill Laborers': 3.0,
    'Managers': 5.2,
    'Medicine staff': 4.6,
    'Private service staff': 3.5,
    'Realty agents': 3.8,
    'Sales staff': 3.0,
    'Secretaries': 4.0,
    'Security staff': 4.0,
    'Waiters/barmen staff': 2.5,
    'Other': 3.0
}

if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            if "edu_medians" in meta:
                edu_medians = meta["edu_medians"]
            if "occ_medians" in meta:
                occ_medians = meta["occ_medians"]
    except Exception:
        pass

# Categorical Label Encoding Mappings (derived from training dataset)
ENCODING_MAPS = {
    'CODE_GENDER': {'F': 0, 'M': 1},
    'FLAG_OWN_CAR': {'N': 0, 'Y': 1},
    'FLAG_OWN_REALTY': {'N': 0, 'Y': 1},
    'NAME_INCOME_TYPE': {
        'Commercial associate': 0,
        'Pensioner': 1,
        'State servant': 2,
        'Student': 3,
        'Working': 4,
    },
    'NAME_EDUCATION_TYPE': {
        'Academic degree': 0,
        'Higher education': 1,
        'Incomplete higher': 2,
        'Lower secondary': 3,
        'Secondary / secondary special': 4,
    },
    'NAME_FAMILY_STATUS': {
        'Civil marriage': 0,
        'Married': 1,
        'Separated': 2,
        'Single / not married': 3,
        'Widow': 4,
    },
    'NAME_HOUSING_TYPE': {
        'Co-op apartment': 0,
        'House / apartment': 1,
        'Municipal apartment': 2,
        'Office apartment': 3,
        'Rented apartment': 4,
        'With parents': 5,
    },
    'OCCUPATION_TYPE': {
        'Accountants': 0,
        'Cleaning staff': 1,
        'Cooking staff': 2,
        'Core staff': 3,
        'Drivers': 4,
        'HR staff': 5,
        'High skill tech staff': 6,
        'IT staff': 7,
        'Laborers': 8,
        'Low-skill Laborers': 9,
        'Managers': 10,
        'Medicine staff': 11,
        'Private service staff': 12,
        'Realty agents': 13,
        'Sales staff': 14,
        'Secretaries': 15,
        'Security staff': 16,
        'Waiters/barmen staff': 17,
        'Other': 8,
    },
}


def age_to_days_birth(age_years: float) -> int:
    """Convert age in years to DAYS_BIRTH (negative days since birth)."""
    return -int(age_years * 365.25)


def employment_years_to_days_employed(employment_years: float) -> int:
    """Convert employment duration in years to DAYS_EMPLOYED (negative if employed)."""
    if employment_years <= 0:
        return 0
    return -int(employment_years * 365.25)


def compute_risk_attributions(user_dict: dict, proba: float) -> list:
    """
    Computes qualitative risk factor drivers (SHAP proxy / adverse action reasons)
    to comply with fair lending disclosure requirements.
    """
    drivers = []
    income = float(user_dict.get('AMT_INCOME_TOTAL', 0))
    employed_years = float(user_dict.get('EMPLOYED_YEARS', 0))
    realty = user_dict.get('FLAG_OWN_REALTY', 0)
    car = user_dict.get('FLAG_OWN_CAR', 0)
    income_to_edu = user_dict.get('INCOME_TO_EDU_MEDIAN', 1.0)
    tenure_to_occ = user_dict.get('TENURE_TO_OCC_MEDIAN', 1.0)

    if income >= 250000:
        drivers.append({"factor": "High Income Profile", "impact": "Positive", "detail": f"Annual income of ${income:,.0f} provides strong debt service capacity."})
    elif income < 100000:
        drivers.append({"factor": "Modest Income Threshold", "impact": "Adverse", "detail": "Annual income provides narrow debt service coverage."})

    if employed_years >= 4.0:
        drivers.append({"factor": "Stable Employment Tenor", "impact": "Positive", "detail": f"{employed_years:.1f} years continuous job stability demonstrates durable cashflow."})
    elif employed_years < 1.0:
        drivers.append({"factor": "Limited Job Tenure", "impact": "Adverse", "detail": "Less than 1 year at current employer increases earnings volatility."})

    if realty == 1:
        drivers.append({"factor": "Asset Ownership (Real Estate)", "impact": "Positive", "detail": "Unencumbered residential property ownership provides solid balance sheet backing."})

    if car == 1 and realty == 1:
        drivers.append({"factor": "Multi-Asset Ownership", "impact": "Positive", "detail": "Ownership of both real property and vehicle indicates prime collateral capacity."})

    if income_to_edu > 1.2:
        drivers.append({"factor": "Above-Peer Earnings Velocity", "impact": "Positive", "detail": f"Earning {income_to_edu:.2f}x the median income of educational cohort."})

    if tenure_to_occ > 1.2:
        drivers.append({"factor": "Superior Occupational Stability", "impact": "Positive", "detail": f"Job tenure is {tenure_to_occ:.2f}x higher than career peer median."})

    return drivers


# =============================================================================
# Web Application Routes
# =============================================================================
@app.route("/")
def index():
    """Enterprise Risk Analytics Executive Home & Portal Overview."""
    benchmarks = get_benchmark_metrics()
    return render_template(
        "landing_page.html",
        benchmarks=benchmarks,
        threshold=threshold,
        threshold_percent=round(threshold * 100, 1),
    )


@app.route("/predict", methods=["GET"])
def show_input_form():
    """Credit Card Application Form with Credit Risk Context."""
    return render_template(
        "form.html",
        threshold=threshold,
        threshold_percent=round(threshold * 100, 1),
    )


@app.route("/result", methods=["POST"])
def get_prediction():
    """Evaluates Credit Application, computes PD, Scorecard Score, and Risk Grade."""
    try:
        age_years = float(request.form.get("AGE_YEARS", 30))
        employed_years = float(request.form.get("EMPLOYED_YEARS", 3))
        income = float(request.form.get("AMT_INCOME_TOTAL", 150000.0))
        cnt_children = int(request.form.get("CNT_CHILDREN", 0))
        cnt_fam = float(request.form.get("CNT_FAM_MEMBERS", 2.0))

        days_birth = age_to_days_birth(age_years)
        days_employed = employment_years_to_days_employed(employed_years)

        raw_gender = request.form.get("CODE_GENDER", "M").strip().upper()
        raw_car = request.form.get("FLAG_OWN_CAR", "N").strip().upper()
        raw_realty = request.form.get("FLAG_OWN_REALTY", "Y").strip().upper()
        raw_income_type = request.form.get("NAME_INCOME_TYPE", "Working").strip()
        raw_education = request.form.get("NAME_EDUCATION_TYPE", "Secondary / secondary special").strip()
        raw_family = request.form.get("NAME_FAMILY_STATUS", "Married").strip()
        raw_housing = request.form.get("NAME_HOUSING_TYPE", "House / apartment").strip()
        raw_occupation = request.form.get("OCCUPATION_TYPE", "Laborers").strip()

        flag_work_phone = int(request.form.get("FLAG_WORK_PHONE", 0))
        flag_phone = int(request.form.get("FLAG_PHONE", 0))
        flag_email = int(request.form.get("FLAG_EMAIL", 0))

        # Advanced Domain & Peer Features
        income_per_fam = income / max(1.0, cnt_fam)
        emp_to_age = employed_years / max(18.0, age_years)
        inc_to_age = income / max(18.0, age_years)
        children_ratio = cnt_children / max(1.0, cnt_fam)
        total_contacts = flag_work_phone + flag_phone + flag_email
        has_car_realty = int(raw_car == 'Y' and raw_realty == 'Y')
        inc_to_edu = income / max(1.0, edu_medians.get(raw_education, 150000.0))
        tenure_to_occ = employed_years / max(0.5, occ_medians.get(raw_occupation, 3.0))

        # Map to training encoding values
        encoded_data = {
            'CODE_GENDER': ENCODING_MAPS['CODE_GENDER'].get(raw_gender, 0),
            'FLAG_OWN_CAR': ENCODING_MAPS['FLAG_OWN_CAR'].get(raw_car, 0),
            'FLAG_OWN_REALTY': ENCODING_MAPS['FLAG_OWN_REALTY'].get(raw_realty, 1),
            'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': income,
            'NAME_INCOME_TYPE': ENCODING_MAPS['NAME_INCOME_TYPE'].get(raw_income_type, 4),
            'NAME_EDUCATION_TYPE': ENCODING_MAPS['NAME_EDUCATION_TYPE'].get(raw_education, 4),
            'NAME_FAMILY_STATUS': ENCODING_MAPS['NAME_FAMILY_STATUS'].get(raw_family, 1),
            'NAME_HOUSING_TYPE': ENCODING_MAPS['NAME_HOUSING_TYPE'].get(raw_housing, 1),
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'FLAG_MOBIL': 1,
            'FLAG_WORK_PHONE': flag_work_phone,
            'FLAG_PHONE': flag_phone,
            'FLAG_EMAIL': flag_email,
            'OCCUPATION_TYPE': ENCODING_MAPS['OCCUPATION_TYPE'].get(raw_occupation, 8),
            'CNT_FAM_MEMBERS': cnt_fam,
            'AGE_YEARS': age_years,
            'EMPLOYED_YEARS': employed_years,
            'INCOME_PER_FAM_MEMBER': income_per_fam,
            'EMPLOYMENT_TO_AGE_RATIO': emp_to_age,
            'INCOME_TO_AGE_RATIO': inc_to_age,
            'CHILDREN_RATIO': children_ratio,
            'TOTAL_CONTACTS': total_contacts,
            'HAS_CAR_AND_REALTY': has_car_realty,
            'INCOME_TO_EDU_MEDIAN': inc_to_edu,
            'TENURE_TO_OCC_MEDIAN': tenure_to_occ,
        }

        user_df = pd.DataFrame([encoded_data])

        for col in train_columns:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[train_columns]

        # Model Inference: Probability of Default (PD)
        pd_proba = float(model.predict_proba(user_df)[:, 1][0])
        prediction = int(pd_proba >= threshold)

        # Convert PD to Credit Scorecard (300-850), Risk Tier, Regulatory Stage
        scorecard_info = CreditRiskEvaluator.pd_to_scorecard_points(pd_proba)
        risk_drivers = compute_risk_attributions(encoded_data, pd_proba)

        # Simulated Stress PD (under Moderate Adverse Scenario)
        stress_info = CreditRiskEvaluator.simulate_macroeconomic_stress(
            pd_proba, unemployment_shock_pct=2.5, gdp_contraction_pct=1.5
        )

        return render_template(
            "result.html",
            probability=round(pd_proba, 4),
            probability_percent=round(pd_proba * 100, 2),
            threshold=round(threshold, 4),
            threshold_percent=round(threshold * 100, 1),
            prediction=prediction,
            credit_score=scorecard_info["credit_score"],
            risk_tier=scorecard_info["risk_tier"],
            regulatory_stage=scorecard_info["regulatory_stage"],
            recommendation=scorecard_info["recommendation"],
            recommended_product=scorecard_info["recommended_product"],
            risk_drivers=risk_drivers,
            stressed_pd_percent=stress_info["stressed_pd_pct"],
            pd_multiplier=stress_info["pd_multiplier"],
            user_age=int(age_years),
            user_employment=employed_years,
            user_income=income,
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=f"Model Inference Error: {str(e)}",
            threshold=round(threshold, 4),
            threshold_percent=round(threshold * 100, 1),
        )


@app.route("/model-risk-evaluation")
def model_risk_evaluation():
    """
    Global Risk Analytics (GRA) Model Risk Evaluation & Validation Dashboard.
    Displays Gini, KS Statistic, Decile Rank-Ordering, PSI Drift,
    Macroeconomic Stress Testing, and SR 11-7 Model Governance Audit.
    """
    benchmarks = get_benchmark_metrics()
    return render_template("model_risk_evaluation.html", data=benchmarks)


@app.route("/audit-report")
def audit_report():
    """Formal Model Risk Evaluation & Validation Audit Certificate (SR 11-7 / Basel III)."""
    benchmarks = get_benchmark_metrics()
    return render_template("audit_report.html", data=benchmarks)


@app.route("/api/stress-test", methods=["POST"])
def api_stress_test():
    """
    Interactive API endpoint for real-time Macroeconomic Stress Simulation.
    Receives macro shock parameters and recalculates stressed portfolio PD and losses.
    """
    req = request.get_json() or {}
    base_pd = float(req.get("base_pd", 0.0214))
    unemployment_shock = float(req.get("unemployment_shock", 2.5))
    gdp_contraction = float(req.get("gdp_contraction", 1.5))
    rate_shock = float(req.get("rate_shock", 150))

    result = CreditRiskEvaluator.simulate_macroeconomic_stress(
        baseline_pd=base_pd,
        unemployment_shock_pct=unemployment_shock,
        gdp_contraction_pct=gdp_contraction,
        interest_rate_shock_bps=rate_shock,
    )
    return jsonify(result)


@app.route("/api/validation-summary", methods=["GET"])
def api_validation_summary():
    """Returns programmatic validation metrics for model governance monitoring."""
    return jsonify(get_benchmark_metrics())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)