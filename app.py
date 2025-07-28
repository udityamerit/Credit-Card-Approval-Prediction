from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and metadata
model = joblib.load("models/Random_Forest_best_model.pkl")
with open("models/best_threshold.txt", "r") as f:
    threshold = float(f.read())
train_columns = joblib.load("models/train_columns.pkl")

# Helper functions for conversions
def age_to_days_birth(age_years):
    """Convert age in years to DAYS_BIRTH (negative days since birth)"""
    return -int(age_years * 365.25)

def employment_years_to_days_employed(employment_years):
    """Convert employment duration in years to DAYS_EMPLOYED (negative if employed)"""
    if employment_years == 0:
        return 0  # Unemployed
    return -int(employment_years * 365.25)

# Credit card recommendations based on risk level
def get_credit_recommendations(prediction, probability):
    """Get credit card recommendations based on risk assessment"""
    
    if prediction == 0:  # Good Credit Risk
        if probability < 0.1:  # Excellent credit
            return {
                "category": "Premium Credit Cards",
                "cards": [
                    {"name": "Platinum Rewards Card", "features": "High credit limit, premium rewards, travel benefits"},
                    {"name": "Cashback Elite Card", "features": "5% cashback on all purchases, no annual fee"},
                    {"name": "Travel Master Card", "features": "Air miles, lounge access, travel insurance"}
                ]
            }
        elif probability < 0.3:  # Good credit
            return {
                "category": "Standard Credit Cards",
                "cards": [
                    {"name": "Rewards Plus Card", "features": "2% cashback, moderate credit limit"},
                    {"name": "Shopping Card", "features": "Special discounts on retail, 1.5% rewards"},
                    {"name": "Fuel Saver Card", "features": "Extra rewards on fuel purchases"}
                ]
            }
        else:  # Fair credit
            return {
                "category": "Basic Credit Cards",
                "cards": [
                    {"name": "Starter Credit Card", "features": "Low credit limit, basic rewards"},
                    {"name": "Building Credit Card", "features": "Credit building features, financial education"},
                    {"name": "Secured Credit Card", "features": "Requires security deposit, helps build credit"}
                ]
            }
    else:  # Bad Credit Risk
        return {
            "category": "Alternative Financial Products",
            "cards": [
                {"name": "Prepaid Card", "features": "No credit check required, spending control"},
                {"name": "Secured Credit Card", "features": "Security deposit required, credit building opportunity"},
                {"name": "Debit Card Plus", "features": "Enhanced debit features, cashback on purchases"}
            ]
        }

@app.route("/")
def form_page():
    return render_template("landing_page.html")

@app.route("/predict", methods=["GET"])
def show_input_form():
    # Pass threshold to the form template
    return render_template("form.html", threshold=threshold, threshold_percent=round(threshold * 100, 2))

@app.route("/result", methods=["POST"])
def get_prediction():
    try:
        age_years = int(request.form.get("AGE_YEARS", 0))
        employed_years = float(request.form.get("EMPLOYED_YEARS", 0))

        # Convert user-friendly inputs to model format
        days_birth = age_to_days_birth(age_years)
        days_employed = employment_years_to_days_employed(employed_years)

        user_data = {
            'CODE_GENDER': request.form.get("CODE_GENDER", "").strip().upper(),
            'FLAG_OWN_CAR': request.form.get("FLAG_OWN_CAR", "").strip().upper(),
            'FLAG_OWN_REALTY': request.form.get("FLAG_OWN_REALTY", "").strip().upper(),
            'CNT_CHILDREN': int(request.form.get("CNT_CHILDREN", 0)),
            'AMT_INCOME_TOTAL': float(request.form.get("AMT_INCOME_TOTAL", 0)),
            'NAME_INCOME_TYPE': request.form.get("NAME_INCOME_TYPE", "").strip(),
            'NAME_EDUCATION_TYPE': request.form.get("NAME_EDUCATION_TYPE", "").strip(),
            'NAME_FAMILY_STATUS': request.form.get("NAME_FAMILY_STATUS", "").strip(),
            'NAME_HOUSING_TYPE': request.form.get("NAME_HOUSING_TYPE", "").strip(),
            'BIRTH(Years)': days_birth,  # Properly converted
            'EMPLOYED(Years)': days_employed,  # Properly converted
            'FLAG_WORK_PHONE': int(request.form.get("FLAG_WORK_PHONE", 0)),
            'FLAG_PHONE': int(request.form.get("FLAG_PHONE", 0)),
            'FLAG_EMAIL': int(request.form.get("FLAG_EMAIL", 0)),
            'CNT_FAM_MEMBERS': float(request.form.get("CNT_FAM_MEMBERS", 0)),
        }

        # Preprocess for prediction
        user_df = pd.DataFrame([user_data])
        user_encoded = pd.get_dummies(user_df)

        for col in train_columns:
            if col not in user_encoded.columns:
                user_encoded[col] = 0

        user_encoded = user_encoded[train_columns]

        # Predict
        proba = model.predict_proba(user_encoded)[:, 1][0]
        prediction = int(proba >= threshold)
        
        # Enhanced result text with risk level
        if prediction == 0:
            if proba < 0.1:
                result_text = "EXCELLENT CREDIT RISK — Eligible for Premium Credit Cards"
                risk_level = "Excellent"
            elif proba < 0.3:
                result_text = "GOOD CREDIT RISK — Eligible for Standard Credit Cards"
                risk_level = "Good"
            else:
                result_text = "FAIR CREDIT RISK — Eligible for Basic Credit Cards"
                risk_level = "Fair"
            result_class = "good"
        else:
            result_text = "BAD CREDIT RISK — Not Eligible for Credit Card"
            risk_level = "Poor"
            result_class = "bad"

        # Get credit card recommendations
        recommendations = get_credit_recommendations(prediction, proba)

        return render_template("result.html", 
                             probability=round(proba, 4),
                             probability_percent=round(proba * 100, 2),
                             threshold=round(threshold, 4),
                             threshold_percent=round(threshold * 100, 2),
                             result=result_text,
                             result_class=result_class,
                             risk_level=risk_level,
                             recommendations=recommendations,
                             user_age=age_years,
                             user_employment=employed_years)

    except Exception as e:
        return render_template("result.html", 
                             error=f"Error occurred: {str(e)}",
                             threshold=round(threshold, 4),
                             threshold_percent=round(threshold * 100, 2))

@app.route("/about")
def about_page():
    """About page explaining the credit risk assessment"""
    return render_template("about.html", 
                         threshold=round(threshold, 4),
                         threshold_percent=round(threshold * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)