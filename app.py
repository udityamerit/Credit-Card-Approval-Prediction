from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model, threshold, and training columns
model = joblib.load("models/Random_Forest_best_model.pkl")
with open("models/best_threshold.txt", "r") as f:
    threshold = float(f.read())
train_columns = joblib.load("models/train_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get age and employment in years, convert to negative days
            age_years = int(request.form.get("AGE_YEARS", 0))
            employed_years = float(request.form.get("EMPLOYED_YEARS", 0))

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
                'DAYS_BIRTH': -abs(age_years * 365),
                'DAYS_EMPLOYED': -abs(employed_years * 365),
                'FLAG_WORK_PHONE': int(request.form.get("FLAG_WORK_PHONE", 0)),
                'FLAG_PHONE': int(request.form.get("FLAG_PHONE", 0)),
                'FLAG_EMAIL': int(request.form.get("FLAG_EMAIL", 0)),
                'CNT_FAM_MEMBERS': float(request.form.get("CNT_FAM_MEMBERS", 0)),
            }

            # DataFrame + one-hot encoding
            user_df = pd.DataFrame([user_data])
            user_encoded = pd.get_dummies(user_df)

            # Add missing columns
            for col in train_columns:
                if col not in user_encoded.columns:
                    user_encoded[col] = 0

            user_encoded = user_encoded[train_columns]

            # Predict
            proba = model.predict_proba(user_encoded)[:, 1][0]
            prediction = int(proba >= threshold)
            result_text = "GOOD CREDIT RISK — Eligible for Credit Card" if prediction == 0 else "BAD CREDIT RISK — Not Eligible for Credit Card"

            return render_template("index.html", probability=round(proba, 4), result=result_text)

        except Exception as e:
            return render_template("index.html", error=f"❌ Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
