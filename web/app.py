from flask import Flask, render_template, request
import joblib
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

app = Flask(__name__)

# Load everything saved by train_model.py, once, at startup
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "threshold.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# Only ask the user for the fields that matter most (top drivers) +
# a few standard HR fields, to keep the form short for a live demo.
# Every other feature gets a sensible default value behind the scenes.
FORM_FIELDS = [
    {"name": "Age", "label": "Age", "type": "number", "default": 30},
    {"name": "MonthlyIncome", "label": "Monthly Income (Rs / $)", "type": "number", "default": 5000},
    {"name": "TotalWorkingYears", "label": "Total Working Years", "type": "number", "default": 5},
    {"name": "YearsAtCompany", "label": "Years At This Company", "type": "number", "default": 3},
    {"name": "DistanceFromHome", "label": "Distance From Home (km)", "type": "number", "default": 5},
    {"name": "OverTime", "label": "Works Overtime?", "type": "select", "options": ["Yes", "No"], "default": "No"},
    {"name": "JobSatisfaction", "label": "Job Satisfaction (1=Low - 4=High)", "type": "number", "default": 3},
    {"name": "WorkLifeBalance", "label": "Work-Life Balance (1=Bad - 4=Great)", "type": "number", "default": 3},
    {"name": "Department", "label": "Department", "type": "select",
     "options": ["Sales", "Research & Development", "Human Resources"], "default": "Sales"},
    {"name": "JobRole", "label": "Job Role", "type": "select",
     "options": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                 "Manufacturing Director", "Healthcare Representative", "Manager",
                 "Sales Representative", "Research Director", "Human Resources"],
     "default": "Sales Executive"},
    {"name": "MaritalStatus", "label": "Marital Status", "type": "select",
     "options": ["Single", "Married", "Divorced"], "default": "Single"},
    {"name": "BusinessTravel", "label": "Business Travel", "type": "select",
     "options": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], "default": "Travel_Rarely"},
]

# Sensible default values for every column the model needs that we do NOT
# put on the form (keeps the demo form short without breaking the model).
DEFAULTS = {
    "DailyRate": 800, "Education": 3, "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 65,
    "JobInvolvement": 3, "JobLevel": 2, "MonthlyRate": 14000,
    "NumCompaniesWorked": 2, "PercentSalaryHike": 14, "PerformanceRating": 3,
    "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TrainingTimesLastYear": 2,
    "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
}


def build_feature_row(form_data):
    """Turn the submitted form into the exact row of numbers the model expects."""
    row = {}
    for col in feature_cols:
        if col in form_data:
            value = form_data[col]
        elif col in DEFAULTS:
            value = DEFAULTS[col]
        else:
            value = 0

        if col in encoders:  # text column -> needs encoding
            le = encoders[col]
            if value not in le.classes_:
                value = le.classes_[0]  # fallback if unseen category
            value = le.transform([value])[0]
        else:
            value = float(value)

        row[col] = value
    return pd.DataFrame([row])[feature_cols]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        form_data = {}
        for field in FORM_FIELDS:
            form_data[field["name"]] = request.form.get(field["name"], field["default"])

        X_row = build_feature_row(form_data)
        proba = model.predict_proba(X_row)[0][1]  # probability of "Yes" (will leave)
        prediction = "Yes" if proba >= threshold else "No"

        # Explain WHY, using the model's top overall factors + this employee's values
        reasons = []
        if form_data.get("OverTime") == "Yes":
            reasons.append("Works overtime regularly")
        if float(form_data.get("JobSatisfaction", 3)) <= 2:
            reasons.append("Low job satisfaction")
        if float(form_data.get("WorkLifeBalance", 3)) <= 2:
            reasons.append("Poor work-life balance")
        if float(form_data.get("YearsAtCompany", 3)) <= 1:
            reasons.append("Very new to the company")
        if float(form_data.get("MonthlyIncome", 5000)) < 3000:
            reasons.append("Relatively low income")
        if not reasons:
            reasons.append("No major single risk factor - overall pattern-based prediction")

        result = {
            "prediction": prediction,
            "risk_percent": round(proba * 100, 1),
            "reasons": reasons,
        }

    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        result=result,
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
