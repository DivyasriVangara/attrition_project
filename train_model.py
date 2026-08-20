"""
Employee Attrition Prediction - Model Training Script
--------------------------------------------------------
What this file does (plain English, for the team):

1. Loads the employee dataset (data/emp_attrition.csv)
2. Cleans it up:
     - Drops useless columns (same value for every employee)
     - Converts text columns (like "Yes"/"No", "Male"/"Female") into numbers
       because ML models only understand numbers
3. Splits data into "training" (80%) and "testing" (20%) sets
4. Trains a Random Forest model - basically hundreds of small decision
   trees voting together on "will this employee leave?"
5. Checks how accurate it is on data it has NEVER seen before
6. Saves the trained model + the encoders to model/ so the web app can use them

Run this once with:  python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import json
import os

DATA_PATH = "data/emp_attrition.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("STEP 1: Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {df.shape[0]} employees, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# STEP 2: Drop columns that carry no useful signal
# ---------------------------------------------------------------------------
# EmployeeCount, Over18, StandardHours are the SAME value for every single
# employee in this dataset -> a model can't learn anything from a constant.
# EmployeeNumber is just an ID, not a real feature.
drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------------------------------------------------------------------------
# STEP 3: Encode text columns into numbers
# ---------------------------------------------------------------------------
target_col = "Attrition"
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove(target_col)

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])  # No=0, Yes=1

feature_cols = [c for c in df.columns if c != target_col]

print(f"STEP 2-3: Encoded {len(categorical_cols)} text columns -> numbers")
print(f"  Features used: {len(feature_cols)}")

# ---------------------------------------------------------------------------
# STEP 4: Train / test split
# ---------------------------------------------------------------------------
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"STEP 4: Train set = {len(X_train)} employees, Test set = {len(X_test)} employees")

# ---------------------------------------------------------------------------
# STEP 5: Train the model
# ---------------------------------------------------------------------------
# class_weight="balanced" matters here: only ~16% of employees in this
# dataset actually left, so without this the model gets lazy and just
# predicts "No" for everyone and still looks "accurate".
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
)
print("STEP 5: Training Random Forest model...")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# STEP 6: Evaluate
# ---------------------------------------------------------------------------
# We lower the decision threshold from the default 0.5 to 0.35.
# Why: missing a real leaver (false negative) costs HR far more than a
# false alarm (false positive) - so we deliberately make the model a bit
# more "trigger-happy" about flagging risk. This is a normal, explainable
# real-world choice for attrition prediction.
y_proba = model.predict_proba(X_test)[:, 1]
THRESHOLD = 0.35
y_pred = (y_proba >= THRESHOLD).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== RESULTS (show this on your slide) =====")
print(f"Accuracy: {acc*100:.2f}%")
print(f"F1 Score: {f1:.3f}")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

# Feature importance - "what drives attrition" - great for your presentation
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nTop 10 factors driving attrition:")
print(importances.head(10))

# ---------------------------------------------------------------------------
# Save everything the web app needs
# ---------------------------------------------------------------------------
joblib.dump(model, f"{MODEL_DIR}/model.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
joblib.dump(target_encoder, f"{MODEL_DIR}/target_encoder.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
joblib.dump(THRESHOLD, f"{MODEL_DIR}/threshold.pkl")

metrics = {
    "accuracy": round(acc * 100, 2),
    "f1_score": round(f1, 3),
    "threshold": THRESHOLD,
    "top_factors": importances.head(8).round(3).to_dict(),
}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved model + encoders + metrics to '{MODEL_DIR}/'")
print("Now run the web app: cd web && python app.py")
