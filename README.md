# Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to
leave the company, based on their profile (income, satisfaction, overtime,
tenure, etc.). Built for the project expo — includes a trained model and a
live web demo.

## What's in this folder

```
attrition_project/
├── data/
│   └── emp_attrition.csv       ← IBM HR Analytics dataset (1,470 employees)
├── train_model.py              ← trains the model, prints accuracy, saves it
├── model/                      ← created after you run train_model.py
│   ├── model.pkl                (the trained model)
│   ├── encoders.pkl             (converts text -> numbers)
│   ├── metrics.json             (accuracy, F1 score, top factors)
│   └── ...
└── web/
    ├── app.py                  ← Flask web server (the live demo)
    └── templates/index.html    ← the web page
```

## How to run it (do this once per teammate's laptop)

```bash
# 1. Install the required Python libraries
pip install pandas scikit-learn flask joblib

# 2. Train the model (only needs to be done once — creates the model/ folder)
python train_model.py

# 3. Start the web app
cd web
python app.py

# 4. Open in your browser
http://127.0.0.1:5000
```

## How the model works (explain this at your expo booth)

1. **Data**: 1,470 real (anonymized) employee records — age, income,
   satisfaction scores, overtime, department, etc. — each labeled with
   whether that employee actually left ("Attrition": Yes/No).
2. **Training**: A **Random Forest** model (many small decision trees voting
   together) learns the patterns that separate people who left from people
   who stayed.
3. **Evaluation**: The model is tested on 294 employees it never saw during
   training. Current results:
   - **Accuracy: ~83%**
   - **F1 score (for the "Yes, will leave" class): ~0.46**
   - Catches roughly **47% of employees who actually leave** — the harder,
     more important number for this kind of imbalanced problem (most
     employees stay, so raw accuracy alone is a misleading headline stat).
4. **Top predictors** the model relies on most: Monthly Income, Age, Total
   Working Years, Overtime, Years at Company, Distance From Home.
5. **Web demo**: enter a hypothetical employee's details, get an instant
   risk percentage and the specific factors driving that prediction.

## Team role suggestions

| Person | Focus |
|---|---|
| Data / model person | Re-run `train_model.py`, try tweaking parameters, explain the metrics |
| Web/UI person | Customize `web/templates/index.html`, style tweaks, add more fields |
| Presentation lead | Slides, live demo script, explain "why" using metrics.json |
| Everyone | Understand the 5 steps above well enough to answer judge questions |

## Likely judge questions (and quick answers)

- **"Why Random Forest?"** — It handles mixed numeric/categorical data well
  and gives us feature importance (which factors matter most) for free.
- **"Why not 100% accuracy?"** — Real employees leave for many personal
  reasons the data can't capture; 100% would actually signal the model
  memorized the data rather than learning general patterns (overfitting).
- **"How do you handle imbalance?"** (only 16% of employees left) — we use
  `class_weight="balanced_subsample"` and a lowered decision threshold
  (0.35 instead of 0.5) so the model doesn't just lazily predict "No" for
  everyone.
- **"Is this real company data?"** — No, it's a well-known public dataset
  released by IBM for research/education, with realistic but fictional
  employee records.
