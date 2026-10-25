from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)
features = ["age", "bmi", "blood_pressure", "glucose", "cholesterol", "smoking"]
X = data[features]
y = data["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, random_state=42
    ),
}

rows = []
fitted = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
    }
    rows.append(row)
    fitted[name] = model
    print(pd.Series(row).to_string())
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
results.to_csv(ROOT / "data" / "model_comparison.csv", index=False)
best_name = results.iloc[0]["Model"]
joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
joblib.dump(features, MODEL_DIR / "feature_columns.joblib")
with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as file:
    json.dump({
        "model": best_name,
        "features": features,
        "dataset": "synthetic demonstration data",
        "warning": "Metrics describe simulated labels and are not evidence of clinical performance."
    }, file, indent=2)

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))
print(f"\nBest model by F1: {best_name}")
print("Saved model and feature list to models/.")
print("Saved comparison to data/model_comparison.csv.")
print("\nReminder: high scores on synthetic labels do not establish real-world medical usefulness.")
