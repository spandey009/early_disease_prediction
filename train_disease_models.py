from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Pima dataset has no header row.
DIABETES_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def train_diabetes():
    path = DATA_DIR / "diabetes.csv"

    df = pd.read_csv(path, header=None)

    # If the file already has headers, read it normally.
    if "Outcome" in df.columns:
        df = pd.read_csv(path)
    else:
        df.columns = DIABETES_COLUMNS

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Outcome"])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)

    # In this dataset, zero values for these measurements
    # generally represent missing measurements.
    for column in [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]:
        X[column] = X[column].replace(0, float("nan"))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Save model and matching feature names.
    joblib.dump(model, MODEL_DIR / "diabetes_model.joblib")
    joblib.dump(
        list(X.columns),
        MODEL_DIR / "diabetes_feature_columns.joblib",
    )

    print("Diabetes model saved.")


def train_optional_dataset(name, filename, target):
    path = DATA_DIR / filename

    if not path.exists():
        print(f"{filename} not found; skipping {name}.")
        return

    df = pd.read_csv(path)
    if target not in df.columns:
        print(f"Target '{target}' not found in {filename}; skipping.")
        return

    df = df.dropna(subset=[target])
    df = df.drop(columns=["patient_id", "id"], errors="ignore")

    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        print(f"Insufficient target classes in {filename}; skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    model_name = f"{name}_model.joblib"
    features_name = f"{name}_feature_columns.joblib"

    joblib.dump(model, MODEL_DIR / model_name)
    joblib.dump(list(X.columns), MODEL_DIR / features_name)

    predictions = model.predict(X_test)
    print(f"{name}: accuracy = {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    train_diabetes()

    train_optional_dataset(
        "heart_disease",
        "heart_disease.csv",
        "target",
    )

    train_optional_dataset(
        "hypertension",
        "hypertension.csv",
        "level",
    )

    print("Training script completed.")