from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data" / "dataset.csv")

expected = [
    "patient_id", "age", "bmi", "blood_pressure",
    "glucose", "cholesterol", "smoking", "label"
]
print("=" * 60)
print("DATASET AUDIT")
print("=" * 60)
print("Shape:", data.shape)
print("\nFirst five rows:")
print(data.head().to_string(index=False))
print("\nColumn validation:", data.columns.tolist() == expected)
print("\nMissing values:\n", data.isna().sum().to_string())
print("\nDuplicate rows:", data.duplicated().sum())
print("Duplicate IDs:", data["patient_id"].duplicated().sum())
print("\nLabel counts:\n", data["label"].value_counts().sort_index().to_string())
print("\nNumeric summary:\n", data.drop(columns="patient_id").describe().round(2).to_string())
print("\nIMPORTANT: this is a synthetic demonstration dataset, not real patient data.")
