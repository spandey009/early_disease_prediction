
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "best_model.joblib"
FEATURES_FILE = ROOT / "models" / "feature_columns.joblib"
INFO_FILE = ROOT / "models" / "model_info.json"

st.set_page_config(
    page_title="Early Health Risk Demo",
    page_icon="🩺",
    layout="centered"
)

st.title("Early Health Risk Prediction")
st.subheader("Machine Learning Demonstration")

st.warning(
    "Educational prototype using a dataset that includes synthetic data. "
    "This model has not been clinically validated. Do not use its output "
    "to diagnose disease or make treatment decisions."
)

if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
    st.error("Required model files are missing.")
    st.info("Run Notebook 4 (Model Training) and Notebook 5 (Model Evaluation) first.")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
    features = list(joblib.load(FEATURES_FILE))
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

if INFO_FILE.exists():
    try:
        with open(INFO_FILE, encoding="utf-8") as file:
            model_info = json.load(file)
        model_name = model_info.get("model", "Best saved model")
    except (OSError, json.JSONDecodeError):
        model_name = "Best saved model"
else:
    model_name = "Best saved model"

st.caption(f"Loaded model: {model_name}")

expected_features = [
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "smoking"
]

if set(features) != set(expected_features):
    st.error(
        "The saved model features do not match the current input form. "
        f"Expected: {expected_features}. Found: {features}"
    )
    st.stop()

st.subheader("Enter example health measurements")

with st.form("risk_form"):
    age = st.slider("Age (years)", 25, 75, 45)
    bmi = st.slider("BMI", 18.0, 40.0, 28.0, 0.1)
    blood_pressure = st.slider(
        "Blood pressure (illustrative mmHg)", 60, 180, 120
    )
    glucose = st.slider("Glucose (illustrative units)", 70, 200, 115)
    cholesterol = st.slider(
        "Cholesterol (illustrative units)", 150, 280, 210
    )
    smoking = st.selectbox("Smoking status", ["No", "Yes"])

    submitted = st.form_submit_button("Generate demonstration prediction")

if submitted:
    input_data = {
        "age": age,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "smoking": 1 if smoking == "Yes" else 0
    }

    row = pd.DataFrame([input_data])[features]

    try:
        prediction = int(model.predict(row)[0])

        st.subheader("Model Output")
        st.write(f"Predicted synthetic label: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)

            if 1 in classes:
                score = float(probabilities[classes.index(1)])
                st.metric("Model score for synthetic label 1", f"{score:.1%}")

        st.caption(
            "This is a model output for a synthetic binary label, not a "
            "medically meaningful disease probability. Consult a qualified "
            "healthcare professional for health concerns."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
