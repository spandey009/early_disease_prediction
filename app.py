from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Early Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0e1117;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
.result-card {
    background: #ffffff;
    color: #1f2937;
    border-radius: 22px;
    padding: 35px 25px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 18px;
}
.risk-number {
    font-size: 65px;
    font-weight: 800;
    margin: 8px 0;
}
.risk-label {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
}
.risk-track {
    width: 100%;
    height: 17px;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
    margin: 20px 0 8px 0;
}
.suggestion-card {
    background: #172a25;
    border: 1px solid #28745c;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
}
.info-text {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Early Disease Prediction System")
st.write(
    "Enter the health details below to view the model's prediction score "
    "and general health suggestions."
)

st.warning(
    "Educational project only. Results are not a medical diagnosis "
    "or clinically validated disease probability."
)

DISEASE_FILES = {
    "Diabetes": (
        "diabetes_model.joblib",
        "diabetes_feature_columns.joblib"
    ),
    "Heart Disease": (
        "heart_disease_model.joblib",
        "heart_disease_feature_columns.joblib"
    ),
    "Hypertension": (
        "hypertension_model.joblib",
        "hypertension_feature_columns.joblib"
    ),
    "Generic Demonstration": (
        "best_model.joblib",
        "feature_columns.joblib"
    ),
}

DEFAULTS = {
    "Pregnancies": 0.0,
    "Glucose": 110.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35.0,
    "age": 45.0,
    "sex": 1.0,
    "cp": 0.0,
    "trestbps": 120.0,
    "chol": 200.0,
    "fbs": 0.0,
    "restecg": 0.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 0.0,
    "slope": 0.0,
    "ca": 0.0,
    "thal": 0.0,
    "bmi": 25.0,
    "blood_pressure": 120.0,
    "glucose": 110.0,
    "cholesterol": 200.0,
    "smoking": 0.0,
    "sbp": 120.0,
    "dbp": 80.0,
}

LABELS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Glucose level",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin level",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age",
    "age": "Age",
    "sex": "Sex code",
    "cp": "Chest pain type code",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar code",
    "restecg": "ECG code",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina code",
    "oldpeak": "ST depression",
    "slope": "ST slope code",
    "ca": "Major vessels code",
    "thal": "Thalassemia code",
    "bmi": "BMI",
    "blood_pressure": "Blood pressure",
    "glucose": "Glucose level",
    "cholesterol": "Cholesterol",
    "smoking": "Smoking code",
    "sbp": "Systolic blood pressure",
    "dbp": "Diastolic blood pressure",
}

SUGGESTIONS = {
    "Diabetes": [
        "Limit sugary drinks and highly processed foods.",
        "Eat balanced meals with vegetables and fibre-rich foods.",
        "Stay physically active according to your ability.",
        "Discuss glucose screening with a healthcare professional."
    ],
    "Heart Disease": [
        "Avoid smoking and second-hand smoke.",
        "Choose heart-friendly foods and limit excess salt and saturated fat.",
        "Maintain regular physical activity as advised by a professional.",
        "Discuss blood pressure and cholesterol checks with your doctor."
    ],
    "Hypertension": [
        "Reduce excess salt in your diet.",
        "Stay physically active and maintain a healthy routine.",
        "Avoid smoking and limit alcohol.",
        "Measure blood pressure correctly and consult a healthcare professional."
    ],
    "Generic Demonstration": [
        "Use the result only as a demonstration of machine-learning output.",
        "Maintain a balanced diet and regular physical activity.",
        "Discuss any health concerns with a qualified healthcare professional."
    ]
}


def load_model_and_features(disease):
    model_file, feature_file = DISEASE_FILES[disease]
    model_path = MODEL_DIR / model_file
    feature_path = MODEL_DIR / feature_file

    if not model_path.exists() or not feature_path.exists():
        return None, None, (
            f"The model for {disease} is not available yet. "
            f"Required file: {model_file}"
        )

    try:
        model = joblib.load(model_path)
        features = joblib.load(feature_path)
        return model, list(features), None
    except Exception as error:
        return None, None, f"Unable to load model: {error}"


def get_classifier(model):
    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            return model.named_steps["classifier"]
        return list(model.named_steps.values())[-1]
    return model


def get_positive_score(model, row, prediction):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(row)[0]
    classifier = get_classifier(model)
    classes = list(classifier.classes_)

    positive_index = None

    for index, label in enumerate(classes):
        if str(label).strip().lower() in ("1", "1.0", "yes", "true", "positive"):
            positive_index = index
            break

    if positive_index is None:
        if len(classes) == 2:
            positive_index = 1
        else:
            return None

    return float(probabilities[positive_index]) * 100


def show_result(disease, score):
    if score is None:
        st.error("This model does not provide a usable class score.")
        return

    if score < 30:
        level = "LOW MODEL SCORE"
        color = "#10b981"
        message = (
            "The model assigned a relatively low score to the positive class. "
            "This does not rule out disease."
        )
    elif score < 60:
        level = "MODERATE MODEL SCORE"
        color = "#f59e0b"
        message = (
            "The model assigned a moderate score to the positive class. "
            "Consider discussing your health indicators with a professional."
        )
    else:
        level = "HIGH MODEL SCORE"
        color = "#ef4444"
        message = (
            "The model assigned a high score to the positive class. "
            "Consider consulting a healthcare professional for proper assessment."
        )

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:17px;color:#64748b;font-weight:600;">
                {disease.upper()} — POSITIVE-CLASS MODEL SCORE
            </div>
            <div class="risk-number" style="color:{color};">
                {score:.1f}%
            </div>
            <div class="risk-label" style="color:{color};">
                {level}
            </div>
            <div style="font-size:17px;color:#475569;">
                {message}
            </div>
            <div class="risk-track">
                <div style="
                    width:{score}%;
                    height:100%;
                    background:{color};
                    border-radius:20px;
                "></div>
            </div>
            <div style="
                display:flex;
                justify-content:space-between;
                color:#64748b;
                font-size:13px;
            ">
                <span>0% — Low score</span>
                <span>Moderate</span>
                <span>100% — High score</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Suggestions")

    for suggestion in SUGGESTIONS[disease]:
        st.markdown(
            f"""
            <div class="suggestion-card">
                <span style="color:#34d399;font-size:18px;">✓</span>
                <span style="color:#f3f4f6;"> {suggestion}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Important: The percentage is the model's score for the positive class, "
        "not a clinically validated probability of developing the disease. "
        "Follow medical advice for diagnosis and treatment."
    )


disease = st.selectbox(
    "Select disease",
    list(DISEASE_FILES.keys())
)

model, features, error = load_model_and_features(disease)

if error:
    st.error(error)
    st.info(
        "Place the required trained model and feature-column files in the "
        "`models` folder. Then restart the application."
    )
    st.stop()

st.subheader("Enter Health Details")
st.write("Fill in the available information below.")

values = {}

with st.form("health_form"):
    columns = st.columns(2)

    for index, feature in enumerate(features):
        with columns[index % 2]:
            label = LABELS.get(
                str(feature),
                str(feature).replace("_", " ").replace("-", " ").title()
            )

            values[feature] = st.number_input(
                label,
                value=float(DEFAULTS.get(str(feature), 0.0)),
                key=f"input_{disease}_{feature}"
            )

    submitted = st.form_submit_button(
        "Predict Risk Score",
        use_container_width=True,
        type="primary"
    )

if submitted:
    row = pd.DataFrame([values], columns=features)

    try:
        prediction = model.predict(row)[0]
        score = get_positive_score(model, row, prediction)

        if disease == "Generic Demonstration":
            st.info(
                "This model uses a synthetic generic dataset. "
                "Its score must not be interpreted as the risk of a specific disease."
            )
            show_result(disease, score)
        elif score is None:
            st.error("Could not calculate the positive-class score.")
        else:
            show_result(disease, score)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the selected model and its feature columns match."
        )
else:
    st.info("Enter the details and click **Predict Risk Score** to see the result.")