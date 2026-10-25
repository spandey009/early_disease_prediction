# Early Disease Prediction Using Machine Learning

An educational machine-learning project that explores health-related data to demonstrate disease classification, model-based risk scoring, and explainable predictions through a Streamlit web application.

## Overview

Early Disease Prediction is designed to demonstrate how machine-learning algorithms can analyse health indicators and produce predictions through a simple, interactive interface.

The application includes a Streamlit interface, trained classification models, input forms, prediction scores, and general health recommendations.

**Important:** This is an educational prototype, not a clinically validated medical system. Model scores must not be interpreted as a person's actual probability of developing a disease.

## Objectives

- Explore and preprocess health-related datasets.
- Train machine-learning classification models.
- Compare model performance using appropriate evaluation metrics.
- Build an interactive application using Streamlit.
- Display model prediction scores in an understandable format.
- Provide general health information and preventive suggestions.
- Highlight ethical considerations and limitations of AI in healthcare.

## Features

- Interactive health-data input forms.
- Disease selection where the corresponding trained model is available.
- Model prediction and positive-class score display.
- Visual score indicator.
- General health suggestions.
- Educational disclaimer and limitations.
- Support for model training and exploratory data analysis.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Models:** Logistic Regression, Random Forest, and XGBoost where configured
- **Visualization:** Matplotlib
- **Explainability:** SHAP, where supported by the configured application
- **Web Application:** Streamlit
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook and Visual Studio Code

## Project Structure

```text
early_disease_prediction/
├── app.py
├── train_disease_models.py
├── requirements.txt
├── README.md
├── data/
│   ├── dataset.csv
│   ├── data_dictionary.csv
│   └── disease-specific datasets, when available
├── models/
│   └── saved model and feature-column files
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
└── reports/
    └── evaluation and data-audit reports, when available
```

The exact files may differ depending on which datasets and models have been generated.

## Dataset

The project contains a 5,000-record synthetic demonstration dataset with the following fields:

| Feature | Description |
|---|---|
| `patient_id` | Synthetic record identifier |
| `age` | Age |
| `bmi` | Body Mass Index |
| `blood_pressure` | Blood pressure measurement |
| `glucose` | Glucose measurement |
| `cholesterol` | Cholesterol measurement |
| `smoking` | Dataset-defined smoking indicator |
| `label` | Synthetic target label |

The generic dataset is synthetic and intended for demonstrating a machine-learning workflow. Its labels are not clinical diagnoses.

Disease-specific models require their corresponding datasets and trained model files. Results from the generic synthetic model must not be presented as diabetes, heart disease, or hypertension risk.

## Machine-Learning Workflow

1. Load the dataset.
2. Audit data quality and inspect missing values.
3. Perform exploratory data analysis.
4. Preprocess the input features.
5. Split the dataset into training and testing sets.
6. Train the configured classification models.
7. Evaluate model performance using appropriate metrics.
8. Save trained models and feature information.
9. Load available models in the Streamlit application.
10. Display predictions and model scores.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/spandey009/early_disease_prediction.git
cd early_disease_prediction
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use compatible package versions listed in `requirements.txt`, particularly for loading saved scikit-learn models.

## Run the Application

From the project root, execute:

```bash
streamlit run app.py
```

Open the local address displayed in the terminal.

The application can only load a disease-specific model when its required model and feature-column files are present in the `models/` directory.

## Train the Models

If the relevant datasets are available, run:

```bash
python train_disease_models.py
```

The training script uses the configured datasets and saves the models it successfully trains.

For example, the diabetes model requires `data/diabetes.csv`. Heart disease and hypertension models require their respective datasets in the expected formats.

A model is skipped if its required dataset or target column is missing. Check the training script and dataset schema before training additional models.

## Model Evaluation

Model performance should be reported using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, where appropriate

Actual evaluation values should be taken from the saved evaluation reports or reproducible evaluation runs. No performance value should be assumed without supporting results.

## Understanding Prediction Scores

The application displays a model score for the positive class when the model supports probability prediction.

A score such as `24%` means the model assigned a score of 0.24 to that class for the supplied inputs. It does **not** establish a clinically validated 24% chance of developing the disease.

The score may be affected by the dataset, class balance, model calibration, and training methodology.

## Ethical Considerations

- This project is for educational and research demonstration only.
- Synthetic labels must not be treated as medical ground truth.
- Predictions must not replace clinical examination or professional medical advice.
- Health data should be handled responsibly and with appropriate privacy safeguards.
- Model limitations and dataset provenance should be disclosed.
- Disease-specific conclusions require suitable datasets, reliable evaluation, and appropriate clinical validation.

## Limitations

- The generic demonstration dataset is synthetic.
- Synthetic results may not generalize to real patients.
- Not every disease-specific model may be available in every project version.
- Model scores are not necessarily calibrated probabilities.
- The application is not approved or validated for medical diagnosis.
- General health suggestions are informational and are not individualized treatment plans.

## Future Scope

- Integrate larger, well-documented public health datasets.
- Evaluate disease-specific models on held-out test data.
- Improve probability calibration and class-imbalance handling.
- Add feature-importance and SHAP explanations where appropriate.
- Improve accessibility and input validation.
- Conduct external validation before considering real-world healthcare use.

## Disclaimer

This application is an academic machine-learning prototype. It does not diagnose, treat, cure, or prevent any disease. Do not make healthcare decisions based on its outputs. Consult a qualified healthcare professional for medical concerns.

## Author

**Saurabh Pandey**

GitHub: [spandey009](https://github.com/spandey009)

## License

No license is specified unless a `LICENSE` file is added to the repository.