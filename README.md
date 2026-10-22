# Diabetes Prediction Using Machine Learning

An end-to-end machine learning project that explores health-related measurements, compares classification models, tunes Logistic Regression, and evaluates diabetes outcome predictions.

> **Disclaimer:** This project is for educational purposes only and is not intended for medical diagnosis. Predictions should not be used as a substitute for professional medical advice.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Model Performance](#model-performance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Project Overview

Diabetes prediction is a binary classification task. This project uses the Pima Indians Diabetes Dataset to investigate patterns in health-related measurements and compare machine learning approaches for predicting the target variable, `Outcome`.

The project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Handling selected zero values as missing measurements
- Stratified train-test splitting
- Cross-validation model comparison
- Logistic Regression hyperparameter tuning with GridSearchCV
- Held-out test evaluation
- Permutation feature importance
- Saving a trained model pipeline

## Dataset

The project uses the **Pima Indians Diabetes Dataset**.

| Item | Description |
|---|---|
| Records | 768 |
| Input features | 8 |
| Target | `Outcome` |
| Negative class | `0` |
| Positive class | `1` |

### Input Features

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

The dataset represents a specific population, so model performance may not generalize to other populations.

## Workflow

1. Load and inspect the dataset.
2. Perform EDA on feature distributions, correlations, and class balance.
3. Investigate zero values in selected measurement columns.
4. Replace selected zero values with missing values and impute using training-set medians.
5. Split data into stratified training and test sets.
6. Compare six classification algorithms using 5-fold stratified cross-validation.
7. Tune Logistic Regression with GridSearchCV using ROC-AUC scoring.
8. Evaluate models on a held-out test set.
9. Analyze feature importance using permutation importance.
10. Save the trained model pipeline for future use.

## Model Performance

### Cross-Validation Results

Mean scores across five stratified folds.

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 76.2% | 70.1% | 0.672 | 0.844 |
| Support Vector Machine (SVM) | 76.1% | 76.2% | 0.690 | 0.837 |
| Random Forest | 76.9% | 59.3% | 0.642 | 0.822 |
| Gradient Boosting | 75.6% | 59.8% | 0.631 | 0.819 |
| XGBoost | 75.6% | 59.3% | 0.628 | 0.817 |
| K-Nearest Neighbors (KNN) | 73.0% | 56.5% | 0.594 | 0.784 |

### Held-Out Test-Set Comparison

Results from the separate comparison on the 154-record test set. KNN was not measured in this comparison.

| Model | Accuracy | Recall | F1-Score |
|---|---:|---:|---:|
| Logistic Regression | 73.4% | 70.4% | 0.650 |
| Support Vector Machine (SVM) | 72.7% | 72.2% | 0.650 |
| Random Forest | 74.0% | 57.4% | 0.608 |
| Gradient Boosting | 76.0% | 57.4% | 0.626 |
| XGBoost | 77.9% | 63.0% | 0.667 |
| KNN | Not measured | — | — |

## Hyperparameter Tuning

Logistic Regression was tuned using `GridSearchCV`, with ROC-AUC as the scoring metric.

| Item | Result |
|---|---|
| Best parameters | `C = 0.1`, `solver = lbfgs` |
| Mean cross-validation ROC-AUC | 0.845 |
| Test accuracy | 70.78% |
| Test ROC-AUC | 0.8104 |
| Positive-class recall | 66.67% |
| Positive-class F1-score | 0.615 |

The tuned Logistic Regression results are reported separately from the initial model comparison. The different test accuracy values should be checked against the corresponding notebook runs and preprocessing pipelines.

## Key Findings

- **Highest cross-validation accuracy:** Random Forest — 76.9%.
- **Highest cross-validation recall:** SVM — 76.2%.
- **Highest cross-validation F1-score:** SVM — 0.690.
- **Highest initial cross-validation ROC-AUC:** Logistic Regression — 0.844; the tuned version achieved 0.845.
- **Highest observed test accuracy:** XGBoost — 77.9%.
- **Highest observed test F1-score:** XGBoost — 0.667.
- **Highest observed test recall:** SVM — 72.2%.

These results describe the reported experiments and do not establish clinical effectiveness or guarantee performance on unseen populations.

## Tech Stack

| Category | Tools & Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Evaluation & Tuning | Stratified Cross-Validation, GridSearchCV, ROC-AUC |
| Web Application | Streamlit |
| Model Persistence | Joblib / Pickle |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Cloud Deployment | AWS (EC2) |

## Project Structure

The exact filenames may vary depending on the files included in your repository. A typical structure for this project may look like:

```text
diabetes-prediction-ml/
├── data/                  # Dataset files, if included
├── notebooks/             # EDA and model development notebooks, if included
├── models/                # Saved model pipeline, if included
├── app.py                 # Interactive app, if included
├── requirements.txt       # Python dependencies, if included
└── README.md
```

Only keep entries in this structure that match files actually present in your repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Waihaad/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Create and activate a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

If your repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by your notebook or application.

### 4. Run the project

Open the relevant Jupyter notebook and run its cells in order. If your repository includes a Streamlit application and its required model artifacts, launch it with:

```bash
streamlit run app.py
```

The application command is applicable only if `app.py` is a Streamlit app and all required files are available.

## Future Improvements

- Model Optimization:Explore advanced techniques to improve predictive performance.
- Explainable AI: Use SHAP or LIME to explain individual predictions.
- Generalization Testing:** Evaluate performance on independent datasets and diverse populations.
- Data Quality: Improve missing-value handling and feature engineering.
- Responsible AI: Assess model bias and communicate prediction limitations.
- Model Monitoring:Explore ways to detect changes in data and model performance over time.

## Deployment

-The Streamlit application was deployed on **AWS EC2** as part of the project     deployment process. The AWS resources were subsequently removed to avoid ongoing  infrastructure costs.

Screenshots of the application interface and prediction results demonstrate the application's functionality.

**Note: A live demo is currently unavailable because the AWS resources have been removed.

## Author

**Waihaad Ahmed**

- [LinkedIn](https://www.linkedin.com/in/waihaad-ahmed-b73b12307)
- [GitHub](https://github.com/Waihaad)
- [Email](mailto:waihaad@gmail.com)
