# Diabetes Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict diabetes outcomes from health-related measurements. It covers exploratory data analysis, data preprocessing, model comparison, hyperparameter tuning, model evaluation, and feature importance analysis.

The project is built using Python and scikit-learn, with XGBoost included for model comparison.

**Note:** This project is for educational purposes only and is not intended for medical diagnosis.

## Dataset

The project uses the Pima Indians Diabetes Dataset.

- **Total records:** 768
- **Input features:** 8
- **Target variable:** `Outcome`
- `0` — Negative class
- `1` — Positive class

### Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The dataset represents a specific population and may not generalize to other populations.

## Project Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis (EDA).
3. Analyze feature distributions, correlations, and class balance.
4. Investigate zero values in selected measurement columns.
5. Replace selected zero values with missing values and impute using training-set medians.
6. Split the data into stratified training and test sets.
7. Compare six machine learning algorithms using cross-validation.
8. Tune Logistic Regression using GridSearchCV and ROC-AUC scoring.
9. Evaluate the tuned model on the held-out test set.
10. Analyze feature importance using permutation importance.
11. Save the trained model pipeline for future use.

## Machine Learning Models

The following algorithms were compared:

- Logistic Regression
- Support Vector

## Author

**Waihaad Ahmed**

[LinkedIn](https://www.linkedin.com/in/waihaad-ahmed-b73b12307) • [GitHub](https://github.com/Waihaad) • [Email](mailto:waihaad@gmail.com)