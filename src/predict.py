
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "diabetes_logistic_regression.pkl"
)


def load_model():
    """Load the saved machine learning pipeline."""
    return joblib.load(MODEL_PATH)


def predict_diabetes(model, input_data):
    """Return the model prediction and estimated positive-class probability."""
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])

    return prediction, probability
