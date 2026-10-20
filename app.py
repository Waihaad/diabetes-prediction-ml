
import streamlit as st
import pandas as pd
from src.predict import load_model, predict_diabetes

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction")
st.write(
    "Enter the measurements below to receive a machine-learning "
    "prediction. This tool is for educational purposes only."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Check that "
        "models/diabetes_logistic_regression.pkl exists."
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Health Measurements")

    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose", min_value=0.0, max_value=300.0, value=120.0
    )
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0.0, max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0
    )
    insulin = st.number_input(
        "Insulin", min_value=0.0, max_value=900.0, value=80.0
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=80.0, value=25.0
    )
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=30
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    prediction, probability = predict_diabetes(model, input_data)

    st.subheader("Prediction Result")
    st.metric(
        "Estimated positive-class probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning("The model predicts the positive class.")
    else:
        st.success("The model predicts the negative class.")

    st.caption(
        "This is an educational model output, not a medical diagnosis. "
        "Do not use it to make healthcare decisions."
    )
