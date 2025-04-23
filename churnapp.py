import streamlit as st
import pandas as pd
import pickle

# Load pipeline model (preprocessing + Logistic Regression)
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔍 Predict Customer Churn")
st.write("Enter customer details below to predict churn using Logistic Regression.")

# Example feature inputs (customize based on your dataset!)
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# Build input DataFrame
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "InternetService": InternetService,
    "Contract": Contract,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}])

# Predict
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result:")
        st.write(f"**Churn Prediction:** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Churn Probability:** {probability:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
