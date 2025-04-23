import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔍 Predict Customer Churn")
st.write("Enter customer details below to predict churn using Logistic Regression.")

# Create input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submit = st.form_submit_button("Predict Churn")

# Prepare prediction
if submit:
    try:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result")
        st.write(f"**Customer ID:** {customerID}")
        st.write(f"**Churn Prediction:** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Churn Probability:** {probability:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
