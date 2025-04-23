import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load the full pipeline model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# App title
st.title("🔍 Customer Churn Prediction App")
st.write("Upload a CSV file with customer data to predict churn probability using Logistic Regression.")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview", df.head())

    try:
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        # Append predictions
        df["Churn Prediction"] = predictions
        df["Churn Probability"] = probabilities

        st.success("✅ Churn Prediction Complete")
        st.write(df[["Churn Prediction", "Churn Probability"]])

        st.download_button(
            label="📥 Download Results",
            data=df.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
else:
    st.info("📂 Please upload a CSV file to begin.")
