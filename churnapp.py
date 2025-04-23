#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:09:38 2025

@author: munen
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load model using pickle
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Load encoder and scaler using pickle
@st.cache_resource
def load_helpers():
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return encoder, scaler

encoder, scaler = load_helpers()

# App title
st.title("🔍 Customer Churn Prediction App")
st.write("Upload a CSV file with customer data to predict churn probability.")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview", df.head())

    try:
        # Assuming object columns are categorical and need encoding
        df_cat = df.select_dtypes(include='object')
        df_num = df.select_dtypes(include=np.number)

        df_encoded = encoder.transform(df_cat)
        df_scaled = scaler.transform(df_num)

        # Combine features
        X = np.hstack((df_encoded, df_scaled))

        # Prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Append results to original DataFrame
        df["Churn Prediction"] = predictions
        df["Churn Probability"] = probabilities

        st.success("✅ Churn Prediction Complete")
