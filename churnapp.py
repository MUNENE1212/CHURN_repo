import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Telco Customer Churn Prediction App")
st.markdown("""
This application predicts whether a customer will churn based on their characteristics.
Enter the customer information below and click 'Predict Churn' to see the result.
""")

# Function to load the model
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_model.pkl' is in the same directory as this app.")
        return None

# Load the model
model = load_model()

# Create two columns for the form
col1, col2 = st.columns(2)

# Input form
with st.form("prediction_form"):
    st.subheader("Customer Information")
    
    # First column of inputs
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    # Second column of inputs
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Third row of inputs
    st.subheader("Contract and Payment Information")
    col3, col4 = st.columns(2)
    
    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * monthly_charges, step=100.0)
    
    # Submit button
    submit_button = st.form_submit_button("Predict Churn")

# If the form is submitted
if submit_button and model is not None:
    # Prepare the input data - create a DataFrame with the same structure as training data
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [str(total_charges)]  # Convert to string to match dataset format
    }
    
    # Create DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Display the input data
    st.subheader("Customer Profile")
    profile_df = pd.DataFrame.from_dict({k: v[0] for k, v in input_data.items()}, orient='index', columns=['Value'])
    st.dataframe(profile_df)
    
    try:
        # The pipeline handles preprocessing so we can directly use the input data
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Display prediction
        st.subheader("Prediction Results")
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            if prediction[0] == 'Yes':
                st.error("⚠️ This customer is likely to churn!")
                churn_prob_index = np.where(model.classes_ == 'Yes')[0][0]
                st.metric("Churn Probability", f"{probability[0][churn_prob_index]:.2%}")
            else:
                st.success("✅ This customer is likely to stay!")
                stay_prob_index = np.where(model.classes_ == 'No')[0][0]
                st.metric("Retention Probability", f"{probability[0][stay_prob_index]:.2%}")
        
        with col_pred2:
            # Visualization of probability
            fig, ax = plt.subplots(figsize=(4, 3))
            
            # Get the correct indices for 'Yes' and 'No' classes
            yes_idx = np.where(model.classes_ == 'Yes')[0][0] if 'Yes' in model.classes_ else 1
            no_idx = np.where(model.classes_ == 'No')[0][0] if 'No' in model.classes_ else 0
            
            sns.barplot(
                x=['Stay', 'Churn'], 
                y=[probability[0][no_idx], probability[0][yes_idx]], 
                palette=["green", "red"], 
                ax=ax
            )
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            st.pyplot(fig)
        
        # Show top factors influencing the prediction
        st.subheader("Customer Risk Analysis")
        st.markdown("""
        The likelihood of churn is influenced by various factors including:
        - Contract type (month-to-month contracts have higher churn)
        - Tenure (newer customers are more likely to churn)
        - Payment method (electronic checks show higher churn rates)
        - Internet service type (fiber optic customers show higher churn)
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Please check the input data format and ensure it matches what the model expects.")

# Additional information section
with st.expander("About This App"):
    st.markdown("""
    ### How to Use This App
    1. Enter the customer details in the form above
    2. Click the "Predict Churn" button
    3. View the prediction results and probability
    
    ### About the Model
    This model was trained on Telco customer data to predict whether customers will churn (cancel service) 
    or remain a customer. The prediction is based on various factors including demographics, 
    services subscribed to, contract terms, and payment information.
    
    The model uses a Logistic Regression algorithm with preprocessing for both numerical and categorical features.
    
    ### Important Note
    For this app to work correctly, you need to have the trained model file 'best_model.pkl' 
    in the same directory as this app.
    """)
