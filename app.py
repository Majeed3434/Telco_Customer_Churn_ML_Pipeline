import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📡 Telco Customer Churn Dashboard")
st.markdown("Predict if a customer will leave based on their profile and service usage.")

# 1. Load Models
@st.cache_resource
def load_models():
    return joblib.load('models_bundle.joblib')

models = load_models()

# 2. User Input Sidebar
st.sidebar.header("Customer Information")
def user_input_features():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = tenure * monthly_charges # Simple estimation for user convenience
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    # Matching the structure of the training columns
    data = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': internet, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': tech_support, 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 3. Model Selection and Prediction
model_choice = st.selectbox("Choose Model", list(models.keys()))
selected_model = models[model_choice]

if st.button("Predict Churn"):
    prediction = selected_model.predict(input_df)
    prob = selected_model.predict_proba(input_df)[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", "CHURN" if prediction[0] == 1 else "RETAINED")
    with col2:
        st.metric("Churn Probability", f"{prob:.2%}")

    # Visualization: Probability Gauge
    fig = px.pie(values=[prob, 1-prob], names=['Churn Risk', 'Retention Safety'], 
                 hole=.4, color_discrete_sequence=['red', 'green'])
    st.plotly_chart(fig)