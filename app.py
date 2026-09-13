from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

MODEL_PATH = Path(__file__).with_name("churn_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to estimate whether the customer is likely to churn. The prediction uses the same preprocessing and model saved during notebook training.")

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded. Ensure churn_pipeline.pkl is in the same folder as app.py.")
    st.exception(exc)
    st.stop()

with st.form("customer_form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        region = st.selectbox("Region", ["East", "North", "South", "West"])
        tenure_months = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=75.0, step=1.0)
        total_charges = st.number_input("Total charges", min_value=0.0, value=900.0, step=10.0)
        contract_type = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    with col2:
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment method", ["Electronic check", "Bank transfer", "Credit card", "Mailed check"])
        num_support_calls = st.number_input("Number of support calls", min_value=0, max_value=20, value=1, step=1)
        late_payments_last_year = st.number_input("Late payments in the last year", min_value=0, max_value=12, value=0, step=1)
        avg_monthly_usage_gb = st.number_input("Average monthly usage (GB)", min_value=0.0, value=200.0, step=10.0)

    submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "region": region,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "num_support_calls": num_support_calls,
        "late_payments_last_year": late_payments_last_year,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
    }])

    try:
        prediction = model.predict(customer)[0]
        if prediction == "Yes":
            st.error("Prediction: Yes — this customer is likely to churn.")
        else:
            st.success("Prediction: No — this customer is not likely to churn.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            class_probabilities = dict(zip(model.classes_, probabilities))
            churn_probability = float(class_probabilities.get("Yes", 0.0))
            st.metric("Churn probability", f"{churn_probability:.1%}")
            st.progress(churn_probability)
    except Exception as exc:
        st.error("Prediction failed. Please review the entered values and try again.")
        st.exception(exc)
