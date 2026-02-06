import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to check if it is **Fraudulent or Legit**.")

st.divider()

# ---- USER INPUTS (FRONTEND) ----
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

st.subheader("PCA Features (V1 to V28)")
v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0)
    v_features.append(v)

# ---- PREDICTION ----
if st.button("🔍 Predict Transaction"):
    input_data = np.array([[time, *v_features, amount]])

    # Scale Time & Amount
    input_data[:, 0] = scaler.transform([[time]])[0][0]
    input_data[:, -1] = scaler.transform([[amount]])[0][0]

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
