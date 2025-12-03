import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# Load Model, Scaler, and Feature Columns
# ============================================================

@st.cache_resource
def load_assets():
    model = pickle.load(open("xgb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_cols = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, feature_cols

model, scaler, feature_cols = load_assets()

# ============================================================
# UI
# ============================================================

st.title("Hotel Reservation Cancellation Predictor")
st.write("Enter the reservation details below to predict the likelihood of cancellation.")

# Collect inputs (only 4 are used by the model)
lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=400, value=30)
total_nights = st.number_input("Total Nights", min_value=1, max_value=30, value=2)
avg_price_per_room = st.number_input("Average Price per Room", min_value=20.0, max_value=500.0, value=120.0)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=1)

# ============================================================
# Build Input DataFrame (only 4 features)
# ============================================================

input_df = pd.DataFrame([{
    'lead_time': lead_time,
    'total_nights': total_nights,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests
}])

# Ensure correct ordering of columns
input_df = input_df[feature_cols]

# ============================================================
# Convert to numpy array (XGBoost expects NO column names)
# ============================================================

X = input_df.values.astype(np.float32)

# Scale using the stored scaler
scaled = scaler.transform(X)

# ============================================================
# Predict
# ============================================================

prediction = int(model.predict(scaled)[0])
prob = float(model.predict_proba(scaled)[0][1])

# ============================================================
# Display Results
# ============================================================

st.subheader("Prediction Results")

if prediction == 1:
    st.error(f"⚠️ This reservation is likely to be **CANCELED**.\n\n**Probability:** {prob:.2f}")
else:
    st.success(f"✅ This reservation is likely to be **NOT CANCELED**.\n\n**Probability:** {prob:.2f}")

st.write("Model: XGBoost (4-feature version) | Streamlit Deployment | Version 1.0")
