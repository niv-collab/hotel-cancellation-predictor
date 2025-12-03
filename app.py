import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# Load Model, Scaler, Feature Columns
# ============================================================

@st.cache_resource
def load_assets():
    model = pickle.load(open("xgb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, feature_columns

model, scaler, feature_columns = load_assets()


# ============================================================
# UI
# ============================================================

st.title("Hotel Reservation Cancellation Predictor")
st.write("Enter the reservation details below to predict the likelihood of cancellation.")

# -------------------------------
# User Inputs
# -------------------------------

lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=400, value=30)
total_nights = st.number_input("Total Nights", min_value=1, max_value=30, value=2)
avg_price_per_room = st.number_input("Average Price per Room", min_value=20.0, max_value=500.0, value=120.0)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=1)
required_car_parking_space = st.selectbox("Car Parking Required?", [0, 1])
repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])
no_of_children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=5, value=2)

type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
room_type_reserved = st.selectbox("Room Type Reserved", [
    "Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"
])
market_segment_type = st.selectbox("Market Segment", [
    "Online","Offline","Corporate","Complementary"
])


# ============================================================
# Build Input DataFrame
# ============================================================

input_dict = {
    'lead_time': lead_time,
    'total_nights': total_nights,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests,
    'required_car_parking_space': required_car_parking_space,
    'repeated_guest': repeated_guest,
    'no_of_children': no_of_children,
    'no_of_adults': no_of_adults,
    'type_of_meal_plan': type_of_meal_plan,
    'room_type_reserved': room_type_reserved,
    'market_segment_type': market_segment_type
}

input_df = pd.DataFrame([input_dict])

# Ensure consistent dtype for categorical columns
input_df['type_of_meal_plan'] = input_df['type_of_meal_plan'].astype(str)
input_df['room_type_reserved'] = input_df['room_type_reserved'].astype(str)
input_df['market_segment_type'] = input_df['market_segment_type'].astype(str)


# ============================================================
# Feature Engineering (MUST MATCH TRAINING)
# ============================================================

input_df['price_per_night'] = input_df['avg_price_per_room'] / input_df['total_nights']
input_df['many_special_requests'] = (input_df['no_of_special_requests'] >= 3).astype(int)
input_df['adults_x_nights'] = input_df['no_of_adults'] * input_df['total_nights']
input_df['children_x_nights'] = input_df['no_of_children'] * input_df['total_nights']
input_df['prior_cancellation_flag'] = 0
input_df['has_rebooking_history'] = 0
input_df['is_weekend_arrival'] = 0
input_df['arrival_day_of_week'] = 0


# ============================================================
# One-Hot Encode EXACTLY Like Training
# ============================================================

input_encoded = pd.get_dummies(input_df)

# Add missing columns from training
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Remove any extra columns
input_encoded = input_encoded[feature_columns]


# ============================================================
# Convert dtype & Scale
# ============================================================

input_encoded = input_encoded.astype(np.float32)
scaled = scaler.transform(input_encoded)


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

st.write("Model: XGBoost | Streamlit Deployment | Version 1.0")
