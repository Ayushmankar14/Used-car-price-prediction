import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")

# Load model and features
@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("car_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_columns = load_model()

# -------------------- CSS Styling --------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        margin: 0;
        padding: 0;
        height: 100%;
    }
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1533473359331-0135ef1b58bf');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .banner {
        background-color: rgba(0, 0, 0, 0.7);
        color: #ffffff;
        padding: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .main-container {
        background-color: rgba(255,255,255,0.93);
        padding: 2rem;
        max-width: 700px;
        margin: 2rem auto;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0,0,0,0.25);
    }
    h1, h3 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Banner Message --------------------
st.markdown('<div class="banner">Welcome to the Used Car Resale Price Predictor</div>', unsafe_allow_html=True)

# -------------------- Main UI Container --------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1 style='color: #2F4F4F;'>🚘 Car Price Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the car details below to predict its selling price.")

# -------------------- Input Fields --------------------
year = st.slider("Year of Purchase", 2000, datetime.now().year, 2017)
car_age = datetime.now().year - year
present_price = st.number_input("Present Price (in Lakhs)", value=5.0)
kms_driven = st.number_input("Kilometers Driven", value=30000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [
    "First Owner", "Second Owner", "Third Owner",
    "Fourth & Above Owner", "Test Drive Car"
])

# -------------------- Input Data Builder --------------------
def create_input():
    row = {col: 0 for col in feature_columns}
    row["car_age"] = car_age
    row["present_price"] = present_price
    row["kms_driven"] = kms_driven

    fuel_col = f"fuel_type_{fuel_type.lower()}"
    seller_col = f"seller_type_{seller_type.lower().replace(' ', '_')}"
    transmission_col = f"transmission_{transmission.lower()}"
    owner_col = f"owner_{owner.lower().replace(' ', '_').replace('&', 'and')}"

    for col in [fuel_col, seller_col, transmission_col, owner_col]:
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[feature_columns]

# -------------------- Predict --------------------
if st.button("Predict Price 💸"):
    try:
        input_df = create_input()
        prediction = model.predict(input_df)[0]
        st.markdown(
            f"<h3 style='color: black;'>💰 Estimated Selling Price: "
            f"<span style='color:#D70000;'>₹ {prediction:,.2f} Lakhs</span></h3>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)
