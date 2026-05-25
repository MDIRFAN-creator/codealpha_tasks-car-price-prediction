import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/car_price_model.pkl")

st.title("Car Price Prediction App")

present_price = st.number_input("Present Price")
driven_kms = st.number_input("Driven Kilometers")
owner = st.number_input("Number of Previous Owners")

car_age = st.slider("Car Age", 0, 20)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

selling_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# Encoding
fuel_map = {"CNG": 0, "Diesel": 1, "Petrol": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Automatic": 0, "Manual": 1}

input_data = np.array([[
    present_price,
    driven_kms,
    fuel_map[fuel_type],
    seller_map[selling_type],
    trans_map[transmission],
    owner,
    car_age
]])

if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Car Price: ₹ {prediction[0]:.2f} Lakhs"
    )