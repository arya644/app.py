import streamlit as st
import numpy as np
import joblib

model = joblib.load("stock_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📈 Stock Price Movement Predictor")

open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
volume = st.number_input("Total Trade Quantity", min_value=0.0)
turnover = st.number_input("Turnover (Lacs)", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[open_price, high_price, low_price, volume, turnover]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("📈 Stock Price will go UP")
    else:
        st.error("📉 Stock Price will go DOWN")

