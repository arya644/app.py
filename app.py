import streamlit as st
import joblib
import numpy as np

model=joblib.load('Stock_model.pkl')
scaler=joblib.load('scaler.pkl')

st.title("Stock price prediction")

st.write("Predict whether stock price will go UP or DOWN")

Open=st.number_input("Open")
High=st.number_input("High")
Low=st.number_input("Low")
volume=st.number_input("Total Trade Quantity")
turnover=st.number_input("Turnover(Lacs)")

if st.button("Predict"):
    input_data = np.array([[open_price, high_price, low_price, volume, turnover]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("📈 Stock Price will go UP")
    else:
        st.error("📉 Stock Price will go DOWN")



