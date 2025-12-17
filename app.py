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


if st.button("predict"):
  input.data=np.array([[Open,High,Low,volume,turnover]])
  input.scaler=scaler.transform(input_data)
  prediction=model.predict(input.scaler)


  if prediction[0]==1:
    st.write("Stock price will go UP")
    else:
      st.error("stock price will go DOWN")

