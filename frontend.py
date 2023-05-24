import streamlit as st
import pandas as pd
import joblib

st.title("Shirt Size Prediction")
st.subheader("Find your perfect fit here!")

age = st.number_input("Enter your age:", 1, 117)
weight = st.number_input("Enter your weight:", 1, 136)
height = st.number_input("Enter your height:", 1, 193)

user_input = pd.DataFrame([[weight, age, height]], columns=["weight", "age", "height"])

model = joblib.load('model.pkl')

prediction = model.predict(user_input)

if st.button("Predict"):
    st.write("Your size is", prediction[0])

pd.show_versions()