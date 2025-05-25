
#  Name: Mageshvari A		Batch ID:  CDS (2024-25) 15062024
#  Topic:  Ensemble Models

### Streamlit Code for Gradient Boost Model ###

import streamlit as st
import pandas as pd
import joblib  

# Load the trained Gradient Boosting model
model = joblib.load("gradientboost_model.pkl")

st.title("Cloth Company Sales Prediction App")

# Define input fields based on dataset features
CompPrice = st.number_input("Competitor Price", value=0.0)
Income = st.number_input("Income", value=0.0)
Advertising = st.number_input("Advertising", value=0.0)
Population = st.number_input("Population", value=0)
Price = st.number_input("Product Price", value=0.0)
ShelveLoc = st.selectbox("Shelf Location", [0, 1, 2])  # Encoded values for categories
Age = st.number_input("Age", value=0)
Education = st.number_input("Education", value=0)
Urban = st.selectbox("Urban", [0, 1])  # Encoded 0 or 1
US = st.selectbox("US", [0, 1])  # Encoded 0 or 1

# Create a DataFrame from inputs
user_data = pd.DataFrame({
    "CompPrice": [CompPrice],
    "Income": [Income],
    "Advertising": [Advertising],
    "Population": [Population],
    "Price": [Price],
    "ShelveLoc": [ShelveLoc],
    "Age": [Age],
    "Education": [Education],
    "Urban": [Urban],
    "US": [US]
})

# Make prediction
if st.button("Predict Sales"):
    prediction = model.predict(user_data)
    result = "High Sales" if prediction[0] == 1 else "Low Sales"
    st.write(f"### Prediction: {result}")
