import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model pipeline
@st.cache_resource
def load_model():
    return joblib.load('wine_quality_model.pkl')

model = load_model()

st.title("Wine Quality Prediction")

st.write("""
Provide the input features below to predict the wine quality.
""")

# Define the input features based on the original dataset columns
numerical_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                  'pH', 'sulphates', 'alcohol']

# Type options
type_options = ['red', 'white']

# Collect user inputs
inputs = {}

for col in numerical_cols:
    if col == 'pH':
        inputs[col] = st.number_input(col, min_value=2.0, max_value=4.0, value=3.0, step=0.01, format="%.2f")
    elif col == 'density':
        inputs[col] = st.number_input(col, min_value=0.9, max_value=1.1, value=0.995, step=0.0001, format="%.4f")
    elif col == 'alcohol':
        inputs[col] = st.number_input(col, min_value=5.0, max_value=15.0, value=10.0, step=0.1, format="%.1f")
    else:
        inputs[col] = st.number_input(col, min_value=0.0, max_value=20.0, value=1.0, step=0.01, format="%.2f")

# Type input
wine_type = st.selectbox("Type", type_options)

# Prepare input DataFrame for prediction
input_df = pd.DataFrame([inputs])

# Add one-hot encoded Type column as per training: 'Type_White Wine'
input_df['Type_White Wine'] = 1 if wine_type == 'white' else 0

# For debugging: Show input dataframe
st.subheader("Input DataFrame Preview")
st.write(input_df)

if st.button("Predict Quality"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
