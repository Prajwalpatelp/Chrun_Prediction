import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle
import os

# Optional: Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('encoder.pkl', 'rb') as file:
    encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('gender_map.pkl', 'rb') as file:
    gender_map = pickle.load(file)

# Streamlit app UI
st.title('Customer Churn Prediction')
st.write('This app predicts the probability of a customer churning based on their details.')

# User input fields
geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', list(gender_map.keys()))
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_map[gender]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography]
    })

    # One-hot encode Geography
    geo_df = pd.DataFrame({'Geography': [geography]})
    geo_encoded = encoder_geo.transform(geo_df).toarray()

    # Manually construct column names
    geo_categories = encoder_geo.categories_[0]
    geo_columns = [f"Geography_{cat}" for cat in geo_categories]
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

    # Combine encoded and numerical features
    input_data = input_data.drop('Geography', axis=1)
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display results
    st.metric("Churn Probability (%)", f"{prediction_proba * 100:.2f}%")

    if prediction_proba > 0.5:
        st.error('⚠️ The customer is likely to churn.')
    else:
        st.success('✅ The customer is not likely to churn.')
