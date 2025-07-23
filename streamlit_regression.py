import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load the encoders and scaler
with open('one_hot_Geo_reg.pkl', 'rb') as file:
    encoder_geo = pickle.load(file)

with open('scaler_reg.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder_Gender_reg.pkl', 'rb') as file:
    gender_map = pickle.load(file)

# Streamlit app UI
st.title('Estimated Salary Prediction App')
st.write('This app predicts the estimated salary of a customer based on their details.')

# User input fields
geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', gender_map.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])  # This must match the training column name
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Predict button
if st.button("Predict Estimated Salary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_map.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited],  # 🔁 Match the original training column
        'Geography': [geography]
    })

    # One-hot encode Geography
    geo_df = pd.DataFrame({'Geography': [geography]})
    geo_encoded = encoder_geo.transform(geo_df).toarray()
    geo_columns = [f"Geography_{cat}" for cat in encoder_geo.categories_[0]]
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

    # Combine encoded and numerical features
    input_data = input_data.drop('Geography', axis=1)
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    estimated_salary = prediction[0][0]

    # Display result
    st.metric("Estimated Salary (₹)", f"{estimated_salary:,.2f}")
