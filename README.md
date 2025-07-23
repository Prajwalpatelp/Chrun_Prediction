# 🔮 Customer Churn Prediction using ANN

Predicting customer churn using an **Artificial Neural Network (ANN)** trained on banking customer data. This project helps companies proactively identify customers who are likely to leave and take preventive action.

---

## 🧠 Overview

Customer churn is a critical problem in many industries, especially in banking and telecom. This project builds a classification model using a **deep learning approach (ANN)** to predict whether a customer will churn based on historical features.

---

## 📁 Files & Directory Structure


Churn_Prediction/
│
├── app.py # Streamlit app to run predictions interactively
├── churn_modeling.ipynb # Jupyter Notebook used for model training
├── churn_modeling.csv # Dataset used for training and evaluation
├── model.h5 # Trained ANN model saved in HDF5 format
├── encoder.pkl # OneHotEncoder for 'Geography'
├── scaler.pkl # StandardScaler for feature normalization
├── gender_map.pkl # Gender encoding map
├── requirements.txt # All project dependencies
└── README.md # Project documentation