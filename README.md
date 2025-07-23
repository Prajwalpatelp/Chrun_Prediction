# 🔮 Customer Churn Prediction using ANN

Predicting customer churn using an **Artificial Neural Network (ANN)** trained on banking customer data. This project helps companies proactively identify customers who are likely to leave and take preventive action.

---

## 🧠 Overview

Customer churn is a critical problem in many industries, especially in banking and telecom. This project builds a classification model using a **deep learning approach (ANN)** to predict whether a customer will churn based on historical features.

---

## 📁 Files & Directory Structure


Churn_Prediction/
     │
     ├── 📊 churn_modeling.csv         ──▶ Dataset (raw input data)
     │
     ├── 📓 churn_modeling.ipynb       ──▶ Model Training & Evaluation Notebook
     │
     ├── 📁 encoder.pkl                ┐
     ├── 📁 scaler.pkl                 ├──▶ Preprocessing Artifacts
     ├── 📁 gender_map.pkl             ┘
     │
     ├── 🧠 model.h5                   ──▶ Trained ANN Model (HDF5 format)
     │
     ├── 🧪 requirements.txt           ──▶ Project Dependencies
     │
     ├── 🌐 app.py                     ──▶ Streamlit App for Prediction
     │
     └── 📄 README.md                  ──▶ Project Documentation
