# 🔮 Customer Churn Prediction using ANN

Predicting customer churn using an **Artificial Neural Network (ANN)** trained on banking customer data. This project helps companies proactively identify customers who are likely to leave and take preventive action.

---

## 🧠 Overview

Customer churn is a critical problem in many industries, especially in banking and telecom. This project builds a classification model using a **deep learning approach (ANN)** to predict whether a customer will churn based on historical features.

---

## 📁 Files & Directory Structure


graph TD
    A[📁 Churn_Prediction] --> B[📊 churn_modeling.csv<br>Dataset (raw input data)]
    A --> C[📓 churn_modeling.ipynb<br>Model Training & Evaluation]
    A --> D[📁 Preprocessing Artifacts]
    D --> D1[encoder.pkl]
    D --> D2[scaler.pkl]
    D --> D3[gender_map.pkl]
    A --> E[🧠 model.h5<br>Trained ANN Model (HDF5 format)]
    A --> F[🧪 requirements.txt<br>Project Dependencies]
    A --> G[🌐 app.py<br>Streamlit App for Prediction]
    A --> H[📄 README.md<br>Project Documentation]
