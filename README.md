# 🔮 Customer Churn Prediction using ANN

Predicting customer churn using an **Artificial Neural Network (ANN)** trained on banking customer data. This project helps companies proactively identify customers who are likely to leave and take preventive action.

---

## 🧠 Overview

Customer churn is a critical problem in many industries, especially in banking and telecom. This project builds a classification model using a **deep learning approach (ANN)** to predict whether a customer will churn based on historical features.

---

## 📁 Files & Directory Structure


graph TD
    A[⬇️ Clone Repository] --> B[📦 Install Requirements<br>pip install -r requirements.txt]
    B --> C[📓 Run Notebook<br>churn_modeling.ipynb]
    C --> D[🧠 Train ANN Model<br>model.h5]
    D --> E[💾 Save Artifacts<br>encoder.pkl, scaler.pkl, gender_map.pkl]
    E --> F[🌐 Launch Streamlit App<br>python app.py]
    F --> G[📊 Make Predictions<br>Using Trained Model via UI]

